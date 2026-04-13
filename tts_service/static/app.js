const state = {
    config: null,
    voices: [],
    history: [],
};

function parseJsonOrEmpty(text) {
    const value = text.trim();
    if (!value) {
        return {};
    }
    return JSON.parse(value);
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, options);
    if (!response.ok) {
        let message = `Request failed: ${response.status}`;
        try {
            const payload = await response.json();
            if (payload.detail) {
                message = payload.detail;
            }
        } catch {
            // Ignore parse errors.
        }
        throw new Error(message);
    }
    return response.json();
}

function setStatus(message, isError = false) {
    const card = document.getElementById("latest-result");
    card.classList.toggle("error", isError);
    card.classList.remove("empty");
    card.innerHTML = `<p>${message}</p>`;
}

function renderConfig() {
    if (!state.config) {
        return;
    }
    document.getElementById("config-model").textContent = state.config.model;
    document.getElementById("config-quantize").textContent = `${state.config.quantize_bits}-bit`;
    document.getElementById("config-default-voice").textContent = state.config.default_voice;
    document.getElementById("config-steps").textContent = String(state.config.diffusion_steps);
}

function renderVoiceOptions() {
    const select = document.getElementById("preferred-voice");
    select.innerHTML = "";
    for (const voice of state.voices) {
        const option = document.createElement("option");
        option.value = voice.speaker;
        option.textContent = voice.speaker;
        if (voice.is_default) {
            option.selected = true;
        }
        select.appendChild(option);
    }
}

function renderVoices() {
    const list = document.getElementById("voice-list");
    const template = document.getElementById("voice-item-template");
    list.innerHTML = "";

    for (const voice of state.voices) {
        const node = template.content.firstElementChild.cloneNode(true);
        node.querySelector(".voice-name").textContent = voice.speaker;
        node.querySelector(".voice-meta").textContent = voice.transcript_preview || "没有 transcript。";
        node.querySelector(".voice-audio").src = voice.audio_url;
        node.querySelector(".voice-transcript").value = voice.transcript;

        const defaultPill = node.querySelector(".default-pill");
        defaultPill.hidden = !voice.is_default;

        const cachePill = node.querySelector(".cache-pill");
        cachePill.textContent = voice.cache_ready ? "缓存已就绪" : "未缓存";
        cachePill.classList.toggle("ready", voice.cache_ready);

        node.querySelector(".save-transcript").addEventListener("click", async () => {
            const transcript = node.querySelector(".voice-transcript").value;
            await requestJson(`/api/voices/${encodeURIComponent(voice.speaker)}/transcript`, {
                method: "PUT",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ transcript }),
            });
            await loadVoices();
        });

        node.querySelector(".warm-cache").addEventListener("click", async () => {
            await requestJson(`/api/voices/${encodeURIComponent(voice.speaker)}/cache`, { method: "POST" });
            await loadVoices();
        });

        node.querySelector(".delete-voice").addEventListener("click", async () => {
            if (!confirm(`删除声音 ${voice.speaker}？`)) {
                return;
            }
            await requestJson(`/api/voices/${encodeURIComponent(voice.speaker)}`, { method: "DELETE" });
            await loadVoices();
        });

        list.appendChild(node);
    }

    renderVoiceOptions();
}

function renderResolutions(container, record) {
    container.innerHTML = "";
    for (const item of record.resolved_speakers) {
        const div = document.createElement("div");
        div.className = "resolution-pill";
        div.textContent = item.used_default
            ? `${item.requested_name} -> ${item.resolved_voice} (default)`
            : `${item.requested_name} -> ${item.resolved_voice}`;
        container.appendChild(div);
    }
}

function renderLatest(record) {
    const card = document.getElementById("latest-result");
    card.classList.remove("empty", "error");
    card.innerHTML = `
    <div class="history-head">
      <div>
        <h3>最新生成: ${record.request_id}</h3>
        <p>${record.generation_seconds.toFixed(2)}s 生成，音频时长 ${record.duration_seconds.toFixed(2)}s</p>
      </div>
      <a class="download-link" href="${record.audio_url}" target="_blank" rel="noopener">下载</a>
    </div>
    <audio controls preload="none" src="${record.audio_url}"></audio>
    <p class="history-text">${record.input_text}</p>
    <div class="resolution-list" id="latest-resolution-list"></div>
  `;
    renderResolutions(document.getElementById("latest-resolution-list"), record);
}

function renderHistory() {
    const list = document.getElementById("generation-history");
    const template = document.getElementById("history-item-template");
    list.innerHTML = "";

    for (const record of state.history) {
        const node = template.content.firstElementChild.cloneNode(true);
        node.querySelector(".history-id").textContent = record.request_id;
        node.querySelector(".history-meta").textContent = `${record.generation_seconds.toFixed(2)}s 生成 · ${record.duration_seconds.toFixed(2)}s 音频 · ${record.output_format}`;
        node.querySelector(".download-link").href = record.audio_url;
        node.querySelector(".history-audio").src = record.audio_url;
        node.querySelector(".history-text").textContent = record.input_text;
        renderResolutions(node.querySelector(".resolution-list"), record);
        list.appendChild(node);
    }

    if (state.history.length > 0) {
        renderLatest(state.history[0]);
    }
}

async function loadConfig() {
    state.config = await requestJson("/api/config");
    renderConfig();
}

async function loadVoices() {
    const payload = await requestJson("/api/voices");
    state.voices = payload.voices;
    renderVoices();
}

async function loadHistory() {
    state.history = await requestJson("/api/generations");
    renderHistory();
}

async function handleVoiceUpload(event) {
    event.preventDefault();
    try {
        const form = event.currentTarget;
        const formData = new FormData(form);
        await fetch("/api/voices", { method: "POST", body: formData }).then(async (response) => {
            if (!response.ok) {
                const text = await response.text();
                throw new Error(text || `Upload failed: ${response.status}`);
            }
        });
        form.reset();
        await loadVoices();
        setStatus("声音已上传并写入 voices/。", false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

async function handleGenerate(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const submitButton = form.querySelector("button[type='submit'], button:not([type]), input[type='submit']");
    const originalLabel = submitButton && "textContent" in submitButton ? submitButton.textContent : null;

    if (submitButton) {
        submitButton.disabled = true;
        if (originalLabel !== null) {
            submitButton.textContent = "生成中...";
        }
    }

    try {
        const payload = {
            text: document.getElementById("generate-text").value,
            output_format: form.output_format.value,
            voice: document.getElementById("preferred-voice").value || null,
            voice_mapping: parseJsonOrEmpty(document.getElementById("voice-mapping").value),
        };
        setStatus("正在生成，请等待...", false);
        const record = await requestJson("/api/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        state.history.unshift(record);
        renderHistory();
    } catch (error) {
        setStatus(error.message, true);
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            if (originalLabel !== null) {
                submitButton.textContent = originalLabel;
            }
        }
    }
}

async function bootstrap() {
    document.getElementById("voice-upload-form").addEventListener("submit", handleVoiceUpload);
    document.getElementById("generate-form").addEventListener("submit", handleGenerate);
    document.getElementById("refresh-voices").addEventListener("click", loadVoices);
    await Promise.all([loadConfig(), loadVoices(), loadHistory()]);
}

bootstrap().catch((error) => setStatus(error.message, true));
