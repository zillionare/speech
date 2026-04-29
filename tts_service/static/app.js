
// 选择目录（使用原生文件选择器）
function selectDirectory(type) {
    const input = document.createElement('input');
    input.type = 'file';
    input.webkitdirectory = true;
    input.directory = true;
    input.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            const path = e.target.files[0].path || e.target.files[0].webkitRelativePath.split('/')[0];
            document.getElementById(`config-${type}-path`).value = path;
        }
    });
    input.click();
}

// 保存配置到服务器
async function saveConfig() {
    const config = {
        voices_path: document.getElementById('config-voices-path').value,
        outputs_path: document.getElementById('config-outputs-path').value,
        default_voice: document.getElementById('config-default-voice-select').value,
        diffusion_steps: parseInt(document.getElementById('config-steps-slider').value),
        quantize_bits: parseInt(document.getElementById('config-quantize-select').value),
        cfg_scale: parseFloat(document.getElementById('config-cfg-slider').value),
        max_speech_tokens: parseInt(document.getElementById('config-max-tokens-slider').value),
        use_semantic: document.getElementById('config-semantic-toggle').checked,
        use_coreml_semantic: document.getElementById('config-coreml-toggle').checked,
        seed: parseInt(document.getElementById('config-seed-input').value),
        max_segment_chars: parseInt(document.getElementById('config-max-segment-slider').value),
        stereo: document.getElementById('config-stereo-toggle').checked,
        spatial_jitter: document.getElementById('config-jitter-toggle').checked,
        segment_gap_seconds: parseFloat(document.getElementById('config-segment-gap-slider').value),
        speaker_gap_seconds: parseFloat(document.getElementById('config-speaker-gap-slider').value),
    };
    
    try {
        await requestJson('/api/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        alert('配置已保存');
    } catch (err) {
        alert('保存失败: ' + err.message);
    }
}

// 重置默认配置
function resetConfig() {
    if (confirm('确定要重置为默认配置吗？')) {
        document.getElementById('config-voices-path').value = './voices';
        document.getElementById('config-outputs-path').value = './outputs';
        document.getElementById('config-steps-slider').value = 30;
        document.getElementById('config-steps-value').textContent = '30';
        document.getElementById('config-quantize-select').value = '8';
        document.getElementById('config-cfg-slider').value = '1.3';
        document.getElementById('config-cfg-value').textContent = '1.3';
        document.getElementById('config-max-tokens-slider').value = '200';
        document.getElementById('config-max-tokens-value').textContent = '200';
        document.getElementById('config-semantic-toggle').checked = true;
        document.getElementById('config-coreml-toggle').checked = false;
        document.getElementById('config-seed-input').value = '42';
        document.getElementById('config-max-segment-slider').value = '200';
        document.getElementById('config-max-segment-value').textContent = '200';
        document.getElementById('config-stereo-toggle').checked = false;
        document.getElementById('config-jitter-toggle').checked = false;
        document.getElementById('config-segment-gap-slider').value = '1.0';
        document.getElementById('config-segment-gap-value').textContent = '1.0s';
        document.getElementById('config-speaker-gap-slider').value = '1.0';
        document.getElementById('config-speaker-gap-value').textContent = '1.0s';
    }
}

// 绑定滑块事件
document.addEventListener('DOMContentLoaded', () => {
    // 扩散步数滑块
    const stepsSlider = document.getElementById('config-steps-slider');
    const stepsValue = document.getElementById('config-steps-value');
    if (stepsSlider && stepsValue) {
        stepsSlider.addEventListener('input', (e) => {
            stepsValue.textContent = e.target.value;
        });
    }
    
    // CFG滑块
    const cfgSlider = document.getElementById('config-cfg-slider');
    const cfgValue = document.getElementById('config-cfg-value');
    if (cfgSlider && cfgValue) {
        cfgSlider.addEventListener('input', (e) => {
            cfgValue.textContent = e.target.value;
        });
    }
    
    // 最大令牌滑块
    const maxTokensSlider = document.getElementById('config-max-tokens-slider');
    const maxTokensValue = document.getElementById('config-max-tokens-value');
    if (maxTokensSlider && maxTokensValue) {
        maxTokensSlider.addEventListener('input', (e) => {
            maxTokensValue.textContent = e.target.value;
        });
    }

    // 分段最大字符数滑块
    const maxSegmentSlider = document.getElementById('config-max-segment-slider');
    const maxSegmentValue = document.getElementById('config-max-segment-value');
    if (maxSegmentSlider && maxSegmentValue) {
        maxSegmentSlider.addEventListener('input', (e) => {
            maxSegmentValue.textContent = e.target.value;
        });
    }

    // 段落间隔滑块
    const segGapSlider = document.getElementById('config-segment-gap-slider');
    const segGapValue = document.getElementById('config-segment-gap-value');
    if (segGapSlider && segGapValue) {
        segGapSlider.addEventListener('input', (e) => {
            segGapValue.textContent = e.target.value + 's';
        });
    }

    // Speaker 切换间隔滑块
    const spkGapSlider = document.getElementById('config-speaker-gap-slider');
    const spkGapValue = document.getElementById('config-speaker-gap-value');
    if (spkGapSlider && spkGapValue) {
        spkGapSlider.addEventListener('input', (e) => {
            spkGapValue.textContent = e.target.value + 's';
        });
    }

    // 保存和重置按钮
    const saveBtn = document.getElementById('save-config');
    const resetBtn = document.getElementById('reset-config');
    if (saveBtn) saveBtn.addEventListener('click', saveConfig);
    if (resetBtn) resetBtn.addEventListener('click', resetConfig);
});

const state = {
    config: null,
    voices: [],
    history: [],
    engine: "local",
};

// 存储从目录选择器或文件选择器中暂定的音频文件，供上传使用
let pendingVoiceFiles = {
    audio: null,
};

function setText(id, value) {
    const node = document.getElementById(id);
    if (node) {
        node.textContent = value;
    }
}

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
                if (typeof payload.detail === "string") {
                    message = payload.detail;
                } else if (Array.isArray(payload.detail)) {
                    message = payload.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
                } else {
                    message = JSON.stringify(payload.detail);
                }
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

function setVoiceStatus(message, isError = false) {
    const el = document.getElementById("voice-status");
    if (!el) return;
    el.classList.toggle("error", isError);
    el.textContent = message;
}

function renderConfig() {
    if (!state.config) {
        return;
    }
    setText("config-model", state.config.model);
    setText("config-quantize", `${state.config.quantize_bits}-bit`);
    setText("config-default-voice", state.config.default_voice);
    setText("config-semantic-mode", state.config.use_coreml_semantic ? "CoreML" : "MLX");
    setText("sidebar-default-voice", state.config.default_voice);

    // Sliders and numeric inputs
    const stepsSlider = document.getElementById("config-steps-slider");
    if (stepsSlider) {
        stepsSlider.value = state.config.diffusion_steps;
        const stepsValue = document.getElementById("config-steps-value");
        if (stepsValue) stepsValue.textContent = String(state.config.diffusion_steps);
    }
    const cfgSlider = document.getElementById("config-cfg-slider");
    if (cfgSlider) {
        cfgSlider.value = state.config.cfg_scale;
        const cfgValue = document.getElementById("config-cfg-value");
        if (cfgValue) cfgValue.textContent = String(state.config.cfg_scale);
    }
    const maxTokensSlider = document.getElementById("config-max-tokens-slider");
    if (maxTokensSlider) {
        maxTokensSlider.value = state.config.max_speech_tokens;
        const maxTokensValue = document.getElementById("config-max-tokens-value");
        if (maxTokensValue) maxTokensValue.textContent = String(state.config.max_speech_tokens);
    }
    const seedInput = document.getElementById("config-seed-input");
    if (seedInput) seedInput.value = state.config.seed;

    const maxSegmentSlider = document.getElementById("config-max-segment-slider");
    if (maxSegmentSlider) {
        maxSegmentSlider.value = state.config.max_segment_chars;
        const maxSegmentValue = document.getElementById("config-max-segment-value");
        if (maxSegmentValue) maxSegmentValue.textContent = String(state.config.max_segment_chars);
    }

    const stereoToggle = document.getElementById("config-stereo-toggle");
    if (stereoToggle) stereoToggle.checked = state.config.stereo;

    const jitterToggle = document.getElementById("config-jitter-toggle");
    if (jitterToggle) jitterToggle.checked = state.config.spatial_jitter;

    const segGapSlider = document.getElementById("config-segment-gap-slider");
    if (segGapSlider) {
        segGapSlider.value = state.config.segment_gap_seconds;
        const segGapValue = document.getElementById("config-segment-gap-value");
        if (segGapValue) segGapValue.textContent = state.config.segment_gap_seconds.toFixed(1) + "s";
    }

    const spkGapSlider = document.getElementById("config-speaker-gap-slider");
    if (spkGapSlider) {
        spkGapSlider.value = state.config.speaker_gap_seconds;
        const spkGapValue = document.getElementById("config-speaker-gap-value");
        if (spkGapValue) spkGapValue.textContent = state.config.speaker_gap_seconds.toFixed(1) + "s";
    }

    // Selects and toggles
    const quantizeSelect = document.getElementById("config-quantize-select");
    if (quantizeSelect) quantizeSelect.value = String(state.config.quantize_bits);
    const semanticToggle = document.getElementById("config-semantic-toggle");
    if (semanticToggle) semanticToggle.checked = state.config.use_semantic;
    const coremlToggle = document.getElementById("config-coreml-toggle");
    if (coremlToggle) coremlToggle.checked = state.config.use_coreml_semantic;

    // Paths and default voice
    const voicesPath = document.getElementById("config-voices-path");
    if (voicesPath) voicesPath.value = state.config.voices_path || "./voices";
    const outputsPath = document.getElementById("config-outputs-path");
    if (outputsPath) outputsPath.value = state.config.outputs_path || "./outputs";
    const defaultVoiceSelect = document.getElementById("config-default-voice-select");
    if (defaultVoiceSelect) defaultVoiceSelect.value = state.config.default_voice;
}

function renderCounters() {
    setText("voice-count", String(state.voices.length));
    setText("history-count", String(state.history.length));
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

function renderDefaultVoiceSelect() {
    const select = document.getElementById("config-default-voice-select");
    const currentValue = select.value;
    select.innerHTML = "";

    if (state.voices.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "暂无声音样本，请先上传";
        select.appendChild(option);
        return;
    }

    for (const voice of state.voices) {
        const option = document.createElement("option");
        option.value = voice.speaker;
        option.textContent = voice.speaker;
        if (voice.speaker === currentValue || voice.is_default) {
            option.selected = true;
        }
        select.appendChild(option);
    }
}

function renderVoices() {
    const list = document.getElementById("voice-list");
    const template = document.getElementById("voice-item-template");
    list.innerHTML = "";

    if (state.voices.length === 0) {
        list.innerHTML = '<div class="result-card empty">当前还没有本地声音样本。</div>';
        renderCounters();
        renderVoiceOptions();
        return;
    }

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
            try {
                await requestJson(`/api/voices/${encodeURIComponent(voice.speaker)}`, { method: "DELETE" });
                setVoiceStatus(`声音 ${voice.speaker} 已删除`, false);
                await loadVoices();
            } catch (err) {
                setVoiceStatus(String(err), true);
            }
        });

        list.appendChild(node);
    }

    renderVoiceOptions();
    renderDefaultVoiceSelect();
    renderCounters();
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

function formatTimeLabel(isoString) {
    if (!isoString) return "";
    const d = new Date(isoString);
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    const isToday = d.getFullYear() === now.getFullYear() &&
                    d.getMonth() === now.getMonth() &&
                    d.getDate() === now.getDate();
    if (isToday) {
        return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }
    return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function renderLatest(record) {
    const card = document.getElementById("latest-result");
    card.classList.remove("empty", "error");
    const timeLabel = formatTimeLabel(record.created_at);
    const timeHtml = timeLabel ? `<span class="time-badge">${timeLabel}</span> · ` : "";
    card.innerHTML = `
    <div class="history-head">
      <div>
        <h3>最新生成: ${record.request_id}</h3>
        <p>${timeHtml}${record.generation_seconds.toFixed(2)}s 生成，音频时长 ${record.duration_seconds.toFixed(2)}s${record.segment_count > 1 ? `，分段 ${record.segment_count}` : ''}</p>
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

    if (state.history.length === 0) {
        const latest = document.getElementById("latest-result");
        latest.classList.add("empty");
        latest.classList.remove("error");
        latest.textContent = "尚未生成任何音频。";
        list.innerHTML = '<div class="result-card empty">最近输出为空，生成后会显示在这里。</div>';
        renderCounters();
        return;
    }

    for (const record of state.history) {
        const node = template.content.firstElementChild.cloneNode(true);
        node.querySelector(".history-id").textContent = record.request_id;
        const timeLabel = formatTimeLabel(record.created_at);
        const timePrefix = timeLabel ? `${timeLabel} · ` : "";
        node.querySelector(".history-meta").textContent = `${timePrefix}${record.generation_seconds.toFixed(2)}s 生成 · ${record.duration_seconds.toFixed(2)}s 音频 · ${record.output_format}${record.segment_count > 1 ? ' · 分段 ' + record.segment_count : ''}`;
        node.querySelector(".download-link").href = record.audio_url;
        node.querySelector(".history-audio").src = record.audio_url;
        node.querySelector(".history-text").textContent = record.input_text;
        renderResolutions(node.querySelector(".resolution-list"), record);
        list.appendChild(node);
    }

    if (state.history.length > 0) {
        renderLatest(state.history[0]);
    }

    renderCounters();
}

function initEngineToggle() {
    const localBtn = document.getElementById("engine-local");
    const remoteBtn = document.getElementById("engine-remote");
    const hint = document.getElementById("engine-hint");
    if (!localBtn || !remoteBtn) return;

    const update = () => {
        localBtn.classList.toggle("active", state.engine === "local");
        remoteBtn.classList.toggle("active", state.engine === "remote");
        if (hint) {
            hint.textContent = state.engine === "local"
                ? "模型参数仅影响本地引擎"
                : "使用远程 Qwen 引擎，模型参数被忽略";
        }
    };

    localBtn.addEventListener("click", () => {
        state.engine = "local";
        update();
    });
    remoteBtn.addEventListener("click", () => {
        state.engine = "remote";
        update();
    });

    // If server default is remote, reflect that
    if (state.config && state.config.use_remote_qwen) {
        state.engine = "remote";
    }
    update();
}

function initSettingsToggle() {
    const section = document.getElementById("system-settings");
    const btn = document.getElementById("toggle-settings");
    if (!section || !btn) return;

    const update = () => {
        const isCollapsed = section.classList.contains("collapsed");
        btn.textContent = isCollapsed ? "展开" : "收起";
    };

    btn.addEventListener("click", () => {
        section.classList.toggle("collapsed");
        update();
    });

    update();
}

async function loadConfig() {
    state.config = await requestJson("/api/config");
    renderConfig();
    initEngineToggle();
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
    const form = event.currentTarget;
    const submitButton = form.querySelector("button[type='submit']");
    const originalLabel = submitButton ? submitButton.textContent : null;

    const speaker = form.querySelector("[name='speaker']").value.trim();
    const transcript = form.querySelector("[name='transcript']").value.trim();

    if (!speaker) {
        setVoiceStatus("请填写 Speaker 名称", true);
        form.querySelector("[name='speaker']").focus();
        return;
    }

    let audioFile = pendingVoiceFiles.audio;
    if (!audioFile) {
        const audioInput = form.querySelector("[name='audio_file']");
        if (!audioInput.files || audioInput.files.length === 0) {
            setVoiceStatus("请选择音频文件", true);
            return;
        }
        const files = Array.from(audioInput.files);
        audioFile = files.find((f) => f.name.toLowerCase().endsWith(".wav"));
    }

    if (!audioFile) {
        setVoiceStatus("请至少选择一个 .wav 音频文件", true);
        return;
    }

    if (!transcript) {
        setVoiceStatus("请填写 Transcript，或同时选择同名的 .txt 文件", true);
        form.querySelector("[name='transcript']").focus();
        return;
    }

    try {
        if (submitButton) {
            submitButton.disabled = true;
            submitButton.textContent = "上传中...";
        }

        const formData = new FormData();
        formData.append("speaker", speaker);
        formData.append("audio_file", audioFile);
        formData.append("transcript", transcript);
        const overwriteCheckbox = form.querySelector("[name='overwrite']");
        if (overwriteCheckbox && overwriteCheckbox.checked) {
            formData.append("overwrite", "on");
        }

        const uploadResponse = await fetch("/api/voices", { method: "POST", body: formData });
        if (!uploadResponse.ok) {
            const text = await uploadResponse.text();
            throw new Error(text || `Upload failed: ${uploadResponse.status}`);
        }

        setVoiceStatus(`声音 ${speaker} 已上传，正在生成缓存（首次需要加载模型，请耐心等待）...`, false);

        // 自动预热缓存
        const cacheResponse = await fetch(`/api/voices/${encodeURIComponent(speaker)}/cache`, { method: "POST" });
        if (!cacheResponse.ok) {
            const text = await cacheResponse.text();
            throw new Error(`缓存生成失败: ${text || cacheResponse.status}`);
        }

        form.reset();
        document.getElementById("audio-selected-name").textContent = "未选择";
        pendingVoiceFiles = { audio: null };
        await loadVoices();
        setVoiceStatus(`声音 ${speaker} 已上传并缓存就绪。`, false);
    } catch (error) {
        setVoiceStatus(error.message, true);
    } finally {
        if (submitButton) {
            submitButton.disabled = false;
            if (originalLabel !== null) {
                submitButton.textContent = originalLabel;
            }
        }
    }
}

async function handlePruneOutputs() {
    const historyCount = state.history.length;
    if (historyCount <= 3) {
        alert(`当前只有 ${historyCount} 条生成记录，不需要清理。`);
        return;
    }

    const toDeleteCount = historyCount - 3;
    const message = (
        `清理旧输出将只保留最近 3 条生成的音频文件，其余 ${toDeleteCount} 条将被永久删除。\n\n` +
        `此操作只删除 outputs/ 目录中的生成结果，不会影响声音库中的样本。\n\n` +
        `此操作不可撤销，是否继续？`
    );

    if (!confirm(message)) {
        return;
    }

    try {
        const result = await requestJson("/api/outputs/prune", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ keep_count: 3 }),
        });
        await loadHistory();
        setStatus(`已清理旧输出：删除 ${result.deleted.length} 条，保留 ${result.kept.length} 条。`, false);
    } catch (error) {
        setStatus(error.message, true);
    }
}

async function streamGenerate(payload, onProgress, onComplete, onError) {
    const response = await fetch("/api/generate/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(text || `Request failed: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (let i = 0; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            if (line.startsWith("event:")) {
                const eventType = line.slice(6).trim();
                const dataLine = lines[i + 1];
                if (dataLine && dataLine.trim().startsWith("data:")) {
                    const raw = dataLine.trim().slice(5).trim();
                    try {
                        const data = JSON.parse(raw);
                        if (eventType === "progress") onProgress(data);
                        else if (eventType === "complete") onComplete(data);
                        else if (eventType === "error") onError(data.message);
                    } catch {
                        // ignore malformed JSON
                    }
                    i++;
                }
            }
        }
    }
}

function setProgressVisible(visible, current, total, text) {
    const wrap = document.getElementById("generate-progress");
    const fill = document.getElementById("generate-progress-fill");
    const label = document.getElementById("generate-progress-text");
    if (!wrap || !fill || !label) return;
    wrap.hidden = !visible;
    if (visible) {
        const pct = total > 0 ? (current / total) * 100 : 0;
        fill.style.width = pct + "%";
        label.textContent = `正在生成第 ${current}/${total} 段: ${text}`;
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

    const rawText = document.getElementById("generate-text").value;
    const cleanedText = stripMarkdown(rawText);
    document.getElementById("generate-text").value = cleanedText;
    const payload = {
        text: cleanedText,
        output_format: form.output_format.value,
        voice: document.getElementById("preferred-voice").value || null,
        voice_mapping: parseJsonOrEmpty(document.getElementById("voice-mapping").value),
        engine: state.engine,
    };

    const maxChars = state.config ? state.config.max_segment_chars : 200;
    const useStream = payload.text.length > maxChars;

    try {
        if (useStream) {
            setStatus("", false);
            setProgressVisible(true, 0, 1, "准备中...");
            await streamGenerate(
                payload,
                (data) => {
                    setProgressVisible(true, data.current, data.total, data.text);
                },
                (record) => {
                    setProgressVisible(false, 0, 0, "");
                    state.history.unshift(record);
                    renderHistory();
                },
                (msg) => {
                    setProgressVisible(false, 0, 0, "");
                    setStatus(msg, true);
                }
            );
        } else {
            setStatus("正在生成，请等待...", false);
            const record = await requestJson("/api/generate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            state.history.unshift(record);
            renderHistory();
        }
    } catch (error) {
        setProgressVisible(false, 0, 0, "");
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

function initNavigation() {
    const links = Array.from(document.querySelectorAll(".nav-link[data-section]"));
    const sections = links
        .map((link) => document.getElementById(link.dataset.section))
        .filter(Boolean);

    const activateSection = (sectionId) => {
        for (const link of links) {
            link.classList.toggle("active", link.dataset.section === sectionId);
        }
    };

    for (const link of links) {
        link.addEventListener("click", () => activateSection(link.dataset.section));
    }

    if (!("IntersectionObserver" in window) || sections.length === 0) {
        return;
    }

    const observer = new IntersectionObserver(
        (entries) => {
            const visible = entries
                .filter((entry) => entry.isIntersecting)
                .sort((left, right) => right.intersectionRatio - left.intersectionRatio)[0];
            if (visible) {
                activateSection(visible.target.id);
            }
        },
        {
            rootMargin: "-25% 0px -55% 0px",
            threshold: [0.2, 0.45, 0.7],
        },
    );

    for (const section of sections) {
        observer.observe(section);
    }
}

async function loadVoiceFromDirectory(dirHandle) {
    const entries = {};
    for await (const [name, handle] of dirHandle.entries()) {
        if (handle.kind === "file") {
            entries[name.toLowerCase()] = handle;
        }
    }

    const wavNames = Object.keys(entries).filter((n) => n.endsWith(".wav"));
    if (wavNames.length === 0) {
        setVoiceStatus("目录中没有找到 .wav 文件", true);
        pendingVoiceFiles = { audio: null };
        return;
    }

    let wavName;
    if (wavNames.length === 1) {
        wavName = wavNames[0];
    } else {
        const speakers = wavNames.map((n) => n.slice(0, -4)).join(", ");
        const choice = prompt(`目录中有多个音频文件，请输入 Speaker 名称：\n可选：${speakers}`);
        if (!choice) {
            pendingVoiceFiles = { audio: null };
            return;
        }
        wavName = (choice + ".wav").toLowerCase();
        if (!entries[wavName]) {
            setVoiceStatus(`目录中未找到 ${choice}.wav`, true);
            pendingVoiceFiles = { audio: null };
            return;
        }
    }

    const speaker = wavName.slice(0, -4);
    const wavHandle = entries[wavName];
    const wavFile = await wavHandle.getFile();

    const form = document.getElementById("voice-upload-form");
    const speakerInput = form.querySelector("[name='speaker']");
    const nameLabel = document.getElementById("audio-selected-name");

    speakerInput.value = speaker;

    const txtName = (speaker + ".txt").toLowerCase();
    let transcript = "";
    if (entries[txtName]) {
        const txtFile = await entries[txtName].getFile();
        transcript = await txtFile.text();
        form.querySelector("[name='transcript']").value = transcript;
        nameLabel.textContent = `${wavFile.name}, ${speaker}.txt`;
        setVoiceStatus(`已从目录读取 ${wavFile.name} 和 ${speaker}.txt`, false);
    } else {
        nameLabel.textContent = wavFile.name;
        setVoiceStatus(`已选择 ${wavFile.name}，未找到同名 .txt，请手动输入 transcript`, false);
    }

    // 将 File System Access API 的惰性 File 转为内存中的标准 File，
    // 避免 fetch/FormData 序列化时卡住
    const arrayBuffer = await wavFile.arrayBuffer();
    const audioFile = new File([arrayBuffer], wavFile.name, { type: wavFile.type || "audio/wav" });
    pendingVoiceFiles = { audio: audioFile };
}

function handleAudioFileChange(event) {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    // 找 .wav 文件
    const wavFile = files.find((f) => f.name.toLowerCase().endsWith(".wav"));
    if (!wavFile) {
        setVoiceStatus("请至少选择一个 .wav 音频文件", true);
        event.target.value = "";
        pendingVoiceFiles = { audio: null };
        return;
    }

    const form = document.getElementById("voice-upload-form");
    const speakerInput = form.querySelector("[name='speaker']");
    const nameLabel = document.getElementById("audio-selected-name");
    const speaker = wavFile.name.slice(0, -4);
    if (!speakerInput.value) {
        speakerInput.value = speaker;
    }

    // 找同名的 .txt（不区分大小写）
    const txtFile = files.find(
        (f) => f.name.toLowerCase() === (speaker + ".txt").toLowerCase(),
    );

    if (txtFile) {
        const reader = new FileReader();
        reader.onload = (e) => {
            form.querySelector("[name='transcript']").value = e.target.result;
            pendingVoiceFiles = { audio: wavFile };
            nameLabel.textContent = `${wavFile.name}, ${txtFile.name}`;
            setVoiceStatus(`已选择 ${wavFile.name} 并读取 ${txtFile.name}`, false);
        };
        reader.readAsText(txtFile);
    } else {
        pendingVoiceFiles = { audio: wavFile };
        nameLabel.textContent = wavFile.name;
        setVoiceStatus(`已选择 ${wavFile.name}，未找到同名 .txt，请手动输入 transcript`, false);
    }
}

async function handleAudioSelect() {
    if (window.showDirectoryPicker) {
        try {
            const dirHandle = await window.showDirectoryPicker();
            await loadVoiceFromDirectory(dirHandle);
            return;
        } catch {
            return;
        }
    }
    const input = document.getElementById("audio-file-input");
    if (input) input.click();
}

function stripMarkdown(text) {
    if (typeof text !== "string") return "";
    // Decode common HTML entities in case content was escaped
    text = text.replace(/&lt;/g, "<")
               .replace(/&gt;/g, ">")
               .replace(/&amp;/g, "&")
               .replace(/&quot;/g, '"')
               .replace(/&#39;/g, "'");
    // Remove YAML frontmatter at the very beginning
    text = text.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n?/, "");
    // Remove HTML comments
    text = text.replace(/<!--[\s\S]*?-->/g, "");
    // Remove Markdown headers (lines starting with # or fullwidth ＃)
    text = text.replace(/^[ \t]*[#＃]+\s+.*$/gm, "");
    // Collapse multiple blank lines into one
    text = text.replace(/\n{3,}/g, "\n\n");
    console.log("[stripMarkdown] output:", JSON.stringify(text.trim()));
    return text.trim();
}

function handleMarkdownFileChange(event) {
    const files = event.target.files;
    if (!files || files.length === 0) return;

    const file = files[0];
    const reader = new FileReader();
    reader.onload = (e) => {
        const raw = e.target.result;
        const cleaned = stripMarkdown(raw);
        document.getElementById("generate-text").value = cleaned;
        document.getElementById("markdown-selected-name").textContent = file.name;
    };
    reader.readAsText(file);
}

function handleMarkdownSelect() {
    const input = document.getElementById("markdown-file-input");
    if (input) input.click();
}

async function bootstrap() {
    initNavigation();
    initSettingsToggle();
    document.getElementById("voice-upload-form").addEventListener("submit", handleVoiceUpload);
    const audioInput = document.getElementById("audio-file-input");
    if (audioInput) audioInput.addEventListener("change", handleAudioFileChange);
    const audioBtn = document.getElementById("audio-select-btn");
    if (audioBtn) audioBtn.addEventListener("click", handleAudioSelect);
    const markdownInput = document.getElementById("markdown-file-input");
    if (markdownInput) markdownInput.addEventListener("change", handleMarkdownFileChange);
    const markdownBtn = document.getElementById("markdown-select-btn");
    if (markdownBtn) markdownBtn.addEventListener("click", handleMarkdownSelect);
    document.getElementById("generate-form").addEventListener("submit", handleGenerate);
    document.getElementById("refresh-voices").addEventListener("click", loadVoices);
    const pruneBtn = document.getElementById("prune-outputs");
    if (pruneBtn) pruneBtn.addEventListener("click", handlePruneOutputs);
    await Promise.all([loadConfig(), loadVoices(), loadHistory()]);
    initPodcastEditor();
}

bootstrap().catch((error) => setStatus(error.message, true));

// ── Podcast Editor ──────────────────────────────────────

const podcastState = {
    projects: [],
    currentProject: null,
    selectedSegmentIndex: -1,
};

function setPodcastStatus(msg, isError = false) {
    const el = document.getElementById("podcast-status");
    if (!el) return;
    el.textContent = msg;
    el.style.color = isError ? "var(--danger)" : "var(--muted)";
}

function setSegmentStatus(msg, isError = false) {
    const el = document.getElementById("podcast-seg-status");
    if (!el) return;
    el.textContent = msg;
    el.style.color = isError ? "var(--danger)" : "var(--muted)";
}

async function loadPodcastProjects() {
    try {
        const data = await requestJson("/api/podcasts");
        podcastState.projects = data.podcasts || [];
        renderPodcastProjects();
    } catch (err) {
        setPodcastStatus(String(err), true);
    }
}

function renderPodcastProjects() {
    const container = document.getElementById("podcast-project-list");
    if (!container) return;
    container.innerHTML = "";
    if (podcastState.projects.length === 0) {
        container.innerHTML = '<p class="field-hint">暂无播客项目</p>';
        return;
    }
    for (const proj of podcastState.projects) {
        const div = document.createElement("div");
        div.className = "podcast-list-item" + (podcastState.currentProject?.id === proj.id ? " active" : "");
        const segDone = proj.segments.filter((s) => s.status === "generated").length;
        div.innerHTML = `<div class="title">${escapeHtml(proj.title)}</div><div class="meta">${segDone}/${proj.segments.length} 段</div>`;
        div.addEventListener("click", () => selectPodcastProject(proj.id));
        container.appendChild(div);
    }
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

async function selectPodcastProject(projectId) {
    try {
        const proj = await requestJson(`/api/podcasts/${projectId}`);
        podcastState.currentProject = proj;
        podcastState.selectedSegmentIndex = -1;
        document.getElementById("podcast-editor").hidden = false;
        renderPodcastProjects();
        renderPodcastSegments();
        hideSegmentEditor();
    } catch (err) {
        setPodcastStatus(String(err), true);
    }
}

function renderPodcastSegments() {
    const container = document.getElementById("podcast-segments-list");
    if (!container || !podcastState.currentProject) return;
    container.innerHTML = "";
    for (const seg of podcastState.currentProject.segments) {
        const div = document.createElement("div");
        div.className = "podcast-segment-row" + (podcastState.selectedSegmentIndex === seg.index ? " active" : "");
        const metaParts = [seg.speaker || "默认"];
        if (seg.voice_ref && seg.voice_ref !== seg.speaker) metaParts.push(seg.voice_ref);
        div.innerHTML = `
            <div class="seg-index">${seg.index + 1}</div>
            <button class="seg-play" data-index="${seg.index}" title="播放">▶</button>
            <div class="seg-text">${escapeHtml(seg.text)}</div>
            <div class="seg-meta">${metaParts.join(" ")}</div>
            <span class="seg-status ${seg.status}">${seg.status === "generated" ? "已生成" : seg.status === "pending" ? "待生成" : "错误"}</span>
        `;
        div.addEventListener("click", (e) => {
            if (e.target.classList.contains("seg-play")) {
                playPodcastSegment(seg.index);
            } else {
                selectPodcastSegment(seg.index);
            }
        });
        container.appendChild(div);
    }
}

function playPodcastSegment(index) {
    const proj = podcastState.currentProject;
    if (!proj) return;
    const seg = proj.segments[index];
    if (!seg || !seg.audio_filename) return;
    const player = document.getElementById("podcast-seg-player");
    if (player) {
        player.src = `/api/podcasts/${proj.id}/audio/${seg.audio_filename}`;
        player.play();
    }
}

function selectPodcastSegment(index) {
    podcastState.selectedSegmentIndex = index;
    renderPodcastSegments();
    showSegmentEditor(index);
}

function hideSegmentEditor() {
    const editor = document.getElementById("podcast-segment-editor");
    if (editor) editor.hidden = true;
}

function showSegmentEditor(index) {
    const proj = podcastState.currentProject;
    if (!proj) return;
    const seg = proj.segments[index];
    if (!seg) return;

    const editor = document.getElementById("podcast-segment-editor");
    editor.hidden = false;

    document.getElementById("podcast-seg-text").value = seg.text;

    // Populate speaker selector
    const speakerSel = document.getElementById("podcast-seg-speaker");
    speakerSel.innerHTML = "";
    const baseSpeakers = new Set();
    for (const v of state.voices) {
        if (!v.is_tone_voice) baseSpeakers.add(v.speaker);
    }
    for (const spk of Array.from(baseSpeakers).sort()) {
        const opt = document.createElement("option");
        opt.value = spk;
        opt.textContent = spk;
        opt.selected = spk === seg.speaker;
        speakerSel.appendChild(opt);
    }

    // Populate voice-ref selector
    updatePodcastVoiceRefSelector(seg.speaker, seg.voice_ref);

    // Set player src if audio exists
    const player = document.getElementById("podcast-seg-player");
    if (seg.audio_filename) {
        player.src = `/api/podcasts/${proj.id}/audio/${seg.audio_filename}`;
    } else {
        player.src = "";
    }
}

function updatePodcastVoiceRefSelector(speaker, selectedRef) {
    const sel = document.getElementById("podcast-seg-voice-ref");
    sel.innerHTML = "";
    // Base voice
    const baseOpt = document.createElement("option");
    baseOpt.value = speaker;
    baseOpt.textContent = `${speaker} (默认)`;
    baseOpt.selected = selectedRef === speaker || !selectedRef;
    sel.appendChild(baseOpt);
    // Tone voices for this speaker
    const toneVoices = state.voices.filter((v) => v.is_tone_voice && v.base_speaker === speaker);
    for (const tv of toneVoices) {
        const opt = document.createElement("option");
        opt.value = tv.speaker;
        opt.textContent = tv.display_name;
        opt.selected = tv.speaker === selectedRef;
        sel.appendChild(opt);
    }
}

async function savePodcastSegment() {
    const proj = podcastState.currentProject;
    if (!proj || podcastState.selectedSegmentIndex < 0) return;
    const idx = podcastState.selectedSegmentIndex;
    const payload = {
        text: document.getElementById("podcast-seg-text").value,
        speaker: document.getElementById("podcast-seg-speaker").value,
        voice_ref: document.getElementById("podcast-seg-voice-ref").value,
    };
    try {
        const updated = await requestJson(`/api/podcasts/${proj.id}/segments/${idx}`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        podcastState.currentProject = updated;
        renderPodcastSegments();
        setSegmentStatus("已保存");
    } catch (err) {
        setSegmentStatus(String(err), true);
    }
}

async function regeneratePodcastSegment() {
    const proj = podcastState.currentProject;
    if (!proj || podcastState.selectedSegmentIndex < 0) return;
    const idx = podcastState.selectedSegmentIndex;
    setSegmentStatus("生成中...");
    try {
        const updated = await requestJson(`/api/podcasts/${proj.id}/segments/${idx}/regenerate`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ engine: state.engine }),
        });
        podcastState.currentProject = updated;
        renderPodcastSegments();
        showSegmentEditor(idx);
        setSegmentStatus("生成完成");
    } catch (err) {
        setSegmentStatus(String(err), true);
    }
}

async function generateAllPodcastSegments() {
    const proj = podcastState.currentProject;
    if (!proj) return;
    setPodcastStatus("正在逐段生成...");
    try {
        const updated = await requestJson(`/api/podcasts/${proj.id}/generate-all`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ engine: state.engine }),
        });
        podcastState.currentProject = updated;
        renderPodcastSegments();
        setPodcastStatus("全部生成完成");
    } catch (err) {
        setPodcastStatus(String(err), true);
    }
}

async function mergePodcast() {
    const proj = podcastState.currentProject;
    if (!proj) return;
    setPodcastStatus("正在合并...");
    try {
        const result = await requestJson(`/api/podcasts/${proj.id}/merge`, { method: "POST" });
        setPodcastStatus(`合并完成: ${result.filename}`);
        // Open merged audio in new tab
        window.open(result.audio_url, "_blank");
    } catch (err) {
        setPodcastStatus(String(err), true);
    }
}

function initPodcastEditor() {
    // New podcast modal
    const newBtn = document.getElementById("podcast-new-btn");
    const modal = document.getElementById("podcast-new-modal");
    const createBtn = document.getElementById("podcast-new-create-btn");
    const cancelBtn = document.getElementById("podcast-new-cancel-btn");

    if (newBtn) newBtn.addEventListener("click", () => { modal.hidden = false; });
    if (cancelBtn) cancelBtn.addEventListener("click", () => { modal.hidden = true; });
    if (createBtn) {
        createBtn.addEventListener("click", async () => {
            const title = document.getElementById("podcast-new-title").value.trim();
            const text = document.getElementById("podcast-new-text").value.trim();
            if (!title || !text) { alert("请填写标题和文本"); return; }
            try {
                const proj = await requestJson("/api/podcasts", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ title, text }),
                });
                modal.hidden = true;
                document.getElementById("podcast-new-title").value = "";
                document.getElementById("podcast-new-text").value = "";
                await loadPodcastProjects();
                await selectPodcastProject(proj.id);
            } catch (err) {
                alert(String(err));
            }
        });
    }

    // Toolbar buttons
    const genAllBtn = document.getElementById("podcast-generate-all-btn");
    if (genAllBtn) genAllBtn.addEventListener("click", generateAllPodcastSegments);
    const mergeBtn = document.getElementById("podcast-merge-btn");
    if (mergeBtn) mergeBtn.addEventListener("click", mergePodcast);

    // Segment editor buttons
    const saveSegBtn = document.getElementById("podcast-seg-save-btn");
    if (saveSegBtn) saveSegBtn.addEventListener("click", savePodcastSegment);
    const regenSegBtn = document.getElementById("podcast-seg-regen-btn");
    if (regenSegBtn) regenSegBtn.addEventListener("click", regeneratePodcastSegment);

    // Speaker change updates voice-ref dropdown
    const speakerSel = document.getElementById("podcast-seg-speaker");
    if (speakerSel) {
        speakerSel.addEventListener("change", () => {
            updatePodcastVoiceRefSelector(speakerSel.value, "");
        });
    }

    // Load initial list
    loadPodcastProjects();
}

