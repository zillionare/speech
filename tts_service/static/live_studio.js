/**
 * Live Studio frontend — independent online broadcast.
 *
 * Protocol (server -> client):
 *   {type:"state", state:"AI_SPEAKING"|"RECORDING"|"FINISHED"|"ERROR"}
 *   {type:"segment_start", index, source:"tts"|"live", speaker, text, resolved_voice}
 *   {type:"audio", index, sample_rate}  -> followed by binary WAV bytes
 *   {type:"record_start", index, target_text}
 *   {type:"restart", to_index, from_index}
 *   {type:"segment_done", index, source, audio_url}
 *   {type:"finished", audio_url}
 *   {type:"error", message}
 *
 * Client -> server:
 *   {type:"ai_finished", index}   after AI wav playback ends
 *   {type:"record_done", index}   after human recording stops
 *   {type:"pause"|"resume"}
 *   binary PCM16 mono 16kHz frames during RECORDING
 *
 * Key invariant: mic is opened ONLY when server state === "RECORDING".
 * AI playback and mic recording never overlap.
 */
(function () {
  "use strict";

  const LiveStudio = {
    ws: null,
    sessionId: null,
    state: "IDLE",
    currentIndex: -1,
    micEnabled: false,
    mediaStream: null,
    audioCtx: null,
    scriptProcessor: null,
    audioWorkletNode: null,
    captureSource: null,
    captureSink: null,
    audioWorkletReady: false,
    pendingWavBlob: null,
    pendingAudioIndex: null,
    playbackAudioEl: null,
    micPermissionGranted: false,
    pauseRequested: false,
    segmentData: [],

    init() {
      const form = document.getElementById("live-studio-form");
      if (form) form.addEventListener("submit", (e) => this.handleSubmit(e));

      const pauseBtn = document.getElementById("live-studio-pause-btn");
      if (pauseBtn) pauseBtn.addEventListener("click", () => this.handleTogglePause());

      const mdBtn = document.getElementById("live-md-select-btn");
      const mdInput = document.getElementById("live-md-file-input");
      if (mdBtn && mdInput) {
        mdBtn.addEventListener("click", () => mdInput.click());
        mdInput.addEventListener("change", (e) => this.handleMdFile(e));
      }

      if (!window.isSecureContext) {
        this.setHint("Firefox 麦克风需要 HTTPS；请打开 https://localhost:8443 或 http://localhost:8123", true);
      }
    },

    handleMdFile(e) {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const name = document.getElementById("live-md-selected-name");
      if (name) name.textContent = file.name;
      const reader = new FileReader();
      reader.onload = () => {
        const raw = String(reader.result || "");
        const cleaned = typeof stripMarkdown === "function" ? stripMarkdown(raw) : raw.trim();
        document.getElementById("live-studio-text").value = cleaned;
      };
      reader.readAsText(file);
    },

    async handleSubmit(e) {
      e.preventDefault();
      const text = document.getElementById("live-studio-text").value.trim();
      if (!text) {
        this.setHint("请输入演播脚本", true);
        return;
      }
       this.setHint("正在创建演播会话...");
      try {
        const resp = await fetch("/api/live/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
           body: JSON.stringify({ text }),
        });
        if (!resp.ok) {
          const err = await resp.json().catch(() => ({}));
          throw new Error(err.detail || `Start failed: ${resp.status}`);
        }
        const data = await resp.json();
         this.sessionId = data.session_id;
         this.pauseRequested = false;
        // Show the session immediately. Permission prompts can be delayed or
        // blocked by the browser, so they must not hide the live UI.
        this.showPanel();
        this.renderSegments(data.segments, data.live_speakers);
        const needsMic = data.segments.some((segment) => segment.source === "live");
        if (needsMic) {
          this.setHint("正在申请录音权限，TTS 播放期间不会录音...");
          try {
            await this.requestMicPermission();
          } catch (err) {
            await fetch(`/api/live/${this.sessionId}/stop`, { method: "POST" }).catch(() => {});
            this.setState("ERROR");
            this.setHint(`录音权限未获得：${err.message}`, true);
            return;
          }
        }
        if (data.asr) this.updateAsrProgress(data.asr);
        this.connectWS();
       } catch (err) {
         this.setHint(`创建失败: ${err.message}`, true);
      }
    },

    handleTogglePause() {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const shouldResume = this.state === "PAUSED" || this.pauseRequested;
      this.pauseRequested = !shouldResume;
      if (shouldResume) {
        if (this.playbackAudioEl && this.state === "PAUSED") {
          this.playbackAudioEl.play().catch(() => {});
        }
        this.send({ type: "resume" });
      } else {
        this.stopMic();
        if (this.playbackAudioEl) this.playbackAudioEl.pause();
        this.send({ type: "pause" });
      }
    },

    showPanel() {
       document.getElementById("live-studio-empty").hidden = true;
       document.getElementById("live-studio-panel").hidden = false;
       document.getElementById("live-studio-pause-btn").disabled = false;
       document.getElementById("live-studio-download").hidden = true;
    },

     renderSegments(segments) {
       const panel = document.getElementById("live-studio-panel");
       this.segmentData = segments;
      // Build a compact segment list once
      let list = panel.querySelector(".live-seg-list");
      if (!list) {
        list = document.createElement("div");
        list.className = "live-seg-list";
        panel.insertBefore(list, panel.firstChild);
      }
      list.innerHTML = "";
      for (const seg of segments) {
         const row = document.createElement("div");
         row.className = "podcast-segment-row";
         row.dataset.index = seg.index;
         row.title = "点击此段，从这里重新录制";
        const tag = seg.source === "live" ? "真人" : "AI";
         row.innerHTML = `
          <div class="seg-index">${seg.index + 1}</div>
           <div class="seg-text">${this.escapeHtml(seg.text)}</div>
           <div class="seg-meta">${seg.speaker || "默认"} · ${tag}</div>
           <span class="seg-status pending">待播</span>
           <audio class="segment-playback" controls preload="none" hidden></audio>
         `;
         row.addEventListener("click", (event) => {
           if (event.target.closest("audio, button, a, input")) return;
           this.requestRestart(seg.index);
         });
         list.appendChild(row);
      }
    },

    markSegment(index, status) {
      const row = document.querySelector(
        `.live-seg-list .podcast-segment-row[data-index="${index}"]`
      );
      if (!row) return;
      const badge = row.querySelector(".seg-status");
      if (!badge) return;
      badge.className = `seg-status ${status}`;
      const labels = {
        pending: "待播", generated: "AI已播", recorded: "已录",
        error: "错误", missing: "未录",
      };
      badge.textContent = labels[status] || status;
    },

    selectSegment(index) {
      const segment = this.segmentData.find((item) => item.index === index);
      if (!segment) return;
      document.querySelectorAll(".live-seg-list .podcast-segment-row").forEach((row) => {
        row.classList.toggle("selected", Number(row.dataset.index) === index);
      });
      this.currentIndex = index;
      this.setTargetText(segment.text, segment.speaker, segment.source);
      const asrText = document.getElementById("live-studio-asr-text");
      if (asrText) asrText.textContent = "";
    },

    setSegmentPlayback(index, audioUrl) {
      const row = document.querySelector(
        `.live-seg-list .podcast-segment-row[data-index="${index}"]`
      );
      const player = row && row.querySelector(".segment-playback");
      if (!player || !audioUrl) return;
      player.src = audioUrl;
      player.hidden = false;
    },

    setFinalDownload(audioUrl) {
      const link = document.getElementById("live-studio-download");
      if (!link || !audioUrl) return;
      link.href = audioUrl;
      link.download = "live-studio.wav";
      link.hidden = false;
    },

    resetSegmentsFrom(index) {
      document.querySelectorAll(".live-seg-list .podcast-segment-row").forEach((row) => {
        if (Number(row.dataset.index) < index) return;
        const badge = row.querySelector(".seg-status");
        if (badge) {
          badge.className = "seg-status pending";
          badge.textContent = "待播";
        }
        row.classList.remove("selected");
        const player = row.querySelector(".segment-playback");
        if (player) {
          player.pause();
          player.removeAttribute("src");
          player.load();
          player.hidden = true;
        }
      });
    },

    requestRestart(index) {
      this.resetSegmentsFrom(index);
      this.selectSegment(index);
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
        this.setHint("当前会话已结束，请重新开始后再从此段重录", true);
        return;
      }
      this.stopMic();
      if (this.playbackAudioEl) this.playbackAudioEl.pause();
      this.setHint(`正在从第 ${index + 1} 段重新开始...`);
      this.send({ type: "restart_from", index });
    },

    connectWS() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${location.host}/ws/live/${this.sessionId}`;
      this.ws = new WebSocket(url);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => this.setHint("演播已连接，开始驱动...");
      this.ws.onmessage = (ev) => this.onMessage(ev);
      this.ws.onerror = () => this.setHint("连接错误", true);
      this.ws.onclose = () => {
        if (this.state !== "FINISHED") this.setHint("连接已断开");
        this.stopMic();
      };
    },

    async onMessage(ev) {
      // Binary frame = AI wav audio bytes
      if (ev.data instanceof ArrayBuffer) {
        this.pendingWavBlob = ev.data;
        if (this.pendingAudioIndex !== null) {
          const index = this.pendingAudioIndex;
          this.pendingAudioIndex = null;
          this.playAiAudio(index);
        }
        return;
      }
      if (typeof ev.data !== "string") return;
      let frame;
      try { frame = JSON.parse(ev.data); } catch { return; }

      switch (frame.type) {
        case "state":
          this.setState(frame.state);
          if (frame.state === "RECORDING") {
            // Mic opens ONLY now, after AI finished
            this.startMic();
          } else if (frame.state === "AI_SPEAKING") {
            // Ensure mic is closed during AI playback
            this.stopMic();
          }
          break;

        case "segment_start":
          this.selectSegment(frame.index);
          this.setTargetText(frame.text, frame.speaker, frame.source);
          this.markSegment(frame.index, "pending");
          break;

        case "audio":
          // The binary WAV is a following WebSocket message. Wait for it
          // instead of trying to decode before it has arrived.
          this.pendingAudioIndex = frame.index;
          if (this.pendingWavBlob) {
            this.pendingAudioIndex = null;
            this.playAiAudio(frame.index);
          }
          break;

        case "record_start":
          // Mic is opened via state=RECORDING; show target text
          this.setTargetText(frame.target_text, "", "live");
          this.setHint("开始录音，识别完整后自动完成；也可手动完成当前段");
          this.showRecordDoneButton(frame.index);
          break;

        case "record_auto_done":
          this.setHint("识别完成，正在进入下一段；如需重录请点击对应段落");
          this.stopMic();
          break;

        case "restart":
          this.stopMic();
          this.resetSegmentsFrom(frame.to_index);
          this.selectSegment(frame.to_index);
          this.setHint(`已从第 ${frame.to_index + 1} 段重新开始`);
          break;

        case "asr_progress":
          this.updateAsrProgress(frame);
          break;

        case "asr_unavailable":
          this.updateAsrProgress({ status: "disabled", progress: 0, message: frame.reason });
          break;

        case "asr_partial":
          document.getElementById("live-studio-asr-text").textContent = frame.text || "";
          break;

        case "alignment":
          this.setHint(`实时识别对齐度 ${(Number(frame.ratio || 0) * 100).toFixed(0)}%`);
          break;

        case "segment_done":
          this.markSegment(frame.index, frame.source === "live" ? "recorded" : "generated");
          this.setSegmentPlayback(frame.index, frame.audio_url);
          if (frame.source === "live") {
            this.setHint(`第 ${frame.index + 1} 段录制完成`);
          } else {
            this.setHint(`第 ${frame.index + 1} 段 AI 播报完成`);
          }
          break;

        case "finished":
          this.setState("FINISHED");
          this.setFinalDownload(frame.audio_url);
          this.setHint("演播全部完成");
          document.getElementById("live-studio-pause-btn").disabled = true;
          break;

        case "error":
        this.setHint(`错误: ${frame.message}`, true);
          if (this.currentIndex >= 0) this.markSegment(this.currentIndex, "error");
          // Stop everything: close mic and allow restart.
          this.stopMic();
          this.state = "ERROR";
          document.getElementById("live-studio-pause-btn").disabled = true;
          break;
      }
    },

    playAiAudio(index) {
      if (!this.pendingWavBlob) {
        return;
      }
      const blob = new Blob([this.pendingWavBlob], { type: "audio/wav" });
      this.pendingWavBlob = null;
      const url = URL.createObjectURL(blob);

      // Use a fresh Audio element each time for reliable ended event
      const old = this.playbackAudioEl;
      if (old) { old.pause(); old.src = ""; }
      const player = document.getElementById("live-studio-player");
      player.src = url;
      this.playbackAudioEl = player;

      this.setHint(`正在播放第 ${index + 1} 段 AI 音频...`);
      const onEnded = () => {
        player.removeEventListener("ended", onEnded);
        URL.revokeObjectURL(url);
        // Confirm playback finished so server can advance (and open mic if next is live)
        this.send({ type: "ai_finished", index });
      };
      player.addEventListener("ended", onEnded);
      if (this.pauseRequested || this.state === "PAUSED") {
        this.setHint("已暂停，点击“继续”恢复播放");
      } else {
        player.play().catch((e) => {
          this.setHint(`播放失败: ${e.message}`, true);
          // Still advance to avoid deadlock
          this.send({ type: "ai_finished", index });
        });
      }
    },

    showRecordDoneButton(index) {
      let btn = document.getElementById("live-record-done-btn");
      if (!btn) {
        btn = document.createElement("button");
        btn.id = "live-record-done-btn";
        btn.type = "button";
        btn.className = "primary";
        btn.textContent = "完成当前段";
        const toolbar = document.querySelector("#live-studio-panel .live-toolbar");
        if (toolbar) toolbar.appendChild(btn);
        btn.addEventListener("click", () => {
          this.stopMic();
          this.send({ type: "record_done", index: this.currentIndex });
          btn.remove();
        });
      }
    },

    async startMic() {
      if (this.micEnabled) return;
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: { ideal: 16000 },
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
        this.audioCtx = this.audioCtx || new AudioContext({
          sampleRate: 16000,
          latencyHint: "interactive",
        });
        await this.audioCtx.resume();
        this.captureSource = this.audioCtx.createMediaStreamSource(this.mediaStream);
        this.captureSink = this.audioCtx.createGain();
        this.captureSink.gain.value = 0;
        this.micEnabled = true;

        if (this.audioCtx.audioWorklet) {
          if (!this.audioWorkletReady) {
            const source = `
              class SpeechCaptureProcessor extends AudioWorkletProcessor {
                constructor() {
                  super();
                  this.buffer = new Int16Array(1024);
                  this.offset = 0;
                }
                process(inputs) {
                  const input = inputs[0] && inputs[0][0];
                  if (!input) return true;
                  for (let i = 0; i < input.length; i++) {
                    const value = Math.max(-1, Math.min(1, input[i]));
                    this.buffer[this.offset++] = value * 32767;
                    if (this.offset === this.buffer.length) {
                      const chunk = this.buffer;
                      this.buffer = new Int16Array(1024);
                      this.offset = 0;
                      this.port.postMessage(
                        { buffer: chunk.buffer, sampleRate },
                        [chunk.buffer]
                      );
                    }
                  }
                  return true;
                }
              }
              registerProcessor("speech-capture", SpeechCaptureProcessor);
            `;
            const url = URL.createObjectURL(new Blob([source], { type: "application/javascript" }));
            await this.audioCtx.audioWorklet.addModule(url);
            URL.revokeObjectURL(url);
            this.audioWorkletReady = true;
          }
          this.audioWorkletNode = new AudioWorkletNode(this.audioCtx, "speech-capture", {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            outputChannelCount: [1],
          });
          this.audioWorkletNode.port.onmessage = (event) => {
            const data = event.data || {};
            if (data.buffer) {
              this.sendPcm16(new Int16Array(data.buffer), data.sampleRate || this.audioCtx.sampleRate);
            }
          };
          this.captureSource.connect(this.audioWorkletNode);
          this.audioWorkletNode.connect(this.captureSink).connect(this.audioCtx.destination);
        } else {
          // Fallback for browsers without AudioWorklet.
          this.scriptProcessor = this.audioCtx.createScriptProcessor(2048, 1, 1);
          this.scriptProcessor.onaudioprocess = (event) => {
            if (!this.micEnabled) return;
            this.sendFloatPcm(event.inputBuffer.getChannelData(0), this.audioCtx.sampleRate);
          };
          this.captureSource.connect(this.scriptProcessor);
          this.scriptProcessor.connect(this.captureSink).connect(this.audioCtx.destination);
        }
        this.micEnabled = true;
        document.getElementById("live-studio-warning").hidden = false;
        this.setHint("麦克风已开启，正在录音...");
      } catch (e) {
        this.setHint(`麦克风错误: ${e.message}`, true);
      }
    },

    async requestMicPermission() {
      if (this.micPermissionGranted) return;
      if (!window.isSecureContext) {
        throw new Error("当前页面不是安全上下文。Firefox 请使用 HTTPS 或 localhost");
      }
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error("当前浏览器不支持麦克风");
      }
      const probe = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: { ideal: 16000 },
          echoCancellation: true,
          noiseSuppression: true,
        },
      });
      probe.getTracks().forEach((track) => track.stop());
      this.micPermissionGranted = true;
    },

    stopMic() {
      this.micEnabled = false;
      if (this.scriptProcessor) {
        this.scriptProcessor.disconnect();
        this.scriptProcessor = null;
      }
      if (this.audioWorkletNode) {
        this.audioWorkletNode.port.onmessage = null;
        this.audioWorkletNode.disconnect();
        this.audioWorkletNode = null;
      }
      if (this.captureSource) {
        this.captureSource.disconnect();
        this.captureSource = null;
      }
      if (this.captureSink) {
        this.captureSink.disconnect();
        this.captureSink = null;
      }
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((t) => t.stop());
        this.mediaStream = null;
      }
      const warn = document.getElementById("live-studio-warning");
      if (warn) warn.hidden = true;
      const btn = document.getElementById("live-record-done-btn");
      if (btn) btn.remove();
    },

    sendFloatPcm(samples, sampleRate) {
      let output = samples;
      if (sampleRate !== 16000) {
        output = this.downsample(samples, sampleRate, 16000);
      }
      const pcm = new Int16Array(output.length);
      for (let i = 0; i < output.length; i++) {
        pcm[i] = Math.max(-1, Math.min(1, output[i])) * 0x7fff;
      }
      this.sendPcm16(pcm, 16000);
    },

    sendPcm16(samples, sampleRate) {
      if (!this.micEnabled || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      if (sampleRate === 16000) {
        this.ws.send(samples.buffer);
        return;
      }
      const floats = new Float32Array(samples.length);
      for (let i = 0; i < samples.length; i++) floats[i] = samples[i] / 32768;
      this.sendFloatPcm(floats, sampleRate);
    },

    downsample(buffer, fromRate, toRate) {
      if (fromRate === toRate) return buffer;
      const ratio = fromRate / toRate;
      const newLen = Math.round(buffer.length / ratio);
      const result = new Float32Array(newLen);
      let offset = 0;
      for (let i = 0; i < newLen; i++) {
        const nextOffset = Math.round((i + 1) * ratio);
        let acc = 0, count = 0;
        for (let j = offset; j < nextOffset && j < buffer.length; j++) {
          acc += buffer[j]; count++;
        }
        result[i] = count > 0 ? acc / count : 0;
        offset = nextOffset;
      }
      return result;
    },

    setTargetText(text, speaker, source) {
      const el = document.getElementById("live-studio-target-text");
      if (!el) return;
      const tag = source === "live" ? "[真人]" : "[AI]";
      const spk = speaker ? `${tag} ${speaker}: ` : `${tag} `;
      el.textContent = spk + text;
    },

    setState(state) {
      this.state = state;
      const el = document.getElementById("live-studio-state");
      if (!el) return;
      el.textContent = state;
      el.className = "seg-status " + (
        state === "AI_SPEAKING" ? "generated" :
        state === "RECORDING" ? "recorded" :
        state === "FINISHED" ? "generated" :
        state === "ERROR" ? "error" : "pending"
      );
      const pauseBtn = document.getElementById("live-studio-pause-btn");
      if (pauseBtn) {
        pauseBtn.disabled = ["IDLE", "FINISHED", "ERROR"].includes(state);
        pauseBtn.textContent = state === "PAUSED" ? "继续" : "暂停";
      }
      if (state === "RECORDING" && !document.getElementById("live-record-done-btn")) {
        this.showRecordDoneButton(this.currentIndex);
      }
    },

    setHint(msg, isError = false) {
      const el = document.getElementById("live-studio-status");
      if (!el) return;
      el.textContent = msg;
      el.style.color = isError ? "var(--danger)" : "var(--muted)";
    },

    updateAsrProgress(status) {
      const wrap = document.getElementById("live-asr-progress");
      const fill = document.getElementById("live-asr-progress-fill");
      const text = document.getElementById("live-asr-progress-text");
      if (!wrap || !fill || !text) return;
      const state = status.status || "idle";
      if (state === "disabled" || (status.enabled === false && state === "idle")) {
        wrap.hidden = true;
        return;
      }
      wrap.hidden = false;
      fill.style.width = `${Math.max(0, Math.min(100, Number(status.progress || 0) * 100))}%`;
      text.textContent = status.message || (state === "ready" ? "已就绪" : "准备中...");
      text.classList.toggle("error", state === "error");
      if (state === "ready") {
        fill.style.width = "100%";
      }
    },

    send(obj) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(obj));
      }
    },

    escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text || "";
      return div.innerHTML;
    },
  };

  window.LiveStudio = LiveStudio;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => LiveStudio.init());
  } else {
    LiveStudio.init();
  }
})();
