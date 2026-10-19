/**
 * Live Studio — 在线演播
 *
 * Flow:
 *   1. Import script → POST /api/live/start → segments rendered immediately
 *   2. Click a segment to select → edit bar shows text → auto-sync on blur/2s
 *   3. Click 开始 → WS drives AI TTS / human recording alternation
 *   4. ASR auto-completes a human segment at >= 85% alignment
 *   5. Click any segment to restart from there
 *   6. Download final merged audio after all segments done
 */
(function () {
  "use strict";

  const LiveStudio = {
    ws: null,
    sessionId: null,
    state: "IDLE",
    currentIndex: -1,
    segmentData: [],
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
    editSyncTimer: null,
    previewMode: false,
    previewAudio: null,
    previewSegments: [],
    previewIndex: 0,

    init() {
      const newBtn = document.getElementById("live-new-btn");
      if (newBtn) newBtn.addEventListener("click", () => this.openNewModal());
      const mdBtn = document.getElementById("live-md-select-btn");
      const mdInput = document.getElementById("live-md-file-input");
      if (mdBtn && mdInput) {
        mdBtn.addEventListener("click", () => mdInput.click());
        mdInput.addEventListener("change", (e) => this.handleMdFile(e));
      }
      const confirmBtn = document.getElementById("live-modal-confirm-btn");
      if (confirmBtn) confirmBtn.addEventListener("click", () => this.handleConfirm());
      const cancelBtn = document.getElementById("live-modal-cancel-btn");
      if (cancelBtn) cancelBtn.addEventListener("click", () => this.closeNewModal());
      const startBtn = document.getElementById("live-studio-start-btn");
      if (startBtn) startBtn.addEventListener("click", () => this.handleStart());
      const pauseBtn = document.getElementById("live-studio-pause-btn");
      if (pauseBtn) pauseBtn.addEventListener("click", () => this.handleTogglePause());
      const previewBtn = document.getElementById("live-studio-preview-btn");
      if (previewBtn) previewBtn.addEventListener("click", () => this.handleTogglePreview());
      const editArea = document.getElementById("live-edit-area");
      if (editArea) {
        editArea.addEventListener("input", () => this.scheduleEditSync());
        editArea.addEventListener("blur", () => this.flushEditSync());
      }
      if (!window.isSecureContext) {
        const secureHost = location.hostname || "localhost";
        this.setHint(`麦克风需要 HTTPS；请打开 https://${secureHost}:8443`, true);
      }
      this.loadSessionList().then((sessions) => {
        if (sessions && sessions.length > 0) {
          this.loadSession(sessions[0].session_id);
        }
      });
    },

    async loadSessionList() {
      try {
        const resp = await fetch("/api/live/sessions");
        let sessions = await resp.json();
        sessions.sort((a, b) => (b.created_at || "").localeCompare(a.created_at || ""));
        const list = document.getElementById("live-session-list");
        if (!list) return sessions;
        list.innerHTML = "";
        for (const s of sessions) {
          const item = document.createElement("div");
          item.className = "live-session-item" + (s.session_id === this.sessionId ? " active" : "");
          item.dataset.sessionId = s.session_id;
          const date = s.created_at ? s.created_at.slice(5, 16).replace("T", " ") : "";
          item.innerHTML = `
            <div class="session-info">
              <div class="session-title">${this.escapeHtml(s.title || s.first_text || s.session_id)}</div>
              <div class="session-meta">${date} · ${s.segment_count}段${s.has_final ? " · 已完成" : ""}</div>
            </div>
            <button class="session-delete" title="删除">×</button>
          `;
          item.addEventListener("click", (e) => {
            if (e.target.closest(".session-delete")) return;
            this.loadSession(s.session_id);
          });
          item.querySelector(".session-delete").addEventListener("click", (e) => {
            e.stopPropagation();
            this.deleteSession(s.session_id);
          });
          list.appendChild(item);
        }
        return sessions;
      } catch {
        return [];
      }
    },

    async loadSession(sessionId) {
      document.querySelectorAll(".live-session-item").forEach((el) => {
        el.classList.toggle("active", el.dataset.sessionId === sessionId);
      });
      try {
        this.stopMic();
        if (this.ws) { this.ws.close(); this.ws = null; }
        const resp = await fetch(`/api/live/${sessionId}`);
        if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`);
        const data = await resp.json();
        this.sessionId = data.session_id;
        this.segmentData = data.segments;
        this.pauseRequested = false;
        this.currentIndex = -1;
        document.getElementById("live-toolbar").hidden = false;
        document.getElementById("live-edit-bar").hidden = false;
        this.renderSegments(data.segments);
        for (const seg of data.segments) {
          if (seg.audio_url) {
            this.markSegment(seg.index, seg.status);
            this.setSegmentPlayback(seg.index, seg.audio_url);
          }
        }
        if (data.segments.length > 0) {
          this.selectSegment(data.segments[0].index, true);
        }
        const isFinished = data.state === "FINISHED" || data.has_final;
        const startBtn = document.getElementById("live-studio-start-btn");
        const pauseBtn = document.getElementById("live-studio-pause-btn");
        const downloadBtn = document.getElementById("live-studio-download-btn");
        if (startBtn) startBtn.disabled = isFinished;
        if (pauseBtn) pauseBtn.disabled = true;
        if (isFinished) {
          this.setState("FINISHED");
          this.setFinalDownload(data.final_audio_url);
          this.setHint("该会话已演播完成");
        } else {
          this.setState("IDLE");
          if (downloadBtn) downloadBtn.hidden = true;
          this.setHint(`已加载 ${data.title}，共 ${data.segments.length} 段`);
        }
        this.updatePreviewButton();
      } catch (err) {
        this.setHint(`加载会话失败: ${err.message}`, true);
      }
    },

    async deleteSession(sessionId) {
      try {
        await fetch(`/api/live/${sessionId}`, { method: "DELETE" });
        const isCurrent = this.sessionId === sessionId;
        if (isCurrent) {
          this.clearSessionUI();
        }
        const sessions = await this.loadSessionList();
        if (isCurrent && sessions && sessions.length > 0) {
          this.loadSession(sessions[0].session_id);
        } else if (isCurrent) {
          this.setHint("已删除所有会话");
        } else {
          this.setHint("已删除会话");
        }
      } catch (err) {
        this.setHint(`删除失败: ${err.message}`, true);
      }
    },

    clearSessionUI() {
      this.stopPreview();
      this.sessionId = null;
      this.segmentData = [];
      this.currentIndex = -1;
      this.setState("IDLE");
      const list = document.getElementById("live-segment-list");
      if (list) list.innerHTML = "";
      const editArea = document.getElementById("live-edit-area");
      if (editArea) { editArea.value = ""; editArea.disabled = true; }
      const toolbar = document.getElementById("live-toolbar");
      if (toolbar) toolbar.hidden = true;
      const editBar = document.getElementById("live-edit-bar");
      if (editBar) editBar.hidden = true;
      const downloadBtn = document.getElementById("live-studio-download-btn");
      if (downloadBtn) downloadBtn.hidden = true;
      const asrBar = document.getElementById("live-asr-bar");
      if (asrBar) asrBar.hidden = true;
      this.stopMic();
      if (this.ws) { this.ws.close(); this.ws = null; }
    },

    openNewModal() {
      document.getElementById("live-new-modal").hidden = false;
      const titleInput = document.getElementById("live-modal-title");
      if (titleInput) { titleInput.value = ""; }
      const ta = document.getElementById("live-modal-text");
      if (ta) { ta.value = ""; ta.focus(); }
      const name = document.getElementById("live-md-name");
      if (name) name.textContent = "可先导入再编辑";
    },

    closeNewModal() {
      document.getElementById("live-new-modal").hidden = true;
    },

    handleConfirm() {
      const ta = document.getElementById("live-modal-text");
      const titleInput = document.getElementById("live-modal-title");
      const text = (ta?.value || "").trim();
      const title = (titleInput?.value || "").trim();
      if (!text) { this.setHint("请输入或导入脚本", true); return; }
      this.closeNewModal();
      this.createSession(text, title);
    },

    // ── Script import ──

    handleMdFile(e) {
      const file = e.target.files && e.target.files[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = () => {
        const raw = String(reader.result || "");
        const cleaned = typeof stripMarkdown === "function" ? stripMarkdown(raw) : raw.trim();
        if (!cleaned) { this.setHint("文件内容为空", true); return; }
        const ta = document.getElementById("live-modal-text");
        if (ta) ta.value = cleaned;
        const name = document.getElementById("live-md-name");
        if (name) name.textContent = file.name;
      };
      reader.readAsText(file);
    },

    async createSession(text, title = "") {
      this.setHint("正在创建会话...");
      try {
        const resp = await fetch("/api/live/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, title }),
        });
        if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || `HTTP ${resp.status}`);
        const data = await resp.json();
        this.sessionId = data.session_id;
        this.segmentData = data.segments;
        this.pauseRequested = false;
        this.currentIndex = -1;
        document.getElementById("live-toolbar").hidden = false;
        document.getElementById("live-edit-bar").hidden = false;
        document.getElementById("live-studio-start-btn").disabled = false;
        document.getElementById("live-studio-pause-btn").disabled = true;
        document.getElementById("live-studio-download-btn").hidden = true;
        this.renderSegments(data.segments);
        if (data.segments.length > 0) {
          this.selectSegment(data.segments[0].index, true);
        }
        if (data.asr) this.updateAsrProgress(data.asr);
        this.setState("IDLE");
        this.setHint(`已导入 ${data.segments.length} 段，点击「开始」演播`);
        this.loadSessionList();
      } catch (err) {
        this.setHint(`创建失败: ${err.message}`, true);
      }
    },

    // ── Start broadcast ──

    async handleStart() {
      if (!this.sessionId) return;
      if (this.state === "FINISHED") {
        this.setHint("该会话已结束，请新建或选择其他会话");
        return;
      }
      const needsMic = this.segmentData.some((s) => s.source === "live");
      if (needsMic) {
        this.setHint("正在申请录音权限...");
        try { await this.requestMicPermission(); }
        catch (err) { this.setHint(`录音权限未获得：${err.message}`, true); return; }
      }
      document.getElementById("live-studio-start-btn").disabled = true;
      document.getElementById("live-studio-pause-btn").disabled = false;
      this.prefetchTts();
      this.connectWS();
    },

    handleTogglePause() {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      const shouldResume = this.state === "PAUSED" || this.pauseRequested;
      this.pauseRequested = !shouldResume;
      if (shouldResume) {
        if (this.playbackAudioEl && this.state === "PAUSED") this.playbackAudioEl.play().catch(() => {});
        this.send({ type: "resume" });
      } else {
        this.stopMic();
        if (this.playbackAudioEl) this.playbackAudioEl.pause();
        this.send({ type: "pause" });
      }
    },

    async prefetchTts() {
      if (!this.sessionId) return;
      try {
        await fetch(`/api/live/${this.sessionId}/prefetch`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({}),
        });
      } catch (err) {
        console.error("[live] prefetchTts failed:", err);
      }
    },

    updatePreviewButton() {
      const btn = document.getElementById("live-studio-preview-btn");
      if (!btn) return;
      const hasAudio = this.segmentData.some(
        (s) => s.audio_url && (s.status === "generated" || s.status === "recorded")
      );
      btn.hidden = !hasAudio && !this.previewMode;
    },

    handleTogglePreview() {
      if (this.previewMode) {
        this.stopPreview();
      } else {
        this.startPreview();
      }
    },

    startPreview() {
      const segments = this.segmentData
        .filter((s) => s.audio_url && (s.status === "generated" || s.status === "recorded"))
        .sort((a, b) => a.index - b.index);
      if (!segments.length) {
        this.setHint("暂无可试听片段", true);
        return;
      }
      this.previewMode = true;
      this.previewSegments = segments;
      this.previewIndex = 0;
      const btn = document.getElementById("live-studio-preview-btn");
      if (btn) btn.textContent = "停止试听";
      this.setHint("开始试听...");
      this.playPreviewSegment();
    },

    playPreviewSegment() {
      if (!this.previewMode || this.previewIndex >= this.previewSegments.length) {
        this.stopPreview();
        return;
      }
      const seg = this.previewSegments[this.previewIndex];
      this.setHint(`试听第 ${seg.index + 1} 段`);
      const audio = new Audio(seg.audio_url);
      audio.onended = () => {
        if (!this.previewMode) return;
        this.previewIndex += 1;
        this.playPreviewSegment();
      };
      audio.onerror = () => {
        if (!this.previewMode) return;
        this.previewIndex += 1;
        this.playPreviewSegment();
      };
      audio.play().catch((err) => {
        console.error("[live] preview play error:", err);
        this.stopPreview();
      });
      this.previewAudio = audio;
    },

    stopPreview() {
      this.previewMode = false;
      if (this.previewAudio) {
        this.previewAudio.pause();
        this.previewAudio.src = "";
        this.previewAudio = null;
      }
      this.previewSegments = [];
      this.previewIndex = 0;
      const btn = document.getElementById("live-studio-preview-btn");
      if (btn) btn.textContent = "试听";
      this.updatePreviewButton();
      this.setHint("试听已停止");
    },

    // ── Segment rendering ──

    renderSegments(segments) {
      this.segmentData = segments;
      const list = document.getElementById("live-segment-list");
      if (!list) return;
      list.innerHTML = "";
      for (const seg of segments) {
        const row = document.createElement("div");
        row.className = "live-seg-row";
        row.dataset.index = seg.index;
        row.title = "点击选中并编辑；录音中点击可从此段重录";
        const tag = seg.source === "live" ? "真人" : "AI";
        row.innerHTML = `
          <span class="seg-num">${String(seg.index + 1).padStart(2, "0")}</span>
          <span class="seg-text">${this.escapeHtml(seg.text)}</span>
          <span class="seg-tag ${seg.source}">${tag}</span>
          <span class="seg-status pending">待播</span>
          <audio class="seg-playback" controls preload="none" hidden></audio>
        `;
        row.addEventListener("click", (event) => {
          if (event.target.closest("audio, button, a, input")) return;
          this.selectSegment(seg.index);
        });
        list.appendChild(row);
      }
    },

    markSegment(index, status) {
      const row = document.querySelector(`.live-seg-row[data-index="${index}"]`);
      if (!row) return;
      const badge = row.querySelector(".seg-status");
      if (!badge) return;
      badge.className = `seg-status ${status}`;
      const labels = { pending: "待播", generated: "已播", recorded: "已录", error: "错误", missing: "未录" };
      badge.textContent = labels[status] || status;
    },

    setSegmentPlayback(index, audioUrl) {
      const row = document.querySelector(`.live-seg-row[data-index="${index}"]`);
      const player = row && row.querySelector(".seg-playback");
      if (!player || !audioUrl) return;
      player.src = audioUrl;
      player.hidden = false;
    },

    selectSegment(index, fromServer = false) {
      const segment = this.segmentData.find((s) => s.index === index);
      if (!segment) return;

      // Only trigger restart when the USER clicks, not from server messages
      if (!fromServer && this.state === "RECORDING" && index !== this.currentIndex
          && this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.requestRestart(index);
        return;
      }

      document.querySelectorAll(".live-seg-row").forEach((row) => {
        row.classList.toggle("selected", Number(row.dataset.index) === index);
      });
      this.currentIndex = index;
      const editArea = document.getElementById("live-edit-area");
      if (editArea) {
        editArea.value = segment.text;
        editArea.disabled = false;
      }
      const asrText = document.getElementById("live-studio-asr-text");
      if (asrText) asrText.textContent = "";
    },

    // ── Edit bar auto-sync ──

    scheduleEditSync() {
      if (this.editSyncTimer) clearTimeout(this.editSyncTimer);
      this.editSyncTimer = setTimeout(() => this.flushEditSync(), 2000);
    },

    flushEditSync() {
      if (this.editSyncTimer) { clearTimeout(this.editSyncTimer); this.editSyncTimer = null; }
      if (this.currentIndex < 0) return;
      const editArea = document.getElementById("live-edit-area");
      if (!editArea) return;
      const newText = editArea.value.trim();
      const segment = this.segmentData.find((s) => s.index === this.currentIndex);
      if (!segment || segment.text === newText) return;
      segment.text = newText;
      // Update display
      const row = document.querySelector(`.live-seg-row[data-index="${this.currentIndex}"] .seg-text`);
      if (row) row.textContent = newText;
      // Sync to server
      if (this.sessionId) {
        fetch(`/api/live/${this.sessionId}/segment/${this.currentIndex}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: newText }),
        }).catch(() => {});
      }
      this.setHint(`第 ${this.currentIndex + 1} 段已更新`);
    },

    // ── Restart from segment ──

    requestRestart(index) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      this.stopMic();
      if (this.playbackAudioEl) this.playbackAudioEl.pause();
      this.resetSegmentsFrom(index);
      this.setHint(`正在从第 ${index + 1} 段重新开始...`);
      this.send({ type: "restart_from", index });
    },

    resetSegmentsFrom(index) {
      document.querySelectorAll(".live-seg-row").forEach((row) => {
        if (Number(row.dataset.index) < index) return;
        const badge = row.querySelector(".seg-status");
        if (badge) { badge.className = "seg-status pending"; badge.textContent = "待播"; }
        row.classList.remove("selected");
        const player = row.querySelector(".seg-playback");
        if (player) { player.pause(); player.removeAttribute("src"); player.load(); player.hidden = true; }
      });
    },

    // ── WebSocket ──

    connectWS() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${location.host}/ws/live/${this.sessionId}`;
      this.ws = new WebSocket(url);
      this.ws.binaryType = "arraybuffer";
      this.ws.onopen = () => this.setHint("已连接，开始演播...");
      this.ws.onmessage = (ev) => this.onMessage(ev);
      this.ws.onerror = () => this.setHint("连接错误", true);
      this.ws.onclose = () => { if (this.state !== "FINISHED") this.setHint("连接已断开"); this.stopMic(); };
    },

    async onMessage(ev) {
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
          if (frame.state === "RECORDING") this.startMic();
          else if (frame.state === "AI_SPEAKING") this.stopMic();
          break;
        case "segment_start":
          this.selectSegment(frame.index, true);
          this.markSegment(frame.index, "pending");
          break;
        case "audio":
          this.pendingAudioIndex = frame.index;
          if (this.pendingWavBlob) {
            this.pendingAudioIndex = null;
            this.playAiAudio(frame.index);
          }
          break;
        case "record_start":
          this.setHint("录音中，说完后自动完成；也可手动点击其他段落重录");
          break;
        case "record_auto_done":
          this.setHint("识别完成，正在进入下一段...");
          this.stopMic();
          break;
        case "restart":
          this.stopMic();
          this.resetSegmentsFrom(frame.to_index);
          this.selectSegment(frame.to_index, true);
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
          document.getElementById("live-asr-bar").hidden = !frame.text;
          break;
        case "alignment":
          const badge = document.getElementById("live-studio-alignment");
          if (badge) badge.textContent = `${(Number(frame.ratio || 0) * 100).toFixed(0)}%`;
          break;
        case "segment_done":
          this.markSegment(frame.index, frame.source === "live" ? "recorded" : "generated");
          this.setSegmentPlayback(frame.index, frame.audio_url);
          const doneSeg = this.segmentData.find((s) => s.index === frame.index);
          if (doneSeg) {
            doneSeg.status = frame.source === "live" ? "recorded" : "generated";
            doneSeg.audio_url = frame.audio_url;
          }
          this.updatePreviewButton();
          this.setHint(`第 ${frame.index + 1} 段完成`);
          break;
        case "finished":
          this.setState("FINISHED");
          this.setFinalDownload(frame.audio_url);
          this.setHint("演播全部完成");
          document.getElementById("live-studio-pause-btn").disabled = true;
          document.getElementById("live-studio-start-btn").disabled = true;
          break;
        case "error":
          this.setHint(`错误: ${frame.message}`, true);
          if (this.currentIndex >= 0) this.markSegment(this.currentIndex, "error");
          this.stopMic();
          this.state = "ERROR";
          document.getElementById("live-studio-pause-btn").disabled = true;
          document.getElementById("live-studio-start-btn").disabled = false;
          break;
      }
    },

    playAiAudio(index) {
      if (!this.pendingWavBlob) return;
      const blob = new Blob([this.pendingWavBlob], { type: "audio/wav" });
      this.pendingWavBlob = null;
      const url = URL.createObjectURL(blob);
      const old = this.playbackAudioEl;
      if (old) { old.pause(); old.src = ""; }
      const player = document.getElementById("live-studio-player");
      player.src = url;
      this.playbackAudioEl = player;
      this.setHint(`正在播放第 ${index + 1} 段 AI 音频...`);
      const onEnded = () => {
        player.removeEventListener("ended", onEnded);
        URL.revokeObjectURL(url);
        this.send({ type: "ai_finished", index });
      };
      player.addEventListener("ended", onEnded);
      if (this.pauseRequested || this.state === "PAUSED") {
        this.setHint("已暂停，点击继续恢复播放");
      } else {
        player.play().catch(() => { this.send({ type: "ai_finished", index }); });
      }
    },

    setFinalDownload(audioUrl) {
      const btn = document.getElementById("live-studio-download-btn");
      if (!btn || !audioUrl) return;
      btn.onclick = async () => {
        btn.disabled = true;
        this.setHint("正在统一响度并合并音频...");
        try {
          const resp = await fetch(`/api/live/${this.sessionId}/export`, { method: "POST" });
          const data = await resp.json().catch(() => ({}));
          if (!resp.ok) throw new Error(data.detail || `HTTP ${resp.status}`);
          const a = document.createElement("a");
          a.href = data.audio_url || audioUrl;
          a.download = "live-studio.wav";
          a.click();
          this.setHint("已导出合并音频");
        } catch (err) {
          this.setHint(`导出失败: ${err.message}`, true);
        } finally {
          btn.disabled = false;
        }
      };
      btn.hidden = false;
    },

    setState(state) {
      this.state = state;
      const el = document.getElementById("live-studio-state");
      if (el) {
        el.textContent = state;
        el.className = "seg-status " + (
          state === "AI_SPEAKING" ? "generated" :
          state === "RECORDING" ? "recorded" :
          state === "FINISHED" ? "generated" :
          state === "ERROR" ? "error" : "pending"
        );
      }
      const pauseBtn = document.getElementById("live-studio-pause-btn");
      if (pauseBtn) {
        pauseBtn.disabled = ["IDLE", "FINISHED", "ERROR"].includes(state);
        pauseBtn.textContent = state === "PAUSED" ? "继续" : "暂停";
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
      if (state === "disabled" || state === "ready" || state === "idle" || (status.enabled === false && state === "idle")) {
        wrap.hidden = true;
        return;
      }
      wrap.hidden = false;
      fill.style.width = `${Math.max(0, Math.min(100, Number(status.progress || 0) * 100))}%`;
      text.textContent = status.message || "准备中...";
      text.classList.toggle("error", state === "error");
    },

    // ── Mic capture (AudioWorklet) ──

    async startMic() {
      if (this.micEnabled) return;
      try {
        console.log("[live] startMic: opening microphone...");
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, sampleRate: { ideal: 16000 }, echoCancellation: true, noiseSuppression: true },
        });
        console.log("[live] startMic: mic stream obtained");
        this.audioCtx = this.audioCtx || new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
        await this.audioCtx.resume();
        console.log("[live] startMic: audio context resumed, state:", this.audioCtx.state);
        this.captureSource = this.audioCtx.createMediaStreamSource(this.mediaStream);
        this.captureSink = this.audioCtx.createGain();
        this.captureSink.gain.value = 0;
        this.micEnabled = true;

        if (this.audioCtx.audioWorklet) {
          if (!this.audioWorkletReady) {
            const source = `
              class P extends AudioWorkletProcessor {
                constructor() { super(); this.b = new Int16Array(1024); this.o = 0; }
                process(inputs) {
                  const i = inputs[0] && inputs[0][0]; if (!i) return true;
                  for (let j = 0; j < i.length; j++) {
                    this.b[this.o++] = Math.max(-1, Math.min(1, i[j])) * 32767;
                    if (this.o === this.b.length) {
                      const c = this.b; this.b = new Int16Array(1024); this.o = 0;
                      this.port.postMessage({ buffer: c.buffer, sampleRate }, [c.buffer]);
                    }
                  }
                  return true;
                }
              }
              registerProcessor("cap", P);
            `;
            const url = URL.createObjectURL(new Blob([source], { type: "application/javascript" }));
            await this.audioCtx.audioWorklet.addModule(url);
            URL.revokeObjectURL(url);
            this.audioWorkletReady = true;
          }
          this.audioWorkletNode = new AudioWorkletNode(this.audioCtx, "cap", {
            numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
          });
          this.audioWorkletNode.port.onmessage = (e) => {
            const d = e.data || {};
            if (d.buffer) this.sendPcm16(new Int16Array(d.buffer), d.sampleRate || this.audioCtx.sampleRate);
          };
          this.captureSource.connect(this.audioWorkletNode);
          this.audioWorkletNode.connect(this.captureSink).connect(this.audioCtx.destination);
        } else {
          this.scriptProcessor = this.audioCtx.createScriptProcessor(2048, 1, 1);
          this.scriptProcessor.onaudioprocess = (e) => {
            if (!this.micEnabled) return;
            this.sendFloatPcm(e.inputBuffer.getChannelData(0), this.audioCtx.sampleRate);
          };
          this.captureSource.connect(this.scriptProcessor);
          this.scriptProcessor.connect(this.captureSink).connect(this.audioCtx.destination);
        }
        document.getElementById("live-studio-warning").hidden = false;
        this.setHint("麦克风已开启，正在录音...");
      } catch (e) {
        console.error("[live] startMic error:", e);
        this.setHint(`麦克风错误: ${e.message}`, true);
      }
    },

    async requestMicPermission() {
      if (this.micPermissionGranted) return;
      if (!window.isSecureContext) throw new Error("当前页面不是安全上下文");
      console.log("[live] requestMicPermission: requesting mic access");
      const probe = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: { ideal: 16000 } } });
      probe.getTracks().forEach((t) => t.stop());
      this.micPermissionGranted = true;
      // Prime the AudioContext inside the user gesture, so that later
      // RECORDING segments can resume/start immediately without waiting
      // for another click.
      this.audioCtx = this.audioCtx || new AudioContext({ sampleRate: 16000, latencyHint: "interactive" });
      await this.audioCtx.resume();
      console.log("[live] requestMicPermission: mic granted, audio context state:", this.audioCtx.state);
    },

    stopMic() {
      this.micEnabled = false;
      if (this.scriptProcessor) { this.scriptProcessor.disconnect(); this.scriptProcessor = null; }
      if (this.audioWorkletNode) { this.audioWorkletNode.port.onmessage = null; this.audioWorkletNode.disconnect(); this.audioWorkletNode = null; }
      if (this.captureSource) { this.captureSource.disconnect(); this.captureSource = null; }
      if (this.captureSink) { this.captureSink.disconnect(); this.captureSink = null; }
      if (this.mediaStream) { this.mediaStream.getTracks().forEach((t) => t.stop()); this.mediaStream = null; }
      const warn = document.getElementById("live-studio-warning");
      if (warn) warn.hidden = true;
    },

    sendFloatPcm(samples, sampleRate) {
      let out = samples;
      if (sampleRate !== 16000) out = this.downsample(samples, sampleRate, 16000);
      const pcm = new Int16Array(out.length);
      for (let i = 0; i < out.length; i++) pcm[i] = Math.max(-1, Math.min(1, out[i])) * 0x7fff;
      this.sendPcm16(pcm, 16000);
    },

    sendPcm16(samples, sampleRate) {
      if (!this.micEnabled || !this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      if (sampleRate === 16000) { this.ws.send(samples.buffer); return; }
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
        const next = Math.round((i + 1) * ratio);
        let acc = 0, count = 0;
        for (let j = offset; j < next && j < buffer.length; j++) { acc += buffer[j]; count++; }
        result[i] = count > 0 ? acc / count : 0;
        offset = next;
      }
      return result;
    },

    send(obj) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(obj));
    },

    escapeHtml(text) {
      const div = document.createElement("div");
      div.textContent = text || "";
      return div.innerHTML;
    },
  };

  window.LiveStudio = LiveStudio;
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => LiveStudio.init());
  else LiveStudio.init();
})();
