/**
 * Live Podcast frontend module.
 *
 * ACC-007-1: MediaRecorder real-time PCM upload (SPEC-007 / ST-LP-007)
 * ACC-011-1: AudioContext real-time playback (SPEC-011 / ST-LP-011)
 * ACC-006-1: WebSocket connection (SPEC-006 / ST-LP-006)
 */

(function () {
  "use strict";

  const LivePodcast = {
    ws: null,
    audioCtx: null,
    mediaStream: null,
    mediaRecorder: null,
    scriptProcessor: null,
    playbackGain: null,
    nextStartTime: 0,
    currentSessionId: null,
    currentProjectId: null,
    state: "IDLE",
    micEnabled: false,

    /**
     * ACC-006-1: Connect WebSocket to live session.
     */
    async startLive(projectId, liveOnly) {
      const resp = await fetch(`/api/podcasts/${projectId}/live/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ live_only: !!liveOnly }),
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Start failed: ${resp.status}`);
      }
      const data = await resp.json();
      this.currentSessionId = data.session_id;
      this.currentProjectId = projectId;
      this.state = data.state;

      this._connectWS();
      this._initPlayback();
      this._showPanel();
      this._setStatus(`Live session: ${data.session_id}`);
      document.getElementById("live-start-btn").disabled = true;
      document.getElementById("live-stop-btn").disabled = false;
    },

    /**
     * ACC-006-1: Open WebSocket connection.
     */
    _connectWS() {
      const proto = location.protocol === "https:" ? "wss:" : "ws:";
      const url = `${proto}//${location.host}/ws/podcasts/${this.currentProjectId}/live/${this.currentSessionId}?role=driver`;
      this.ws = new WebSocket(url);
      this.ws.binaryType = "arraybuffer";

      this.ws.onopen = () => {
        this._setStatus("WebSocket connected");
      };

      this.ws.onmessage = async (ev) => {
        if (ev.data instanceof ArrayBuffer) {
          this._handleBinaryFrame(ev.data);
        } else if (typeof ev.data === "string") {
          this._handleJSONFrame(JSON.parse(ev.data));
        }
      };

      this.ws.onerror = (ev) => {
        this._setStatus("WebSocket error");
      };

      this.ws.onclose = () => {
        this._setStatus("WebSocket closed");
        this._stopMic();
      };
    },

    /**
     * ACC-006-2: Handle JSON control frames from server.
     */
    _handleJSONFrame(frame) {
      switch (frame.type) {
        case "state":
          this.state = frame.state;
          this._setStatus(`State: ${frame.state}`);
          if (frame.state === "RECORDING") {
            this._startMic();
          } else if (frame.state === "AI_SPEAKING") {
            this._stopMic();
          }
          break;
        case "segment_start":
          document.getElementById("live-target-text").textContent = frame.text || "";
          document.getElementById("live-asr-text").textContent = "";
          break;
        case "asr_partial":
          document.getElementById("live-asr-text").textContent = frame.text || "";
          break;
        case "asr_final":
          document.getElementById("live-asr-text").textContent = frame.text || "";
          break;
        case "alignment":
          this._setStatus(`Alignment: ${frame.matched_chars}/${frame.total_chars}`);
          break;
        case "audio_info":
          this._initPlayback(frame.sample_rate || 24000);
          break;
        case "error":
          this._setStatus(`Error: ${frame.message || frame.code}`);
          break;
      }
    },

    /**
     * ACC-011-1: Handle binary PCM frames for playback.
     */
    async _handleBinaryFrame(data) {
      if (!this.audioCtx) return;
      try {
        const buf = await this.audioCtx.decodeAudioData(data.slice(0));
        const src = this.audioCtx.createBufferSource();
        src.buffer = buf;
        src.connect(this.playbackGain);
        const now = this.audioCtx.currentTime;
        const startAt = Math.max(this.nextStartTime, now);
        src.start(startAt);
        this.nextStartTime = startAt + buf.duration;
      } catch (e) {
        // decodeAudioData may fail on raw PCM; ignore
      }
    },

    /**
     * ACC-011-1: Initialize AudioContext for playback.
     */
    _initPlayback(sampleRate) {
      if (this.audioCtx) return;
      const sr = sampleRate || 24000;
      this.audioCtx = new AudioContext({ sampleRate: sr });
      this.playbackGain = this.audioCtx.createGain();
      this.playbackGain.connect(this.audioCtx.destination);
      this.playbackGain.gain.value = 1.0;
      this.nextStartTime = this.audioCtx.currentTime;
    },

    /**
     * ACC-007-1: Start microphone capture with MediaRecorder / ScriptProcessor.
     */
    async _startMic() {
      if (this.micEnabled) return;
      try {
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            channelCount: 1,
            sampleRate: { ideal: 48000 },
            echoCancellation: true,
            noiseSuppression: true,
          },
        });
        const trackRate = this.mediaStream.getAudioTracks()[0].getSettings().sampleRate || 48000;
        this.audioCtx = this.audioCtx || new AudioContext({ sampleRate: trackRate });
        const src = this.audioCtx.createMediaStreamSource(this.mediaStream);
        this.scriptProcessor = this.audioCtx.createScriptProcessor(4096, 1, 1);

        this.scriptProcessor.onaudioprocess = (e) => {
          if (!this.micEnabled) return;
          if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const pcm = e.inputBuffer.getChannelData(0);
            const int16 = new Int16Array(pcm.length);
            for (let i = 0; i < pcm.length; i++) {
              int16[i] = Math.max(-1, Math.min(1, pcm[i])) * 0x7fff;
            }
            this.ws.send(int16.buffer);
          }
        };

        src.connect(this.scriptProcessor).connect(this.audioCtx.destination);
        this.micEnabled = true;

        // Pre-roll beep
        this._playBeep(800, 0.25);

        // Show headphone warning
        document.getElementById("live-warning").hidden = false;

        // Mute playback to avoid feedback
        if (this.playbackGain) {
          this.playbackGain.gain.value = 0;
        }
      } catch (e) {
        this._setStatus(`Microphone error: ${e.message}`);
      }
    },

    /**
     * ACC-007-2: Stop microphone capture.
     */
    _stopMic() {
      this.micEnabled = false;
      if (this.scriptProcessor) {
        this.scriptProcessor.disconnect();
        this.scriptProcessor = null;
      }
      if (this.mediaStream) {
        this.mediaStream.getTracks().forEach((t) => t.stop());
        this.mediaStream = null;
      }
      document.getElementById("live-warning").hidden = true;
      if (this.playbackGain) {
        this.playbackGain.gain.value = 1.0;
      }
    },

    /**
     * Play a beep tone (pre-roll cue).
     */
    _playBeep(freq, duration) {
      if (!this.audioCtx) return;
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.frequency.value = freq;
      gain.gain.value = 0.1;
      osc.connect(gain).connect(this.audioCtx.destination);
      const now = this.audioCtx.currentTime;
      osc.start(now);
      osc.stop(now + duration);
    },

    /**
     * Stop live session.
     */
    async stopLive() {
      if (!this.currentSessionId) return;
      try {
        const resp = await fetch(
          `/api/podcasts/${this.currentProjectId}/live/${this.currentSessionId}/stop`,
          { method: "POST" },
        );
        const data = await resp.json().catch(() => ({}));
        this._setStatus(`Stopped: ${data.state || "unknown"}`);
      } catch (e) {
        this._setStatus(`Stop error: ${e.message}`);
      }
      this._stopMic();
      if (this.ws) {
        this.ws.close();
        this.ws = null;
      }
      document.getElementById("live-start-btn").disabled = false;
      document.getElementById("live-stop-btn").disabled = true;
      this.currentSessionId = null;
    },

    _showPanel() {
      document.getElementById("live-panel").hidden = false;
    },

    _setStatus(msg) {
      const el = document.getElementById("live-status");
      if (el) el.textContent = msg;
    },

    /**
     * Initialize event listeners when DOM is ready.
     */
    init() {
      const startBtn = document.getElementById("live-start-btn");
      const stopBtn = document.getElementById("live-stop-btn");
      if (startBtn) {
        startBtn.addEventListener("click", () => {
          const projectId = window.__currentPodcastProjectId;
          if (!projectId) {
            this._setStatus("Please select a podcast project first");
            return;
          }
          const liveOnly = document.getElementById("live-only-checkbox").checked;
          this.startLive(projectId, liveOnly).catch((e) =>
            this._setStatus(`Start error: ${e.message}`),
          );
        });
      }
      if (stopBtn) {
        stopBtn.addEventListener("click", () => {
          this.stopLive();
        });
      }
    },
  };

  // Expose globally for app.js integration
  window.LivePodcast = LivePodcast;

  // Auto-init on DOMContentLoaded
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => LivePodcast.init());
  } else {
    LivePodcast.init();
  }
})();
