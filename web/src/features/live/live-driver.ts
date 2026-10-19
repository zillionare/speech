// 在线演播驱动：与后端 /ws/live/{session_id} 的帧协议，以及麦克风采集。
// 服务器 → 客户端：JSON 文本帧；audio 帧之后紧接一帧携带完整 WAV 的二进制帧（可能先于 JSON 头到达，需缓冲）。
// 客户端 → 服务器：控制 JSON 帧；录音期间发送 PCM16 单声道 16kHz 的二进制帧。
import type { TranslatorFn } from '@astryxdesign/core/i18n';
import type { LiveAsrStatus } from './live-api';

export interface LiveSegmentStartFrame {
  index: number;
  source: string;
  speaker: string;
  text: string;
  resolved_voice: string;
}

export interface LiveRecordStartFrame {
  index: number;
  target_text: string;
  asr_enabled: boolean;
  asr_ready: boolean;
}

export interface LiveSegmentDoneFrame {
  index: number;
  source: string;
  audio_url: string | null;
}

export interface LiveDriverCallbacks {
  onState: (state: string) => void;
  onSegmentStart: (frame: LiveSegmentStartFrame) => void;
  onRecordStart: (frame: LiveRecordStartFrame) => void;
  onAsrProgress: (status: LiveAsrStatus) => void;
  onAsrPartial: (text: string) => void;
  onAlignment: (ratio: number, matched: number, total: number) => void;
  onSegmentDone: (frame: LiveSegmentDoneFrame) => void;
  onFinished: (audioUrl: string | null) => void;
  onError: (message: string) => void;
  onRestart: (toIndex: number) => void;
  onHint: (message: string, isError?: boolean) => void;
  onMicChange: (active: boolean) => void;
}

// 服务器下发的 JSON 帧字段（仅列出会用到的）。
interface LiveFrame {
  type: string;
  state?: string;
  index?: number;
  source?: string;
  speaker?: string;
  text?: string;
  resolved_voice?: string;
  target_text?: string;
  asr_enabled?: boolean;
  asr_ready?: boolean;
  reason?: string;
  ratio?: number;
  matched?: number;
  total?: number;
  audio_url?: string | null;
  message?: string;
  to_index?: number;
  status?: string;
  progress?: number;
  error?: string;
}

export interface LiveDriverOptions {
  sessionId: string;
  audio: HTMLAudioElement;
  translate: TranslatorFn;
  // 由页面在每次渲染刷新，避免驱动持有陈旧的闭包。
  callbacks: { current: LiveDriverCallbacks | null };
}

export class LiveDriver {
  private ws: WebSocket | null = null;
  private serverState = 'IDLE';
  private currentIndex = -1;
  private micEnabled = false;
  private mediaStream: MediaStream | null = null;
  private audioCtx: AudioContext | null = null;
  private scriptProcessor: ScriptProcessorNode | null = null;
  private audioWorkletNode: AudioWorkletNode | null = null;
  private captureSource: MediaStreamAudioSourceNode | null = null;
  private captureSink: GainNode | null = null;
  private audioWorkletReady = false;
  private pendingWavBlob: ArrayBuffer | null = null;
  private pendingAudioIndex: number | null = null;
  private micPermissionGranted = false;
  private pauseRequested = false;
  private readonly sessionId: string;
  private readonly audio: HTMLAudioElement;
  private readonly translate: TranslatorFn;
  private readonly callbacks: { current: LiveDriverCallbacks | null };

  constructor(options: LiveDriverOptions) {
    this.sessionId = options.sessionId;
    this.audio = options.audio;
    this.translate = options.translate;
    this.callbacks = options.callbacks;
  }

  get state(): string {
    return this.serverState;
  }

  isOpen(): boolean {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  connect(): void {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/live/${this.sessionId}`;
    const ws = new WebSocket(url);
    ws.binaryType = 'arraybuffer';
    ws.onopen = () => this.callbacks.current?.onHint(this.translate('@yovoice.live.connected'));
    ws.onmessage = (event: MessageEvent) => this.onMessage(event);
    ws.onerror = () => this.callbacks.current?.onHint(this.translate('@yovoice.live.connectError'), true);
    ws.onclose = () => {
      if (this.serverState !== 'FINISHED') this.callbacks.current?.onHint(this.translate('@yovoice.live.disconnected'));
      this.stopMic();
    };
    this.ws = ws;
  }

  close(): void {
    this.stopMic();
    if (this.ws) {
      // 主动关闭时不提示“连接已断开”。
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
    this.audio.pause();
  }

  send(frame: Record<string, unknown>): void {
    if (this.isOpen()) this.ws?.send(JSON.stringify(frame));
  }

  // 请求麦克风权限，并在用户手势内预先创建/恢复 AudioContext，
  // 这样后续 RECORDING 段无需再等待一次点击即可开始采集。
  async requestMicPermission(): Promise<void> {
    if (this.micPermissionGranted) return;
    if (!window.isSecureContext) throw new Error('not a secure context');
    const probe = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, sampleRate: { ideal: 16000 } } });
    probe.getTracks().forEach(track => track.stop());
    this.micPermissionGranted = true;
    this.audioCtx = this.audioCtx ?? new AudioContext({ sampleRate: 16000, latencyHint: 'interactive' });
    await this.audioCtx.resume();
  }

  // 暂停/继续：与参考实现一致，暂停时关麦并停住 AI 播放，恢复时续播。
  togglePause(): void {
    if (!this.isOpen()) return;
    const shouldResume = this.serverState === 'PAUSED' || this.pauseRequested;
    this.pauseRequested = !shouldResume;
    if (shouldResume) {
      if (this.serverState === 'PAUSED') void this.audio.play().catch(() => undefined);
      this.send({ type: 'resume' });
    } else {
      this.stopMic();
      this.audio.pause();
      this.send({ type: 'pause' });
    }
  }

  // 从某段重新开始：停麦、停播后通知服务器。
  requestRestart(index: number): void {
    if (!this.isOpen()) return;
    this.stopMic();
    this.audio.pause();
    this.send({ type: 'restart_from', index });
  }

  // 手动结束当前真人段（ASR 未启用或提前说完时使用）。
  sendRecordDone(): void {
    if (this.currentIndex >= 0) this.send({ type: 'record_done', index: this.currentIndex });
  }

  private onMessage(event: MessageEvent): void {
    if (event.data instanceof ArrayBuffer) {
      // WAV 二进制帧：先缓冲，等 audio 头就位后再播放（两者顺序不固定）。
      this.pendingWavBlob = event.data;
      if (this.pendingAudioIndex !== null) {
        const index = this.pendingAudioIndex;
        this.pendingAudioIndex = null;
        this.playAiAudio(index);
      }
      return;
    }
    if (typeof event.data !== 'string') return;
    let frame: LiveFrame;
    try { frame = JSON.parse(event.data) as LiveFrame; } catch { return; }

    switch (frame.type) {
      case 'state': {
        const state = frame.state ?? 'IDLE';
        this.serverState = state;
        this.callbacks.current?.onState(state);
        // 只有 RECORDING 才开麦，其余状态一律关麦。
        if (state === 'RECORDING') void this.startMic();
        else this.stopMic();
        break;
      }
      case 'segment_start': {
        this.currentIndex = frame.index ?? -1;
        this.callbacks.current?.onSegmentStart({
          index: frame.index ?? 0,
          source: frame.source ?? 'tts',
          speaker: frame.speaker ?? '',
          text: frame.text ?? '',
          resolved_voice: frame.resolved_voice ?? '',
        });
        break;
      }
      case 'audio': {
        this.pendingAudioIndex = frame.index ?? 0;
        if (this.pendingWavBlob) {
          const index = this.pendingAudioIndex;
          this.pendingAudioIndex = null;
          this.playAiAudio(index);
        }
        break;
      }
      case 'record_start':
        this.callbacks.current?.onRecordStart({
          index: frame.index ?? 0,
          target_text: frame.target_text ?? '',
          asr_enabled: !!frame.asr_enabled,
          asr_ready: !!frame.asr_ready,
        });
        this.callbacks.current?.onHint(this.translate('@yovoice.live.recordHint'));
        break;
      case 'record_auto_done':
        this.callbacks.current?.onHint(this.translate('@yovoice.live.recordAutoDone'));
        this.stopMic();
        break;
      case 'restart': {
        const to = frame.to_index ?? 0;
        this.currentIndex = to;
        this.stopMic();
        this.callbacks.current?.onRestart(to);
        break;
      }
      case 'asr_progress':
        this.callbacks.current?.onAsrProgress(frame as LiveAsrStatus);
        break;
      case 'asr_unavailable':
        this.callbacks.current?.onAsrProgress({ status: 'disabled', progress: 0, message: frame.reason });
        break;
      case 'asr_partial':
        this.callbacks.current?.onAsrPartial(frame.text ?? '');
        break;
      case 'alignment':
        this.callbacks.current?.onAlignment(Number(frame.ratio ?? 0), frame.matched ?? 0, frame.total ?? 0);
        break;
      case 'segment_done':
        this.callbacks.current?.onSegmentDone({
          index: frame.index ?? 0,
          source: frame.source ?? 'tts',
          audio_url: frame.audio_url ?? null,
        });
        break;
      case 'finished':
        this.serverState = 'FINISHED';
        this.callbacks.current?.onFinished(frame.audio_url ?? null);
        break;
      case 'error':
        this.stopMic();
        this.serverState = 'ERROR';
        this.callbacks.current?.onError(frame.message ?? '');
        break;
    }
  }

  private playAiAudio(index: number): void {
    if (!this.pendingWavBlob) return;
    const blob = new Blob([this.pendingWavBlob], { type: 'audio/wav' });
    this.pendingWavBlob = null;
    const url = URL.createObjectURL(blob);
    const player = this.audio;
    player.pause();
    player.src = url;
    this.callbacks.current?.onHint(this.translate('@yovoice.live.playing', { n: index + 1 }));
    const onEnded = () => {
      player.removeEventListener('ended', onEnded);
      URL.revokeObjectURL(url);
      this.send({ type: 'ai_finished', index });
    };
    player.addEventListener('ended', onEnded);
    if (this.pauseRequested || this.serverState === 'PAUSED') {
      this.callbacks.current?.onHint(this.translate('@yovoice.live.pausedHint'));
    } else {
      void player.play().catch(() => this.send({ type: 'ai_finished', index }));
    }
  }

  // ── 麦克风采集 ──

  private async startMic(): Promise<void> {
    if (this.micEnabled) return;
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: { ideal: 16000 }, echoCancellation: true, noiseSuppression: true },
      });
      this.audioCtx = this.audioCtx ?? new AudioContext({ sampleRate: 16000, latencyHint: 'interactive' });
      await this.audioCtx.resume();
      this.captureSource = this.audioCtx.createMediaStreamSource(this.mediaStream);
      this.captureSink = this.audioCtx.createGain();
      this.captureSink.gain.value = 0;
      this.micEnabled = true;

      if (this.audioCtx.audioWorklet) {
        if (!this.audioWorkletReady) {
          const moduleUrl = this.workletModuleUrl();
          await this.audioCtx.audioWorklet.addModule(moduleUrl);
          URL.revokeObjectURL(moduleUrl);
          this.audioWorkletReady = true;
        }
        const node = new AudioWorkletNode(this.audioCtx, 'cap', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1] });
        node.port.onmessage = (event: MessageEvent) => {
          const data = event.data as { buffer?: ArrayBuffer; sampleRate?: number } | null;
          if (data?.buffer) this.sendPcm16(new Int16Array(data.buffer), data.sampleRate || (this.audioCtx?.sampleRate ?? 16000));
        };
        this.audioWorkletNode = node;
        this.captureSource.connect(node);
        node.connect(this.captureSink).connect(this.audioCtx.destination);
      } else {
        const processor = this.audioCtx.createScriptProcessor(2048, 1, 1);
        processor.onaudioprocess = (event: AudioProcessingEvent) => {
          if (!this.micEnabled) return;
          this.sendFloatPcm(event.inputBuffer.getChannelData(0), this.audioCtx?.sampleRate ?? 16000);
        };
        this.scriptProcessor = processor;
        this.captureSource.connect(processor);
        processor.connect(this.captureSink).connect(this.audioCtx.destination);
      }
      this.callbacks.current?.onMicChange(true);
      this.callbacks.current?.onHint(this.translate('@yovoice.live.micOn'));
    } catch (error) {
      this.micEnabled = false;
      this.callbacks.current?.onHint(this.translate('@yovoice.live.micError', { reason: (error as Error).message }), true);
    }
  }

  private stopMic(): void {
    if (!this.micEnabled && !this.mediaStream && !this.audioWorkletNode && !this.scriptProcessor) return;
    this.micEnabled = false;
    if (this.scriptProcessor) { this.scriptProcessor.onaudioprocess = null; this.scriptProcessor.disconnect(); this.scriptProcessor = null; }
    if (this.audioWorkletNode) { this.audioWorkletNode.port.onmessage = null; this.audioWorkletNode.disconnect(); this.audioWorkletNode = null; }
    if (this.captureSource) { this.captureSource.disconnect(); this.captureSource = null; }
    if (this.captureSink) { this.captureSink.disconnect(); this.captureSink = null; }
    if (this.mediaStream) { this.mediaStream.getTracks().forEach(track => track.stop()); this.mediaStream = null; }
    this.callbacks.current?.onMicChange(false);
  }

  private workletModuleUrl(): string {
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
    const url = URL.createObjectURL(new Blob([source], { type: 'application/javascript' }));
    return url;
  }

  private sendFloatPcm(samples: Float32Array, sampleRate: number): void {
    const out = sampleRate !== 16000 ? this.downsample(samples, sampleRate, 16000) : samples;
    const pcm = new Int16Array(out.length);
    for (let i = 0; i < out.length; i++) pcm[i] = Math.max(-1, Math.min(1, out[i])) * 0x7fff;
    this.sendPcm16(pcm, 16000);
  }

  private sendPcm16(samples: Int16Array, sampleRate: number): void {
    if (!this.micEnabled || !this.isOpen()) return;
    if (sampleRate === 16000) { this.ws?.send(samples.buffer); return; }
    const floats = new Float32Array(samples.length);
    for (let i = 0; i < samples.length; i++) floats[i] = samples[i] / 32768;
    this.sendFloatPcm(floats, sampleRate);
  }

  private downsample(buffer: Float32Array, fromRate: number, toRate: number): Float32Array {
    if (fromRate === toRate) return buffer;
    const ratio = fromRate / toRate;
    const newLen = Math.round(buffer.length / ratio);
    const result = new Float32Array(newLen);
    let offset = 0;
    for (let i = 0; i < newLen; i++) {
      const next = Math.round((i + 1) * ratio);
      let acc = 0;
      let count = 0;
      for (let j = offset; j < next && j < buffer.length; j++) { acc += buffer[j]; count++; }
      result[i] = count > 0 ? acc / count : 0;
      offset = next;
    }
    return result;
  }
}
