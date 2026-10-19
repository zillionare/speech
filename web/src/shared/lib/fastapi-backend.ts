import type { AudioAsset, EngineInfo, State, Voice } from '../workbench';
import catalog from './catalog.json';
import { toBase64 } from './sound';
import { CallError, parseCallError } from './call-error';

// 界面直接跑在 FastAPI 服务（Speech Studio）上：RPC 走 /api/yovoice/call，状态经 /ws/yovoice 推送。
// 契约与桌面宿主桥（bridge.ts）一致，方法名与 preview-backend.ts 相同，便于两边对照。
const apiRoot = '/api/yovoice';
const socketPath = '/ws/yovoice';
const maxUploadBytes = 20 * 1024 * 1024;

async function rpc<T>(method: string, data: unknown = {}): Promise<T> {
  const response = await fetch(`${apiRoot}/call`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ method, data }),
  });
  if (!response.ok) throw new CallError('@yovoice.error.unknown', { detail: `HTTP ${response.status}` });
  const payload = (await response.json()) as { result?: unknown; error?: unknown };
  if (payload.error) throw parseCallError(payload.error);
  return payload.result as T;
}

// 状态自带界面词条目录与运行库信息，浏览器端没有随包助手运行时。
export async function fastapiCall<T>(method: string, data: unknown = {}): Promise<T> {
  if (method !== 'state.get') return rpc<T>(method, data);
  const result = await rpc<{ state: State; engine?: EngineInfo }>(method, data);
  return {
    state: result.state,
    catalog,
    engine: result.engine ?? { version: '', minimum: '' },
    desktop: false,
    assistant: { available: false },
  } as unknown as T;
}

// 状态推送：单个共享 WebSocket，断开后按退避重连，直到界面不再订阅。
const listeners = new Set<(state: State) => void>();
let socket: WebSocket | null = null;
let retries = 0;
let retryTimer: ReturnType<typeof setTimeout> | undefined;

function socketUrl(): string {
  const url = new URL(socketPath, window.location.href);
  url.protocol = url.protocol === 'https:' ? 'wss:' : 'ws:';
  return url.toString();
}

function scheduleReconnect() {
  if (!listeners.size) return;
  clearTimeout(retryTimer);
  retryTimer = setTimeout(connect, Math.min(500 * 2 ** retries++, 8000));
}

function connect() {
  if (typeof WebSocket === 'undefined') return;
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;
  const next = new WebSocket(socketUrl());
  socket = next;
  next.onopen = () => { retries = 0; };
  next.onmessage = (event) => {
    try {
      const message = JSON.parse(String(event.data)) as { event?: string; state?: State };
      if (message.event === 'state' && message.state) {
        const state = message.state;
        listeners.forEach((listener) => listener(state));
      }
    } catch { /* 无法解析的推送直接忽略。 */ }
  };
  next.onclose = () => { if (socket === next) socket = null; scheduleReconnect(); };
  next.onerror = () => { next.close(); };
}

export function subscribeFastapi(listener: (state: State) => void) {
  listeners.add(listener);
  connect();
  return () => {
    listeners.delete(listener);
    if (listeners.size) return;
    clearTimeout(retryTimer);
    retries = 0;
    const current = socket;
    socket = null;
    current?.close();
  };
}

// 媒体统一经服务端解析：voices 指向音色库，outputs 支持 `podcast~<作品>~<文件>` 子目录标记。
export function fastapiMediaUrl(kind: 'voices' | 'outputs', file: string): string {
  if (!file) return '';
  if (file.startsWith('/')) return file;
  return `${apiRoot}/media/${kind}/${encodeURIComponent(file)}`;
}

export function assertUploadSize(file: Blob) {
  if (file.size > maxUploadBytes) throw new CallError('@yovoice.error.audioTooLarge');
}

export async function importTimelineUpload(file: File, name: string): Promise<AudioAsset> {
  assertUploadSize(file);
  return rpc<AudioAsset>('timeline.import', { name, base64: await toBase64(file) });
}

export async function importVoiceUpload(file: File): Promise<Voice> {
  assertUploadSize(file);
  return rpc<Voice>('voice.record', {
    name: file.name.replace(/\.[^.]+$/, '').slice(0, 120),
    base64: await toBase64(file),
  });
}

export { CallError };
