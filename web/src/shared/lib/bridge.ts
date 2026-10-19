import type { Draft, State } from '../workbench';
import { CallError, parseCallError } from './call-error';

declare global { interface Window { __workbenchPlatform?: 'macos'; __workbenchMediaBase?: string; __workbenchDraft?: Draft; chrome?: { webview?: { postMessage: (data: unknown) => void; addEventListener: (name: string, listener: (event: MessageEvent) => void) => void } } } }

// 桌面宿主（WebView2 / WKWebView）注入的消息通道；浏览器预览中不存在。
const native = window.chrome?.webview;
export const isDesktop = !!native;
export const isMac = window.__workbenchPlatform === 'macos';

const listeners = new Set<(state: State) => void>();
const pending = new Map<string, { resolve: (value: unknown) => void; reject: (reason: Error) => void; timer: ReturnType<typeof setTimeout> | undefined }>();
const updateListeners = new Set<(update: unknown) => void>();
native?.addEventListener('message', ({ data }) => {
  if (data.event === 'state') { listeners.forEach(fn => fn(data.state)); return; }
  if (data.event === 'update') { updateListeners.forEach(fn => fn(data.update)); return; }
  const callback = pending.get(data.id);
  if (!callback) return;
  clearTimeout(callback.timer); pending.delete(data.id);
  if (data.error) callback.reject(parseCallError(data.error)); else callback.resolve(data.result);
});

export function subscribeDesktop(listener: (state: State) => void) { listeners.add(listener); return () => { listeners.delete(listener); }; }

// 宿主推送的应用更新状态，结构见 features/update/app-update.ts。
export function subscribeDesktopUpdate(listener: (update: unknown) => void) { updateListeners.add(listener); return () => { updateListeners.delete(listener); }; }

// 数据目录迁移在同一次调用里等待用户选择目录并同步复制全部数据，几 GB 模型远超 120 秒；
// 超时会先报未知错误而后台仍在复制，结束后又提示重启，因此只由宿主侧限时。
const longCalls = new Set(['storage.directory']);

export function desktopCall<T>(method: string, data: unknown): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const id = crypto.randomUUID();
    const timer = longCalls.has(method) ? undefined : setTimeout(() => {
      pending.delete(id); reportDesktop('timeout', method, 'no reply within 120s');
      reject(new CallError('@yovoice.error.unknown', { detail: 'desktop timeout' }));
    }, 120000);
    pending.set(id, { resolve: value => resolve(value as T), reject, timer }); native!.postMessage({ id, method, data });
  });
}

// 界面侧异常写入服务日志；只上报不等回复，上报失败也不影响界面。
export function reportDesktop(event: string, method: string, detail: string) {
  try { native?.postMessage({ id: crypto.randomUUID(), method: 'diag.client', data: { event, method, detail: detail.slice(0, 2000) } }); } catch { /* 宿主不可用时放弃上报。 */ }
}

export function desktopMediaUrl(kind: 'voices' | 'outputs', file: string): string {
  const base = window.__workbenchMediaBase;
  return base ? `${base}${kind}/${encodeURIComponent(file)}` : `https://${kind}.workbench.local/${encodeURIComponent(file)}`;
}
