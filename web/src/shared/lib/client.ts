import type { AudioAsset, State, Voice } from '../workbench';
import { toBase64 } from './sound';
import { CallError, parseCallError } from './call-error';
import { desktopCall, desktopMediaUrl, isDesktop, isMac, reportDesktop, subscribeDesktop } from './bridge';
import { assertUploadSize, previewCall, previewImportTimeline, previewImportVoice, previewMediaUrl, subscribePreview } from './preview-backend';

// 界面统一的后端入口：桌面端经宿主桥接调用 Go 服务，浏览器预览使用本地模拟后端。
export { CallError, parseCallError, isDesktop, isMac };

export function subscribe(listener: (state: State) => void) {
  return isDesktop ? subscribeDesktop(listener) : subscribePreview(listener);
}

export function call<T = unknown>(method: string, data: unknown = {}): Promise<T> {
  return isDesktop ? desktopCall<T>(method, data) : previewCall<T>(method, data);
}

export async function importTimelineFile(file: File): Promise<AudioAsset> {
  const name = file.name.replace(/\.[^.]+$/, '').slice(0, 120);
  if (!isDesktop) return previewImportTimeline(file, name);
  assertUploadSize(file);
  return call<AudioAsset>('timeline.import', { name, base64: await toBase64(file) });
}

export async function importVoiceFile(file: File): Promise<Voice> {
  if (!isDesktop) return previewImportVoice(file);
  assertUploadSize(file);
  return call<Voice>('voice.record', { name: file.name.replace(/\.[^.]+$/, ''), base64: await toBase64(file) });
}

// 翻唱扒谱：上传歌曲交给后台用 SheetSage2 提取旋律；浏览器预览不做推理。
export async function coverMelodyFile(projectId: string, file: File): Promise<void> {
  if (!isDesktop) throw new CallError('@yovoice.error.previewDesktopOnly');
  assertUploadSize(file);
  await call('melody.cover', { projectId, base64: await toBase64(file) });
}

export async function mediaUrl(kind: 'voices' | 'outputs', file: string): Promise<string> {
  return isDesktop ? desktopMediaUrl(kind, file) : previewMediaUrl(file);
}

const exportName = (name: string, fallback = 'yovoice') => name.replace(/[\\/:*?"<>|]/g, '_').slice(0, 100) || fallback;

// 浏览器预览没有保存对话框，直接交给浏览器下载。
function download(blob: Blob, fileName: string) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = fileName; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// 只在界面里完成的操作（导出文件等）写入服务日志，事件名记为 client.<event>；上报失败不影响操作本身。
export function logEvent(event: string, detail = '') {
  if (isDesktop) reportDesktop(event, '', detail);
  else void previewCall('diag.client', { event, detail }).catch(() => undefined);
}

// 导出结果写入日志：保存、取消或失败（失败时原样抛出）。
async function logged(format: string, save: () => Promise<boolean>): Promise<boolean> {
  try {
    const saved = await save();
    logEvent(saved ? 'export_saved' : 'export_cancelled', format);
    return saved;
  } catch (error) {
    logEvent('export_failed', `${format}: ${error instanceof Error ? error.message : String(error)}`);
    throw error;
  }
}

export async function saveAudio(blob: Blob, name: string): Promise<boolean> {
  const fileName = `${exportName(name)}.wav`;
  return logged('wav', async () => {
    if (isDesktop) return call<boolean>('audio.export', { name: fileName, base64: await toBase64(blob) });
    download(blob, fileName);
    return true;
  });
}

// 字幕、歌词与 ABC 谱面等 UTF-8 文本：桌面端由宿主弹出保存对话框并写文件，用户取消时返回 false。
export type TextExportFormat = 'srt' | 'vtt' | 'lrc' | 'abc';
export async function saveText(text: string, name: string, format: TextExportFormat): Promise<boolean> {
  const fileName = `${exportName(name)}.${format}`;
  return logged(format, async () => {
    if (isDesktop) return call<boolean>('text.export', { name: fileName, format, text });
    download(new Blob([text], { type: format === 'vtt' ? 'text/vtt' : 'text/plain' }), fileName);
    return true;
  });
}
