import { useSyncExternalStore } from 'react';

// 用户此刻关注的内容：最近一次在助手面板之外选中的文字与播放位置。点进助手输入框时编辑区的选区会丢，
// 所以在选区变化时就记下来；发送一条消息后选中内容随之用掉，避免下一句话又带上旧选区。

const selectionLimit = 1200;
let selection = '';
let playhead = 0;
let installed = false;
const listeners = new Set<() => void>();

function setSelection(next: string) {
  const text = next.trim().slice(0, selectionLimit);
  if (text === selection) return;
  selection = text;
  listeners.forEach(listener => listener());
}

// 助手自己的界面（面板、入口、提示气泡）里的选区与焦点不算用户在编辑区的选择。
const insideAssistant = (node: Node | null | undefined) => {
  const element = node instanceof Element ? node : node?.parentElement;
  return !!element?.closest('.assistant-panel, .assistant-launcher-wrap, .assistant-nudge');
};

function onSelectionChange() {
  const active = document.activeElement;
  if (insideAssistant(active)) return;
  if (active instanceof HTMLTextAreaElement || (active instanceof HTMLInputElement && active.type === 'text')) {
    const { selectionStart: start, selectionEnd: end } = active;
    setSelection(start !== null && end !== null && end > start ? active.value.slice(start, end) : '');
    return;
  }
  const current = document.getSelection();
  if (!current || insideAssistant(current.anchorNode)) return;
  setSelection(current.toString());
}

/** 开始记录选区；重复调用只安装一次。 */
export function trackAssistantFocus() {
  if (installed || typeof document === 'undefined') return;
  installed = true;
  // 有的内核不为输入框里的选区触发 selectionchange，鼠标与键盘松开时再补一次。
  for (const event of ['selectionchange', 'mouseup', 'keyup']) document.addEventListener(event, onSelectionChange, true);
}

/** 播放器上报当前播放位置（秒）。 */
export function reportPlayhead(time: number) { playhead = Number.isFinite(time) ? Math.max(0, time) : 0; }

/** 发送消息时取出选中内容与播放位置；选中内容只用一次。 */
export function takeAssistantFocus(): { selection?: string; playhead?: number } {
  const focus = { selection: selection || undefined, playhead: playhead > 0 ? Math.round(playhead * 10) / 10 : undefined };
  setSelection('');
  return focus;
}

/** 丢弃记下的选区（输入框上的选区提示里点 ×）。 */
export function clearAssistantSelection() { setSelection(''); }

/** 当前记下的选区，用于在输入框上方提示“已附带选中内容”。 */
export function useAssistantSelection() {
  return useSyncExternalStore(listener => { listeners.add(listener); return () => { listeners.delete(listener); }; }, () => selection, () => '');
}
