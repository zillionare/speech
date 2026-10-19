import type { PointerEvent } from 'react';

// 悬浮提示只在文字被截断时出现：进入时检查自身及子元素是否溢出，溢出才挂上完整文字。
export function overflowTitle(event: PointerEvent<HTMLElement>) {
  const element = event.currentTarget;
  const clipped = [element, ...element.querySelectorAll<HTMLElement>('*')].some(node => node.scrollWidth > node.clientWidth + 1 || node.scrollHeight > node.clientHeight + 1);
  const text = element.dataset.fullText ?? element.textContent?.trim() ?? '';
  if (clipped && text) element.title = text;
  else element.removeAttribute('title');
}
