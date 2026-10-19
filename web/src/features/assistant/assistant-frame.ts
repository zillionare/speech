import { useCallback, useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';

// 助手浮窗的位置与尺寸：以视口右下角为锚点（窗口缩放时贴着右下角走），可拖动标题栏移动、拖动边角缩放，
// 结果存到本机；每次变化都夹回视口内，保证标题栏始终够得着。

/** 浮窗几何：right/bottom 是到视口右边、下边的距离，单位为 CSS 像素。 */
export interface PanelRect { right: number; bottom: number; width: number; height: number }

/** 缩放手柄的方向：从左、上、左上角拉伸（锚点在右下，右下两边不动）。 */
export type ResizeEdge = 'left' | 'top' | 'top-left';

const storageKey = 'yovoice-assistant-rect-v4';
const margin = 8;
const minWidth = 280;
const minHeight = 360;

// cssLength 解析界面尺寸变量（变量里可能是 calc()，用一个临时元素量出像素值）。
function cssLength(name: string): number {
  const probe = document.createElement('span');
  probe.style.cssText = `position:fixed;visibility:hidden;height:var(${name})`;
  document.body.appendChild(probe);
  const height = probe.getBoundingClientRect().height;
  probe.remove();
  return height;
}

// 默认落在右侧配置区那一列，右边和底边与声音面板（时间线）外沿对齐，左右各比配置区内容宽出一截。
function defaultRect(): PanelRect {
  const shellGap = cssLength('--spacing-2');
  const inset = cssLength('--spacing-3');
  const width = cssLength('--app-inspector') - cssLength('--spacing-6') - 1;
  return { right: shellGap + inset, bottom: inset, width, height: 600 };
}

function readRect(): PanelRect {
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) ?? 'null') as PanelRect | null;
    if (saved && [saved.right, saved.bottom, saved.width, saved.height].every(Number.isFinite)) return saved;
  } catch { /* 存储不可用或内容损坏时用默认值 */ }
  return defaultRect();
}

// clampRect 把浮窗夹回视口：尺寸不超过视口，四边留出边距，顶部不压住标题栏。
function clampRect(rect: PanelRect): PanelRect {
  const top = cssLength('--app-titlebar');
  const maxWidth = Math.max(minWidth, window.innerWidth - margin * 2);
  const maxHeight = Math.max(minHeight, window.innerHeight - top - margin * 2);
  const width = Math.min(Math.max(rect.width, minWidth), maxWidth);
  const height = Math.min(Math.max(rect.height, minHeight), maxHeight);
  const right = Math.min(Math.max(rect.right, margin), window.innerWidth - width - margin);
  const bottom = Math.min(Math.max(rect.bottom, margin), window.innerHeight - top - height - margin);
  return { right: Math.max(right, 0), bottom: Math.max(bottom, 0), width, height };
}

export function usePanelFrame() {
  const [rect, setRect] = useState<PanelRect>(() => clampRect(readRect()));
  const [moving, setMoving] = useState(false);
  const drag = useRef<{ x: number; y: number; start: PanelRect; edge: ResizeEdge | 'move' } | null>(null);

  useEffect(() => {
    const onResize = () => setRect(current => clampRect(current));
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  useEffect(() => {
    if (moving) return;
    try { localStorage.setItem(storageKey, JSON.stringify(rect)); } catch { /* 存储不可用时只在本次生效 */ }
  }, [rect, moving]);

  const begin = useCallback((edge: ResizeEdge | 'move') => (event: ReactPointerEvent<HTMLElement>) => {
    if (event.button !== 0) return;
    // 标题栏上的按钮照常点击，只有空白处才开始拖动。
    if (edge === 'move' && (event.target as HTMLElement).closest('button, a, input, textarea')) return;
    event.preventDefault();
    drag.current = { x: event.clientX, y: event.clientY, start: rect, edge };
    setMoving(true);
    const onMove = (move: PointerEvent) => {
      const active = drag.current;
      if (!active) return;
      const dx = move.clientX - active.x;
      const dy = move.clientY - active.y;
      const { start } = active;
      const next = active.edge === 'move'
        ? { ...start, right: start.right - dx, bottom: start.bottom - dy }
        : {
          ...start,
          width: active.edge === 'top' ? start.width : start.width - dx,
          height: active.edge === 'left' ? start.height : start.height - dy,
        };
      setRect(clampRect(next));
    };
    const onUp = () => {
      drag.current = null;
      setMoving(false);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  }, [rect]);

  return { rect, moving, begin };
}
