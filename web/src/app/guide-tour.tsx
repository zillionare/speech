import { useCallback, useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { HStack, VStack } from '@astryxdesign/core/Layout';
import { useTranslator } from '@astryxdesign/core/i18n';

// 每一步高亮的入口，标题沿用导航文案；目标不在页面上时卡片居中显示。
const steps = [
  { id: 'new', target: 'nav-new', title: '@yovoice.project.new' },
  { id: 'text', target: 'nav-text', title: '@yovoice.project.text' },
  { id: 'story', target: 'nav-story', title: '@yovoice.project.story' },
  { id: 'music', target: 'nav-music', title: '@yovoice.project.music' },
  { id: 'voices', target: 'nav-characters', title: '@yovoice.nav.characters' },
  { id: 'settings', target: 'nav-settings', title: '@yovoice.nav.settings' },
  { id: 'assistant', target: 'assistant-toggle', title: '@yovoice.guide.assistantTitle' },
] as const;
// 高亮框比目标外扩的距离与卡片间距，单位 px，需要按目标位置实时计算。
const pad = 6;
const gap = 16;
const cardWidth = 320;

type Box = { top: number; left: number; width: number; height: number };

/**
 * 功能引导：首次进入主界面时依次高亮侧栏的主要入口，说明各自用途。
 * 可随时跳过（Esc），方向键或按钮前后切换；结束或跳过都记为已看过，设置里可重新查看。
 */
export function GuideTour({ onClose }: { onClose: () => void }) {
  const t = useTranslator();
  const [index, setIndex] = useState(0);
  const [box, setBox] = useState<Box | null>(null);
  const card = useRef<HTMLElement>(null);
  const step = steps[index];
  const last = index === steps.length - 1;
  const target = step.target;

  // 目标可能随窗口缩放或侧栏拖动改变位置，逐帧跟随；侧栏收起时目标不可见，卡片改为居中。
  useLayoutEffect(() => {
    let frame = 0;
    const measure = () => {
      const el = target ? document.querySelector<HTMLElement>(`[data-testid="${target}"]`) : null;
      const r = el?.getBoundingClientRect();
      const next = r && r.width > 0 && r.height > 0 ? { top: r.top - pad, left: r.left - pad, width: r.width + pad * 2, height: r.height + pad * 2 } : null;
      setBox(prev => prev && next && prev.top === next.top && prev.left === next.left && prev.width === next.width && prev.height === next.height ? prev : next);
      frame = requestAnimationFrame(measure);
    };
    measure();
    return () => cancelAnimationFrame(frame);
  }, [target]);

  const next = useCallback(() => { if (last) onClose(); else setIndex(i => i + 1); }, [last, onClose]);
  const prev = useCallback(() => setIndex(i => Math.max(0, i - 1)), []);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); onClose(); }
      else if (event.key === 'ArrowRight') { event.preventDefault(); next(); }
      else if (event.key === 'ArrowLeft') { event.preventDefault(); prev(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [next, prev, onClose]);
  // 每一步把焦点放到主按钮上，回车即可继续。
  useEffect(() => { card.current?.querySelector<HTMLButtonElement>('.guide-primary')?.focus(); }, [index]);

  // 卡片放在目标右侧，上下夹在窗口内；放不下时退到目标下方。
  const cardStyle: CSSProperties | undefined = box ? (() => {
    const right = box.left + box.width + gap;
    const top = Math.min(Math.max(gap, box.top), window.innerHeight - gap - 220);
    return right + cardWidth + gap <= window.innerWidth ? { top, left: right } : { top: Math.min(box.top + box.height + gap, window.innerHeight - gap - 220), left: Math.max(gap, Math.min(box.left, window.innerWidth - cardWidth - gap)) };
  })() : undefined;

  return <VStack as="section" className="guide-tour" role="dialog" aria-modal="true" aria-labelledby="guide-title" aria-describedby="guide-body" gap={0}>
    <span className={`guide-spotlight ${box ? '' : 'guide-spotlight-none'}`} aria-hidden style={box ? { top: box.top, left: box.left, width: box.width, height: box.height } : undefined} />
    <VStack ref={card} className={`guide-card ${box ? '' : 'guide-card-center'}`} style={cardStyle} gap={3}>
      <small className="guide-count">{t('@yovoice.guide.count', { current: index + 1, total: steps.length })}</small>
      <h2 id="guide-title">{t(step.title)}</h2>
      <p id="guide-body">{t(`@yovoice.guide.${step.id}`)}</p>
      <HStack className="guide-actions" hAlign="between" vAlign="center" gap={2}>
        {last ? <span /> : <Button size="sm" variant="ghost" label={t('@yovoice.guide.skip')} onClick={onClose} />}
        <HStack gap={2}>
          {index > 0 ? <Button size="sm" label={t('@yovoice.guide.back')} onClick={prev} /> : null}
          <Button size="sm" variant="primary" className="guide-primary" label={t(last ? '@yovoice.guide.done' : '@yovoice.guide.next')} onClick={next} />
        </HStack>
      </HStack>
    </VStack>
  </VStack>;
}
