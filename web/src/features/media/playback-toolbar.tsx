import type { ReactNode } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { HStack } from '@astryxdesign/core/Layout';
import { useTranslator } from '@astryxdesign/core/i18n';
import { Play, Pause, ZoomIn, ZoomOut } from 'lucide-react';
import { formatTime } from '../../shared/workbench';

export function PlaybackToolbar({ time, duration, playing, disabled, loading, toggle, beforePlay, afterPlay, children }: {
  time: number; duration: number; playing: boolean; disabled: boolean; loading?: boolean; toggle: () => void;
  beforePlay?: ReactNode; afterPlay?: ReactNode; children: ReactNode;
}) {
  const t = useTranslator();
  return <HStack className="transport" vAlign="center" gap={4} wrap="wrap">
    <small className="transport-time">{formatTime(time)} <em>/ {formatTime(duration)}</em></small>
    <HStack className="transport-controls" gap={2} vAlign="center">
      {beforePlay}
      <Button label={t(playing ? '@yovoice.player.pause' : '@yovoice.player.play')} isIconOnly icon={playing ? <Pause /> : <Play fill="currentColor" />} size="sm" variant="primary" className="play-main" isDisabled={disabled} isLoading={loading} onClick={toggle} />
      {afterPlay}
    </HStack>
    <HStack className="transport-end" gap={3} vAlign="center" wrap="wrap">{children}</HStack>
  </HStack>;
}

export function TrackZoom({ value, change, disabled, min = 1, max = 8, fit }: { value: number; change: (value: number) => void; disabled: boolean; min?: number; max?: number; fit?: () => void }) {
  const t = useTranslator();
  return <HStack className="track-zoom" gap={0} vAlign="center" role="group" aria-label={t('@yovoice.player.zoomGroup')}>
    <Button label={t('@yovoice.player.zoomOut')} isIconOnly icon={<ZoomOut />} size="sm" variant="ghost" isDisabled={disabled || value <= min} onClick={() => change(Math.max(min, value / 2))} />
    <Button label={`${Number((value * 100).toFixed(2))}%`} aria-label={t('@yovoice.player.zoomFit')} size="sm" variant="ghost" isDisabled={disabled} onClick={() => fit ? fit() : change(1)} />
    <Button label={t('@yovoice.player.zoomIn')} isIconOnly icon={<ZoomIn />} size="sm" variant="ghost" isDisabled={disabled || value >= max} onClick={() => change(Math.min(max, value * 2))} />
  </HStack>;
}
