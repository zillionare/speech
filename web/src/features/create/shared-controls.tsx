import type { ReactNode } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { HStack, VStack } from '@astryxdesign/core/Layout';
import { useTranslator } from '@astryxdesign/core/i18n';
import { AudioLines, ChevronRight, Play } from 'lucide-react';
import type { Draft, Track, Voice } from '../../shared/workbench';

export function ReferenceVoice({ voice, choose, play, label, placeholder }: {
  voice?: Voice; choose: () => void; play: (track: Track) => void; label?: string; placeholder?: string;
}) {
  const t = useTranslator();
  return <VStack gap={3}>
    <h3>{label ?? t('@yovoice.create.referenceVoice')}</h3>
    <HStack className="voice-selected" gap={3} vAlign="center">
      <Button label={voice?.name ?? placeholder ?? t('@yovoice.create.addReference')} icon={<AudioLines className="voice-mark" />} variant="ghost" className="voice-title grow" onClick={choose} />
      {voice ? <Button label={t('@yovoice.create.previewVoice')} isIconOnly icon={<Play />} variant="ghost" onClick={() => play({ ...voice, kind: 'voices', subtitle: t('@yovoice.app.subtitleReference') })} /> : null}
    </HStack>
  </VStack>;
}

export function AdvancedSettings({ open, onOpenChange, children }: { open: boolean; onOpenChange: (open: boolean) => void; children: ReactNode }) {
  const t = useTranslator();
  return <details open={open} onToggle={event => onOpenChange(event.currentTarget.open)} className="advanced">
    <summary><ChevronRight aria-hidden="true" />{t('@yovoice.create.advanced')}</summary>
    <VStack gap={4} paddingBlockStart={4}>{children}</VStack>
  </details>;
}

export function SeedInput({ value, change }: { value: Draft['seed']; change: (patch: Partial<Draft>) => void }) {
  const t = useTranslator();
  return <label className="number-field">{t('@yovoice.create.seed')}<input type="number" min={0} max={2147483647} value={value ?? ''} placeholder={t('@yovoice.create.seedAuto')} onChange={event => change({ seed: event.target.value ? Number(event.target.value) : null })} /></label>;
}
