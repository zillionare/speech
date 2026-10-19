import { AdvancedSettings, SeedInput, ReferenceVoice } from './shared-controls';
import { ModelOptions } from './model-options';
import { VStack } from '@astryxdesign/core/Layout';
import { SegmentedControl, SegmentedControlItem } from '@astryxdesign/core/SegmentedControl';
import { TextArea } from '@astryxdesign/core/TextArea';
import { useTranslator } from '@astryxdesign/core/i18n';
import type { Draft, State, Track } from '../../shared/workbench';

// Breeze TTS 2：声音设计用文字描述音色；克隆需要参考音频与完整原文，描述改作演绎指导。
export function BreezeControls({ draft, state, change, chooseVoice, play, advanced, setAdvanced }: {
  draft: Draft; state: State; change: (patch: Partial<Draft>) => void; chooseVoice: () => void;
  play: (track: Track) => void; advanced: boolean; setAdvanced: (value: boolean) => void;
}) {
  const t = useTranslator();
  const cloning = draft.voiceMode === 'clone';
  const voice = state.voices.find(item => item.id === draft.voiceId);
  return <>
    <VStack gap={3}>
      <h3>{t('@yovoice.vox.generationMode')}</h3>
      <SegmentedControl label={t('@yovoice.vox.generationMode')} size="sm" layout="fill" value={cloning ? 'clone' : 'design'} onChange={value => change({ voiceMode: value as Draft['voiceMode'] })}>
        <SegmentedControlItem value="design" label={t('@yovoice.vox.design')} />
        <SegmentedControlItem value="clone" label={t('@yovoice.vox.clone')} />
      </SegmentedControl>
    </VStack>
    {cloning ? <>
      <ReferenceVoice voice={voice} choose={chooseVoice} play={play} />
      <TextArea label={t('@yovoice.vox.referenceText')} description={t('@yovoice.breeze.referenceTextHelp')} value={draft.referenceText ?? ''} onChange={referenceText => change({ referenceText: referenceText.slice(0, 2000) })} placeholder={t('@yovoice.vox.referenceTextPlaceholder')} rows={3} />
    </> : null}
    <TextArea label={cloning ? t('@yovoice.vox.styleGuidance') : t('@yovoice.vox.voiceDescription')} description={t('@yovoice.breeze.instructionHelp')} value={draft.voiceDescription ?? ''} onChange={voiceDescription => change({ voiceDescription: voiceDescription.slice(0, 500) })} placeholder={cloning ? t('@yovoice.vox.stylePlaceholder') : t('@yovoice.breeze.voiceDescriptionPlaceholder')} rows={3} />
    <AdvancedSettings open={advanced} onOpenChange={setAdvanced}>
      <ModelOptions draft={draft} family="breeze_tts" change={change} />
      <SeedInput value={draft.seed} change={change} />
    </AdvancedSettings>
  </>;
}
