import { useState } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { Switch } from '@astryxdesign/core/Switch';
import { VStack } from '@astryxdesign/core/Layout';
import { useTranslator } from '@astryxdesign/core/i18n';
import { Trash2 } from 'lucide-react';
import { call } from '../../shared/lib/client';
import type { State } from '../../shared/workbench';
import { SettingsGroup, SettingsRow } from '../../shared/ui/settings-group';

// 设置 › Ocho：主动提示开关，以及 Ocho 记住的偏好（逐条删除或全部清除）。偏好由 Ocho 在对话里记下，这里只查看与删除。
export function AssistantSettings({ state, run }: { state: State; run: (task: () => Promise<unknown>) => void }) {
  const t = useTranslator();
  const preferences = state.preferences;
  const memory = [...(state.assistantMemory ?? [])].reverse();
  const [clearing, setClearing] = useState(false);
  return <VStack className="settings-panel" gap={6} id="assistant-panel" role="tabpanel" aria-label={t('@yovoice.assistant.title')}>
    <small className="settings-intro">{t('@yovoice.settings.assistantIntro')}</small>
    <SettingsGroup title={t('@yovoice.settings.assistantBehavior')}>
      <SettingsRow label={t('@yovoice.settings.assistantNudges')} hint={t('@yovoice.settings.assistantNudgesHint')}>
        <Switch label={t('@yovoice.settings.assistantNudges')} isLabelHidden value={!preferences.assistantQuiet}
          onChange={enabled => run(() => call('preferences.save', { ...preferences, assistantQuiet: !enabled }))} />
      </SettingsRow>
    </SettingsGroup>
    <SettingsGroup title={t('@yovoice.settings.assistantMemory')} description={t('@yovoice.settings.assistantMemoryHint')}
      actions={memory.length ? clearing
        ? <Button size="sm" variant="destructive" label={t('@yovoice.settings.assistantMemoryClearConfirm')} onClick={() => { setClearing(false); run(() => call('assistant.forget', { id: '' })); }} />
        : <Button size="sm" variant="ghost" label={t('@yovoice.settings.assistantMemoryClear')} onClick={() => setClearing(true)} /> : undefined}>
      {memory.length ? memory.map(memo => <SettingsRow key={memo.id} className="assistant-memo" label={memo.text} hint={new Date(memo.createdAt).toLocaleDateString()}>
        <Button size="sm" variant="ghost" isIconOnly icon={<Trash2 size={14} />} label={t('@yovoice.settings.assistantMemoryDelete', { text: memo.text })}
          onClick={() => run(() => call('assistant.forget', { id: memo.id }))} />
      </SettingsRow>) : <SettingsRow label={t('@yovoice.settings.assistantMemoryEmpty')} hint={t('@yovoice.settings.assistantMemoryEmptyHint')} />}
    </SettingsGroup>
  </VStack>;
}
