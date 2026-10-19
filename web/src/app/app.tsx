import { useEffect, useState } from 'react';
import { call, subscribe } from '../shared/lib/client';
import { noticeMessage } from '../shared/lib/call-error';
import { createDraft, emptyState, isVoiceModel, normalizeDraft, projectKind, type EngineInfo, type ModelPackage, type State } from '../shared/workbench';
import { isUiLocale } from '../shared/i18n/locale';
import { LocaleShell, ensureUiLocalePersisted, bootLocale } from './locale-shell';
import { useDraftSession } from './use-draft-session';
import { WorkbenchChrome } from './workbench-chrome';

// 应用入口：加载并订阅后端状态，恢复上次打开的作品；界面语言随偏好切换。
export function App() {
  const [state, setState] = useState<State>(emptyState); const [catalog, setCatalog] = useState<ModelPackage[]>([]); const [engine, setEngine] = useState<EngineInfo>({ version: '', minimum: '' });
  const [ready, setReady] = useState(false);
  const [error, setError] = useState('');
  const session = useDraftSession(ready, state, setError);
  const { setDraft, setDraftPersisted } = session;
  useEffect(() => {
    // 助手运行时可用性由 state.get 一次性带回，之后每次状态推送都保留。
    const unsubscribe = subscribe(s => setState(current => ({ ...s, assistantMissing: current.assistantMissing })));
    void call<{ state: State; catalog: ModelPackage[]; engine?: EngineInfo; assistant?: { available: boolean } }>('state.get').then(async result => {
      let next = result.state;
      if (!isUiLocale(next.preferences.uiLocale)) {
        try {
          await ensureUiLocalePersisted(next.preferences, bootLocale);
          next = { ...next, preferences: { ...next.preferences, uiLocale: bootLocale } };
        } catch (e) { setError(noticeMessage(e)); }
      }
      const defaultModelId = next.models.find(m => isVoiceModel(m.id))?.id ?? result.catalog[0]?.id ?? 'index-2.5-q8';
      const savedId = localStorage.getItem('yovoice-active-project');
      const sorted = [...next.drafts].sort((a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? ''));
      const saved = savedId ? next.drafts.find(d => d.id === savedId) : undefined;
      const initial = saved ?? sorted.find(d => projectKind(d) === 'text') ?? sorted[0];
      setState({ ...next, assistantMissing: result.assistant ? !result.assistant.available : undefined }); setCatalog(result.catalog); setEngine(result.engine ?? { version: '', minimum: '' }); setDraft(normalizeDraft(initial ?? createDraft(true, next.preferences.uiLocale, defaultModelId))); setDraftPersisted(!!initial);
      setReady(true);
    }).catch(e => setError(noticeMessage(e)));
    return unsubscribe;
  }, [setDraft, setDraftPersisted]);
  return <LocaleShell preferencesLocale={state.preferences.uiLocale}>
    <WorkbenchChrome state={state} catalog={catalog} engine={engine} ready={ready} session={session} error={error} setError={setError} />
  </LocaleShell>;
}
