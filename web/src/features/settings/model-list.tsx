import { useState } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { HStack, VStack } from '@astryxdesign/core/Layout';
import { SegmentedControl, SegmentedControlItem } from '@astryxdesign/core/SegmentedControl';
import { StatusDot } from '@astryxdesign/core/StatusDot';
import { TextInput } from '@astryxdesign/core/TextInput';
import { useTranslator } from '@astryxdesign/core/i18n';
import { CircleCheck, Download, Pause, Search, Trash2 } from 'lucide-react';
import { call } from '../../shared/lib/client';
import { formatSize, groupModels, type Draft, type ModelPackage, type State } from '../../shared/workbench';
import { SettingsGroup } from '../../shared/ui/settings-group';

function downloadStatusLabel(t: (key: string) => string, downloading: boolean, received: number, size: number, status: string) {
  if (downloading) return received >= size ? t('@yovoice.settings.download.verifying') : t('@yovoice.settings.download.running');
  if (status === 'failed') return t('@yovoice.settings.download.failed');
  return t('@yovoice.settings.download.paused');
}

function modelDescription(t: (key: string) => string, model: ModelPackage) {
  if (model.family === 'soundfont') return t(({ 'generaluser-gs-sf2': '@yovoice.settings.modelDesc.soundfontGeneralUser', 'sgm-v2-sf2': '@yovoice.settings.modelDesc.soundfontSgm' } as Record<string, string>)[model.id] ?? '@yovoice.settings.modelDesc.soundfont');
  if (model.family === 'ace_step') return t('@yovoice.settings.modelDesc.music');
  if (model.family === 'yue2') return t('@yovoice.settings.modelDesc.yue2');
  if (model.family === 'sheetsage2') return t('@yovoice.settings.modelDesc.sheetsage2');
  if (model.family === 'kokoro_tts') return t(model.id === 'kokoro-82m-q8' ? '@yovoice.settings.modelDesc.kokoroOfficial' : '@yovoice.settings.modelDesc.kokoro');
  if (model.family === 'omnivoice') return t('@yovoice.settings.modelDesc.omni');
  if (model.family === 'qwen3_tts') return t(`@yovoice.settings.modelDesc.qwen.${model.variant || 'base'}`);
  if (model.family === 'voxcpm2') return t('@yovoice.settings.modelDesc.vox');
  if (model.family === 'breeze_tts') return t('@yovoice.settings.modelDesc.breeze');
  return model.version === '2.5' ? t('@yovoice.settings.modelDesc.25') : t('@yovoice.settings.modelDesc.basic');
}

// 模型与音色库共用：同一模型的各种量化合并为一行，用精度切换；有任一精度已下载的排在“已下载”组，其余放在“可下载”组。
export function ModelGroups({ models, state, draft, run, focus }: { models: ModelPackage[]; state: State; draft: Draft; run: (task: () => Promise<unknown>) => void; focus?: string }) {
  const t = useTranslator();
  // 生成等主操作与模型下载、导入互不阻塞：下载按钮只看模型安装是否空闲。
  const busy = state.activity?.status === 'running';
  const modelBusy = state.modelActivity?.status === 'running';
  const isInstalled = (model: ModelPackage) => state.models.some(m => m.id === model.id);
  const groups = groupModels(models);
  const installed = groups.filter(group => group.some(isInstalled));
  const available = groups.filter(group => !installed.includes(group));
  // 每组当前选中的精度。
  const [picks, setPicks] = useState<Record<string, string>>({});
  const activity = state.modelActivity;
  const downloadOf = (model: ModelPackage) => activity?.kind === 'download' && (activity.modelId === model.id || (!activity.modelId && activity.total === model.size)) && activity.status !== 'completed' ? activity : null;
  const picked = (group: ModelPackage[]) => group.find(model => model.id === picks[group[0].name])
    ?? group.find(model => model.id === focus) ?? group.find(downloadOf) ?? group.find(model => model.id === draft.modelId)
    ?? group.find(isInstalled) ?? group.find(model => model.precision === 'Q8') ?? group[0];
  // 可下载的模型较多时按名称、精度和用途搜索。
  const [query, setQuery] = useState('');
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  const matches = available.filter(group => group.some(model => { const text = `${model.name} ${model.precision} ${model.family} ${modelDescription(t, model)}`.toLowerCase(); return terms.every(term => text.includes(term)); }));
  const row = (group: ModelPackage[]) => {
    const model = picked(group);
    const ready = isInstalled(model);
    const download = downloadOf(model);
    const downloading = download?.status === 'running';
    const removeBlocked = busy || (modelBusy && !(activity?.kind === 'download' && activity.modelId && activity.modelId !== model.id));
    const inUse = draft.modelId === model.id;
    const owned = group.filter(isInstalled);
    // 应用下载的模型移除时连同文件删除；导入的模型只解除登记。
    const managed = state.models.find(m => m.id === model.id)?.managed ?? false;
    return <VStack className="model-row settings-row" data-model-id={model.id} data-model-ids={group.map(item => item.id).join(' ')} data-installed={ready} key={group[0].name} gap={3}>
      <HStack className="model-summary" gap={4} vAlign="center" wrap="wrap">
        <VStack className="grow" gap={1}>
          <HStack gap={2} vAlign="center" wrap="wrap">
            {owned.length ? <StatusDot variant={group.some(item => item.id === draft.modelId) ? 'accent' : 'success'} label={group.some(item => item.id === draft.modelId) ? t('@yovoice.settings.inUse') : t('@yovoice.settings.verified')} /> : null}
            <p className="settings-row-title">{model.name}</p>
            {group.length === 1 ? <small className="precision">{model.precision}</small> : null}
            <small className="model-size">{formatSize(model.size)}</small>
            {ready ? <small className="ready">{inUse ? t('@yovoice.settings.inUse') : t('@yovoice.settings.verified')}</small> : null}
          </HStack>
          <small>{modelDescription(t, model)}</small>
        </VStack>
        {downloading ? <Button size="sm" label={t('@yovoice.action.pause')} icon={<Pause size={16} />} onClick={() => run(() => call('model.cancel'))} />
          : ready ? <Button size="sm" variant="ghost" label={t('@yovoice.settings.removeModel')} icon={<Trash2 size={16} />} isDisabled={removeBlocked} tooltip={removeBlocked ? t('@yovoice.settings.removeBlocked') : t(managed ? '@yovoice.settings.removeManagedTooltip' : '@yovoice.settings.removeTooltip')} onClick={() => run(() => call('model.forget', { id: model.id }))} />
          : <Button size="sm" label={download ? t('@yovoice.settings.resumeDownload') : t('@yovoice.settings.downloadModel')} icon={<Download size={16} />} isDisabled={modelBusy} onClick={() => run(() => call('model.download', { id: model.id }))} />}
      </HStack>
      {group.length > 1 ? <HStack className="model-precisions" gap={3} vAlign="center" wrap="wrap">
        <SegmentedControl size="sm" label={t('@yovoice.settings.precisionOf', { name: model.name })} value={model.id} onChange={id => setPicks(current => ({ ...current, [model.name]: id }))}>
          {group.map(item => <SegmentedControlItem key={item.id} value={item.id} label={item.precision} icon={isInstalled(item) ? <CircleCheck /> : undefined} />)}
        </SegmentedControl>
        {owned.length ? <small>{t('@yovoice.settings.precisionsInstalled', { list: owned.map(item => item.precision).join(' · ') })}</small> : <small>{t('@yovoice.settings.precisionHint')}</small>}
      </HStack> : null}
      {download ? <VStack className="model-download" gap={2} role="status" aria-label={t('@yovoice.settings.downloadProgress', { name: model.name, precision: model.precision })}>
        <HStack hAlign="between" gap={3}><small>{downloadStatusLabel(t, !!downloading, download.received, model.size, download.status)}</small><small className="model-size">{formatSize(download.received)} / {formatSize(model.size)}</small></HStack>
        <progress aria-label={t('@yovoice.settings.downloadProgress', { name: model.name, precision: model.precision })} value={download.received} max={model.size} />
      </VStack> : null}
    </VStack>;
  };
  return <>
    {installed.length ? <SettingsGroup title={`${t('@yovoice.settings.downloaded')} · ${installed.length}`} className="model-list">{installed.map(row)}</SettingsGroup> : null}
    {available.length ? <SettingsGroup title={t('@yovoice.settings.available')} className="model-list available"
      actions={available.length > 5 ? <TextInput size="sm" className="model-search" label={t('@yovoice.settings.searchModels')} isLabelHidden startIcon={<Search />} placeholder={t('@yovoice.settings.searchModels')} value={query} onChange={setQuery} width="calc(var(--spacing-10) * 6)" /> : undefined}>
      {matches.length ? matches.map(row) : <p className="settings-row model-search-empty">{t('@yovoice.settings.noModelMatch')}</p>}
    </SettingsGroup> : null}
  </>;
}
