import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { ChatComposer, ChatMessage, ChatMessageBubble, ChatMessageList, ChatSendButton } from '@astryxdesign/core/Chat';
import { HStack, VStack } from '@astryxdesign/core/Layout';
import { List, ListItem } from '@astryxdesign/core/List';
import { Token } from '@astryxdesign/core/Token';
import { useTranslator } from '@astryxdesign/core/i18n';
import { ChevronDown, ChevronLeft, Maximize2, Minimize2, PanelLeftClose, PanelLeftOpen, SquarePen, TextQuote, Trash2 } from 'lucide-react';
import { call, mediaUrl } from '../../shared/lib/client';
import type { AssistantBlock, AssistantContext, State, Generation } from '../../shared/workbench';
import { messageBlocks, turnCards } from './assistant-model';
import { AssistantTurn } from './assistant-turn';
import type { usePanelFrame } from './assistant-frame';
import { AssistantAvatar } from './assistant-avatar';
import { clearAssistantSelection, takeAssistantFocus, useAssistantSelection } from './assistant-focus';
import { Selector } from '../../shared/selector';
import voiceArt from '../../assets/assistant/empty-voice.webp';
import musicArt from '../../assets/assistant/empty-music.webp';
import storyArt from '../../assets/assistant/empty-story.webp';

// 助手面板：浮在工作区上的对话卡片，拖标题栏移动、拖左上边角缩放；放大只拉长到窗口高度、宽度不变，向下箭头收起成入口。
// 对话按会话组织（标题处切换历史），发送、停止、新建、删除经 assistant.* RPC；运行中的一轮按内容块实时渲染。

/** 助手要求打开某作品时由应用层执行（面板自己不掌握页面状态）。 */
export interface AssistantPanelActions {
  openWork: (workId: string) => void;
  openSettings: () => void;
}

export interface QuickAction {
  label: string;
  prompt: string;
}

export function AssistantPanel({ state, context, expanded, quickActions, frame, style, onToggleExpand, onClose, actions }: {
  state: State; context: AssistantContext; expanded: boolean; quickActions: QuickAction[];
  frame: ReturnType<typeof usePanelFrame>; style?: CSSProperties;
  onToggleExpand: () => void; onClose: () => void; actions: AssistantPanelActions;
}) {
  const t = useTranslator();
  const live = state.assistantLive;
  const running = !!live?.turn;
  const sessions = state.assistantSessions?.sessions ?? [];
  const current = sessions.find(session => session.id === state.assistantSessions?.currentId);
  const messages = current?.messages ?? [];
  const provider = state.aiProviders?.find(p => p.id === state.aiProviderID);
  const configured = !!provider?.model;
  const runtimeMissing = !!state.assistantMissing;
  const [draft, setDraft] = useState('');
  const [historyOpen, setHistoryOpen] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const audioRef = useRef<{ element: HTMLAudioElement; id: string } | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);
  useEffect(() => () => { audioRef.current?.element.pause(); audioRef.current = null; }, []);
  useEffect(() => { if (running) setHistoryOpen(false); }, [running]);

  const send = (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || !configured || runtimeMissing) return;
    setDraft('');
    setHistoryOpen(false);
    // 选中的文字与播放位置在发送这一刻取出：选区只随这一条消息带上。
    void call('assistant.send', { text: trimmed, context: { ...context, ...takeAssistantFocus() } }).catch(() => undefined);
  };
  const selection = useAssistantSelection();
  const resolveProposal = (id: string, confirm: boolean) => call('assistant.proposal', { id, confirm }).then(() => undefined);
  const newSession = () => {
    if (running) return;
    setHistoryOpen(false);
    void call('assistant.new').catch(() => undefined);
  };
  const openSession = (id: string) => {
    setHistoryOpen(false);
    setDeleting(null);
    void call('assistant.open', { id }).catch(() => undefined);
  };
  const deleteSession = (id: string) => {
    setDeleting(null);
    void call('assistant.delete', { id }).catch(() => undefined);
  };

  const play = async (item: Generation) => {
    const active = audioRef.current;
    if (active && active.id === item.id) {
      active.element.pause();
      audioRef.current = null;
      setPlaying(null);
      return;
    }
    active?.element.pause();
    const url = await mediaUrl('outputs', item.fileName);
    const element = new Audio(url);
    audioRef.current = { element, id: item.id };
    setPlaying(item.id);
    element.onended = () => { if (audioRef.current?.id === item.id) { audioRef.current = null; setPlaying(null); } if (url.startsWith('blob:')) URL.revokeObjectURL(url); };
    await element.play().catch(() => setPlaying(null));
  };

  // 当前作品总是随消息带上；输入框左下角用来切换 AI 服务里的模型，与设置页共用同一份当前模型。
  const choice = (id: string, model: string) => `${id}\n${model}`;
  const modelOptions = (state.aiProviders ?? []).filter(p => p.model || p.models?.length)
    .map(p => ({ type: 'section' as const, title: p.name, options: [...new Set([p.model, ...(p.models ?? [])].filter(Boolean))].map(model => ({ value: choice(p.id, model), label: model })) }));
  const pickModel = (value: string) => { const [id, model] = value.split('\n'); void call('ai.provider.use', { id, model }).catch(() => undefined); };
  const empty = messages.length === 0 && !running;
  const title = current?.title || t('@yovoice.assistant.title');
  const banner = emptyArt[context.workKind || context.page || ''] ?? emptyArt.text;

  // 内容区自己滚动；新内容到来时，只有本来就停在底部才跟随到底，用户往上翻看时不打扰。
  const bodyRef = useRef<HTMLElement | null>(null);
  const pinned = useRef(true);
  const liveBlocks = live?.turn?.blocks;
  useLayoutEffect(() => {
    const body = bodyRef.current;
    if (body && pinned.current) body.scrollTop = body.scrollHeight;
  }, [messages.length, liveBlocks, current?.id, empty]);
  const onScroll = () => {
    const body = bodyRef.current;
    if (body) pinned.current = body.scrollHeight - body.scrollTop - body.clientHeight < 48;
  };

  return <VStack className={`assistant-panel${expanded ? ' expanded' : ''}`} gap={0} style={style} data-moving={frame.moving || undefined} data-testid="assistant-panel" data-open="true" role="complementary" aria-label={t('@yovoice.assistant.title')}>
    <span className="assistant-resize" data-edge="left" aria-hidden onPointerDown={frame.begin('left')} />
    {expanded ? null : <>
      <span className="assistant-resize" data-edge="top" aria-hidden onPointerDown={frame.begin('top')} />
      <span className="assistant-resize" data-edge="top-left" aria-hidden onPointerDown={frame.begin('top-left')} />
    </>}
    <HStack className="assistant-header" vAlign="center" gap={1}
      onPointerDown={frame.begin('move')}
      onDoubleClick={event => { if (!(event.target as HTMLElement).closest('button')) onToggleExpand(); }}>
      {/* 历史会话是从左侧滑出的抽屉，用侧栏图标与右侧的线性图标保持一致。 */}
      <Button data-testid="assistant-history-toggle" size="sm" variant="ghost" isIconOnly icon={historyOpen ? <PanelLeftClose size={15} /> : <PanelLeftOpen size={15} />} label={t('@yovoice.assistant.history')}
        aria-expanded={historyOpen} onClick={() => { setHistoryOpen(open => !open); setDeleting(null); }} />
      <AssistantAvatar working={running} size={24} />
      <span className="assistant-title grow" title={t('@yovoice.assistant.move')}>{title}</span>
      {live?.queued ? <small className="assistant-queued">{t('@yovoice.assistant.queued', { count: live.queued })}</small> : null}
      <Button size="sm" variant="ghost" isIconOnly icon={<SquarePen size={15} />} label={t('@yovoice.assistant.newChat')} isDisabled={running} onClick={newSession} />
      <Button size="sm" variant="ghost" isIconOnly icon={expanded ? <Minimize2 size={15} /> : <Maximize2 size={15} />} label={expanded ? t('@yovoice.assistant.collapse') : t('@yovoice.assistant.expand')} onClick={onToggleExpand} />
      <Button data-testid="assistant-minimize" size="sm" variant="ghost" isIconOnly icon={<ChevronDown size={16} />} label={t('@yovoice.assistant.close')} onClick={onClose} />
    </HStack>
    <VStack className="assistant-body" gap={0} ref={bodyRef} onScroll={onScroll}>
      {empty ? <VStack gap={3} className="assistant-empty" hAlign="center">
        <img className="assistant-empty-art" src={banner} alt="" draggable={false} />
        <p className="assistant-empty-title">{t('@yovoice.assistant.empty')}</p>
        <p className="assistant-empty-hint">{t('@yovoice.assistant.emptyHint')}</p>
        {/* 建议跟着空白引导一起出现，发出第一句后随引导一起消失，不单独占一条，也就不会因收起而让布局跳动。 */}
        {quickActions.length ? <HStack className="assistant-suggestions" gap={2} hAlign="center" wrap="wrap">
          {quickActions.map(action => <Button key={action.label} size="sm" variant="secondary" className="assistant-suggestion" label={action.label}
            onClick={() => send(action.prompt)} />)}
        </HStack> : null}
      </VStack> : <ChatMessageList density="compact" gap={4} align="top" isStreaming={running}>
        {messages.map(message => message.role === 'user'
          ? <ChatMessage key={message.id} sender="user"><ChatMessageBubble>{message.text}</ChatMessageBubble></ChatMessage>
          : <ChatMessage key={message.id} sender="assistant">
            <AssistantTurn blocks={messageBlocks(message)} live={false} message={message} generation={generationCard(messageBlocks(message), state)} playing={playing} onPlay={play} openWork={actions.openWork} onResolve={resolveProposal} />
          </ChatMessage>)}
        {live?.turn ? <ChatMessage key="live" sender="assistant">
          <AssistantTurn blocks={live.turn.blocks ?? []} live generation={null} playing={null} onPlay={() => undefined} openWork={() => undefined} onResolve={resolveProposal} />
        </ChatMessage> : null}
      </ChatMessageList>}
    </VStack>
    {runtimeMissing ? <HStack className="assistant-notice" role="alert" gap={2} vAlign="center">
      <span className="grow">{t('@yovoice.ai.assistantMissing')}</span>
    </HStack> : !configured ? <HStack className="assistant-notice" gap={2} vAlign="center">
      <span className="grow">{t('@yovoice.assistant.setup')}</span>
      <Button size="sm" variant="secondary" label={t('@yovoice.ai.configure')} onClick={actions.openSettings} />
    </HStack> : <VStack className="assistant-dock-bottom" gap={0}>
      <ChatComposer className="assistant-composer" density="compact" value={draft} onChange={setDraft}
        placeholder={t('@yovoice.assistant.placeholder')} isDisabled={runtimeMissing}
        isStopShown={running} onStop={() => void call('assistant.cancel').catch(() => undefined)}
        onSubmit={send} sendButton={<ChatSendButton className="assistant-send" />}
        // 编辑区里选中的文字会随下一条消息带给 Ocho：以引用标签放在输入框里，点 × 不带。
        headerActions={selection ? <Token data-testid="assistant-selection" className="assistant-selection" size="sm" icon={<TextQuote size={12} />}
          label={selection.replace(/\s+/g, ' ')} description={t('@yovoice.assistant.selectionHint')} onRemove={clearAssistantSelection} /> : undefined}
        footerActions={provider ? <Selector label={t('@yovoice.ai.model')} isLabelHidden size="sm" variant="ghost" className="assistant-model" data-testid="assistant-model"
          value={provider.model ? choice(provider.id, provider.model) : undefined} options={modelOptions} onChange={pickModel}
          hasSearch={modelOptions.reduce((n, section) => n + section.options.length, 0) > 12} searchPlaceholder={t('@yovoice.ai.searchModels')} /> : undefined} />
    </VStack>}
    {/* 历史是从左侧滑出的抽屉，右边留一条遮罩露出当前对话，点遮罩或返回即收起。 */}
    {historyOpen ? <>
      <span className="assistant-scrim" aria-hidden onClick={() => setHistoryOpen(false)} />
      <VStack className="assistant-history" gap={0} data-testid="assistant-history" role="dialog" aria-label={t('@yovoice.assistant.history')}>
        {/* 抽屉头：左侧返回，右侧新对话；列表每行一条会话，标题单行截断，条数和时间在下一行，删除按钮悬停时出现。 */}
        <HStack className="assistant-history-header" vAlign="center" gap={1}>
          <Button size="sm" variant="ghost" isIconOnly icon={<ChevronLeft size={16} />} label={t('@yovoice.assistant.back')} onClick={() => setHistoryOpen(false)} />
          <span className="assistant-title grow">{t('@yovoice.assistant.history')}</span>
          <Button size="sm" variant="ghost" isIconOnly icon={<SquarePen size={15} />} label={t('@yovoice.assistant.newChat')} tooltip={t('@yovoice.assistant.newChat')} isDisabled={running} onClick={newSession} />
        </HStack>
        <VStack className="assistant-history-list" gap={0}>
          {sessions.length === 0 ? <small className="assistant-history-empty">{t('@yovoice.assistant.historyEmpty')}</small> : <List density="compact">
            {sessions.map(session => <ListItem key={session.id} className="assistant-history-item" isSelected={session.id === current?.id}
              label={session.title || t('@yovoice.assistant.newChat')}
              description={`${t('@yovoice.assistant.messageCount', { count: session.messages.length })} · ${relativeTime(t, session.updatedAt)}`}
              onClick={() => openSession(session.id)}
              endContent={deleting === session.id
                ? <Button size="sm" variant="destructive" label={t('@yovoice.assistant.deleteConfirm')} onClick={() => deleteSession(session.id)} />
                : <Button size="sm" variant="ghost" isIconOnly className="assistant-history-delete" icon={<Trash2 size={14} />} label={t('@yovoice.assistant.deleteChat', { title: session.title || t('@yovoice.assistant.newChat') })} onClick={() => setDeleting(session.id)} />} />)}
          </List>}
        </VStack>
      </VStack>
    </> : null}
  </VStack>;
}

// 空状态插图：按当前页面换一张（台词/语音、音乐/编曲、故事配音）。
const emptyArt: Record<string, string> = { text: voiceArt, voices: voiceArt, music: musicArt, score: musicArt, story: storyArt };

// generationCard 找到 generate_work 已完成且该作品已有版本的结果，供试听。
function generationCard(blocks: AssistantBlock[], state: State): Generation | null {
  const cards = turnCards(blocks).filter(card => card.tool === 'generate_work');
  if (!cards.some(card => card.state === 'done')) return null;
  const workID = cards.map(card => workIDFromArgs(card.args)).find(Boolean);
  return workID ? state.history.find(item => item.settings.id === workID) ?? null : null;
}

// workIDFromArgs 从工具卡片的参数摘要里取出作品 ID（generate_work 的 args 形如 {"id":"…"}）。
function workIDFromArgs(args?: string): string {
  if (!args) return '';
  try {
    const id = (JSON.parse(args) as { id?: string }).id;
    return typeof id === 'string' ? id : '';
  } catch { return ''; }
}

// relativeTime 显示会话的相对更新时间。
function relativeTime(t: ReturnType<typeof useTranslator>, iso: string): string {
  const minutes = Math.max(0, Math.round((Date.now() - Date.parse(iso)) / 60000));
  if (minutes < 1) return t('@yovoice.assistant.justNow');
  if (minutes < 60) return t('@yovoice.assistant.minutesAgo', { count: minutes });
  const hours = Math.round(minutes / 60);
  if (hours < 24) return t('@yovoice.assistant.hoursAgo', { count: hours });
  return new Date(iso).toLocaleDateString();
}
