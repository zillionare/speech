// 在线演播页：左侧会话列表、中间分段与编辑、底部状态栏。
// UI 只负责状态与渲染，帧协议、麦克风采集与 AI 播放都在 live-driver.ts。
import { useCallback, useEffect, useRef, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { Button } from '@astryxdesign/core/Button';
import { HStack, VStack } from '@astryxdesign/core/Layout';
import { TextArea } from '@astryxdesign/core/TextArea';
import { TextInput } from '@astryxdesign/core/TextInput';
import { useTranslator } from '@astryxdesign/core/i18n';
import { Download, Pause, Play, Plus, Square, Volume2 } from 'lucide-react';
import { AppDialog } from '../../shared/ui/app-dialog';
import {
  deleteLiveSession, exportLiveSession, getLiveSession, listLiveSessions, prefetchLiveTts,
  startLiveSession, updateLiveSegment,
  type LiveAsrStatus, type LiveSegment, type LiveSegmentStatus, type LiveSession,
} from './live-api';
import { LiveDriver, type LiveDriverCallbacks } from './live-driver';

const PREVIEW_FILENAME = 'live-studio.wav';

function formatDate(value: string): string {
  return value ? value.slice(5, 16).replace('T', ' ') : '';
}

function download(url: string, filename: string): void {
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.rel = 'noopener';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

export function LiveStudio() {
  const t = useTranslator();

  const [sessions, setSessions] = useState<LiveSession[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [segments, setSegments] = useState<LiveSegment[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [editText, setEditText] = useState('');
  const [runState, setRunState] = useState('IDLE');
  const [hint, setHint] = useState('');
  const [hintIsError, setHintIsError] = useState(false);
  const [asrStatus, setAsrStatus] = useState<LiveAsrStatus>({});
  const [asrPartial, setAsrPartial] = useState('');
  const [alignment, setAlignment] = useState<number | null>(null);
  const [micActive, setMicActive] = useState(false);
  const [previewMode, setPreviewMode] = useState(false);
  const [starting, setStarting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const [hasFinal, setHasFinal] = useState(false);
  const [newOpen, setNewOpen] = useState(false);
  const [newTitle, setNewTitle] = useState('');
  const [newText, setNewText] = useState('');
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState('');

  const driverRef = useRef<LiveDriver | null>(null);
  const playbackRef = useRef<HTMLAudioElement>(null);
  const previewRef = useRef<HTMLAudioElement>(null);
  const handlersRef = useRef<LiveDriverCallbacks | null>(null);
  const segmentsRef = useRef<LiveSegment[]>([]);
  const selectedIndexRef = useRef<number | null>(null);
  const editTextRef = useRef('');
  const editTimer = useRef<number | null>(null);
  const previewActiveRef = useRef(false);
  const previewQueueRef = useRef<LiveSegment[]>([]);
  const previewIndexRef = useRef(0);

  useEffect(() => { segmentsRef.current = segments; }, [segments]);
  useEffect(() => { selectedIndexRef.current = selectedIndex; }, [selectedIndex]);

  const closeDriver = useCallback(() => { driverRef.current?.close(); driverRef.current = null; }, []);

  // 停止试听：清空队列、复位播放元素。
  const stopPreview = useCallback(() => {
    previewActiveRef.current = false;
    const element = previewRef.current;
    if (element) { element.pause(); element.removeAttribute('src'); element.load(); element.onended = null; element.onerror = null; }
    previewQueueRef.current = [];
    previewIndexRef.current = 0;
    setPreviewMode(false);
    setHint(t('@yovoice.live.previewStopped'));
  }, [t]);

  // 载入会话：断开旧驱动、停播试听，再取详情并复位界面。
  const loadSession = useCallback(async (sid: string) => {
    closeDriver();
    stopPreview();
    try {
      const data = await getLiveSession(sid);
      const first = data.segments[0];
      setSessionId(data.session_id);
      setSegments(data.segments);
      setSelectedIndex(first ? first.index : null);
      setEditText(first ? first.text : '');
      editTextRef.current = first ? first.text : '';
      setHasFinal(!!data.has_final);
      setAsrStatus({});
      setAsrPartial('');
      setAlignment(null);
      setMicActive(false);
      setStarting(false);
      if (data.state === 'FINISHED' || data.has_final) {
        setRunState('FINISHED');
        setHint(t('@yovoice.live.sessionFinished'));
      } else {
        setRunState('IDLE');
        setHint(t('@yovoice.live.loaded', { count: data.segments.length }));
      }
      setHintIsError(false);
    } catch (error) {
      setHint(t('@yovoice.live.loadFailed', { reason: (error as Error).message }));
      setHintIsError(true);
    }
  }, [closeDriver, stopPreview, t]);

  // 首屏：非安全上下文先提示麦克风，再列出会话并打开最新一条。
  useEffect(() => {
    if (!window.isSecureContext) {
      setHint(t('@yovoice.live.insecure', { host: location.hostname || 'localhost' }));
      setHintIsError(true);
    }
    let cancelled = false;
    void (async () => {
      const list = await listLiveSessions().catch(() => [] as LiveSession[]);
      if (cancelled) return;
      setSessions(list);
      if (list.length > 0) void loadSession(list[0].session_id);
    })();
    return () => { cancelled = true; };
  }, [loadSession, t]);

  // 卸载时收尾，避免遗留连接、采集与播放。
  useEffect(() => () => { driverRef.current?.close(); driverRef.current = null; previewRef.current?.pause(); }, []);

  const driving = runState === 'AI_SPEAKING' || runState === 'RECORDING' || runState === 'PAUSED';
  const previewable = segments.some(segment => !!segment.audio_url);

  function ensureDriver(): LiveDriver | null {
    if (!sessionId || !playbackRef.current) return null;
    if (!driverRef.current) {
      driverRef.current = new LiveDriver({ sessionId, audio: playbackRef.current, translate: t, callbacks: handlersRef });
    }
    return driverRef.current;
  }

  function resetFrom(index: number): void {
    setSegments(list => list.map(segment => segment.index >= index ? { ...segment, status: 'pending', audio_url: null } : segment));
    setSelectedIndex(index);
    const segment = segmentsRef.current.find(item => item.index === index);
    const text = segment ? segment.text : '';
    setEditText(text);
    editTextRef.current = text;
    setAsrPartial('');
  }

  function requestRestart(index: number): void {
    const driver = driverRef.current;
    if (!driver || !driver.isOpen()) return;
    driver.requestRestart(index);
    resetFrom(index);
    setHint(t('@yovoice.live.restarting', { n: index + 1 }));
    setHintIsError(false);
  }

  // 点击分段：演播中点击其他段即从该段重录，否则只做选中与编辑。
  function selectSegment(index: number, fromServer = false): void {
    if (!fromServer && driving && index !== selectedIndex) { requestRestart(index); return; }
    setSelectedIndex(index);
    const segment = segmentsRef.current.find(item => item.index === index);
    const text = segment ? segment.text : '';
    setEditText(text);
    editTextRef.current = text;
    if (editTimer.current !== null) { window.clearTimeout(editTimer.current); editTimer.current = null; }
    setAsrPartial('');
  }

  // 编辑防抖 2 秒后同步，失焦立即同步。
  function scheduleEditSync(value: string): void {
    editTextRef.current = value;
    if (editTimer.current !== null) window.clearTimeout(editTimer.current);
    editTimer.current = window.setTimeout(() => { editTimer.current = null; flushEditSync(); }, 2000);
  }

  function flushEditSync(): void {
    if (editTimer.current !== null) { window.clearTimeout(editTimer.current); editTimer.current = null; }
    const index = selectedIndexRef.current;
    if (index === null || !sessionId) return;
    const text = editTextRef.current.trim();
    const segment = segmentsRef.current.find(item => item.index === index);
    if (!segment || segment.text === text) return;
    setSegments(list => list.map(item => item.index === index ? { ...item, text } : item));
    void updateLiveSegment(sessionId, index, text).catch(() => undefined);
    setHint(t('@yovoice.live.segmentUpdated', { n: index + 1 }));
    setHintIsError(false);
  }

  async function refreshSessions(): Promise<LiveSession[]> {
    const list = await listLiveSessions().catch(() => [] as LiveSession[]);
    setSessions(list);
    return list;
  }

  function clearSession(): void {
    setSessionId(null);
    setSegments([]);
    setSelectedIndex(null);
    setEditText('');
    editTextRef.current = '';
    setRunState('IDLE');
    setAsrStatus({});
    setAsrPartial('');
    setAlignment(null);
    setMicActive(false);
    setHasFinal(false);
  }

  async function handleDelete(sid: string, event: ReactMouseEvent<HTMLElement>): Promise<void> {
    event.stopPropagation();
    try {
      await deleteLiveSession(sid);
      const list = await refreshSessions();
      if (sid === sessionId) {
        closeDriver();
        stopPreview();
        if (list.length > 0) void loadSession(list[0].session_id);
        else { clearSession(); setHint(t('@yovoice.live.allDeleted')); }
      } else {
        setHint(t('@yovoice.live.deleted'));
      }
      setHintIsError(false);
    } catch (error) {
      setHint(t('@yovoice.live.deleteFailed', { reason: (error as Error).message }));
      setHintIsError(true);
    }
  }

  async function handleCreate(): Promise<void> {
    const text = newText.trim();
    if (!text) { setCreateError(t('@yovoice.live.needScript')); return; }
    setCreating(true);
    setCreateError('');
    try {
      const data = await startLiveSession(text, newTitle.trim());
      closeDriver();
      stopPreview();
      const first = data.segments[0];
      setSessionId(data.session_id);
      setSegments(data.segments);
      setSelectedIndex(first ? first.index : null);
      setEditText(first ? first.text : '');
      editTextRef.current = first ? first.text : '';
      setHasFinal(false);
      setAsrStatus(data.asr ?? {});
      setAsrPartial('');
      setAlignment(null);
      setRunState('IDLE');
      setNewOpen(false);
      setNewText('');
      setNewTitle('');
      setHint(t('@yovoice.live.created', { count: data.segments.length }));
      setHintIsError(false);
      void refreshSessions();
    } catch (error) {
      setCreateError((error as Error).message);
    } finally {
      setCreating(false);
    }
  }

  async function handleStart(): Promise<void> {
    if (!sessionId) return;
    if (runState === 'FINISHED') { setHint(t('@yovoice.live.alreadyFinished')); setHintIsError(true); return; }
    const driver = ensureDriver();
    if (!driver) return;
    // 含真人段时先申请权限，并在用户手势内预热 AudioContext。
    if (segments.some(segment => segment.source === 'live')) {
      setHint(t('@yovoice.live.requestMic'));
      setHintIsError(false);
      try { await driver.requestMicPermission(); }
      catch (error) { setHint(t('@yovoice.live.micDenied', { reason: (error as Error).message })); setHintIsError(true); return; }
    }
    setStarting(true);
    void prefetchLiveTts(sessionId).catch(() => undefined);
    driver.connect();
  }

  function handleTogglePause(): void {
    driverRef.current?.togglePause();
  }

  function handleFinishRecording(): void {
    driverRef.current?.sendRecordDone();
  }

  function playPreviewSegment(): void {
    if (!previewActiveRef.current || previewIndexRef.current >= previewQueueRef.current.length) { stopPreview(); return; }
    const segment = previewQueueRef.current[previewIndexRef.current];
    setHint(t('@yovoice.live.previewSegment', { n: segment.index + 1 }));
    const element = previewRef.current;
    if (!element) { stopPreview(); return; }
    element.src = segment.audio_url ?? '';
    element.onended = () => { if (!previewActiveRef.current) return; previewIndexRef.current += 1; playPreviewSegment(); };
    element.onerror = () => { if (!previewActiveRef.current) return; previewIndexRef.current += 1; playPreviewSegment(); };
    void element.play().catch(() => stopPreview());
  }

  function startPreview(): void {
    const list = segments.filter(segment => !!segment.audio_url).sort((a, b) => a.index - b.index);
    if (list.length === 0) { setHint(t('@yovoice.live.noPreview')); setHintIsError(true); return; }
    previewActiveRef.current = true;
    previewQueueRef.current = list;
    previewIndexRef.current = 0;
    setPreviewMode(true);
    setHint(t('@yovoice.live.previewing'));
    setHintIsError(false);
    playPreviewSegment();
  }

  function handleTogglePreview(): void {
    if (previewMode) stopPreview();
    else startPreview();
  }

  async function handleExport(): Promise<void> {
    if (!sessionId) return;
    setExporting(true);
    setHint(t('@yovoice.live.exporting'));
    setHintIsError(false);
    try {
      const data = await exportLiveSession(sessionId);
      download(data.audio_url, PREVIEW_FILENAME);
      setHasFinal(true);
      setHint(t('@yovoice.live.exported'));
    } catch (error) {
      setHint(t('@yovoice.live.exportFailed', { reason: (error as Error).message }));
      setHintIsError(true);
    } finally {
      setExporting(false);
    }
  }

  function statusLabel(status: LiveSegmentStatus): string {
    switch (status) {
      case 'generating': return t('@yovoice.live.status.generating');
      case 'generated': return t('@yovoice.live.status.generated');
      case 'recorded': return t('@yovoice.live.status.recorded');
      case 'missing': return t('@yovoice.live.status.missing');
      case 'error': return t('@yovoice.live.status.error');
      default: return t('@yovoice.live.status.pending');
    }
  }

  function stateLabel(state: string): string {
    switch (state) {
      case 'AI_SPEAKING': return t('@yovoice.live.state.aiSpeaking');
      case 'RECORDING': return t('@yovoice.live.state.recording');
      case 'PAUSED': return t('@yovoice.live.state.paused');
      case 'FINISHED': return t('@yovoice.live.state.finished');
      case 'ERROR': return t('@yovoice.live.state.error');
      default: return t('@yovoice.live.state.idle');
    }
  }

  // 每次渲染刷新回调，驱动始终拿到最新的状态闭包。
  handlersRef.current = {
    onState: state => { setRunState(state); setStarting(false); },
    onSegmentStart: frame => {
      setSegments(list => list.map(item => item.index === frame.index ? { ...item, status: frame.source === 'live' ? 'pending' : 'generating' } : item));
      selectSegment(frame.index, true);
    },
    onRecordStart: () => { setAsrPartial(''); setAlignment(null); },
    onAsrProgress: status => setAsrStatus(status),
    onAsrPartial: text => setAsrPartial(text),
    onAlignment: ratio => setAlignment(ratio),
    onSegmentDone: frame => {
      const status: LiveSegmentStatus = frame.source === 'live' ? (frame.audio_url ? 'recorded' : 'missing') : 'generated';
      setSegments(list => list.map(item => item.index === frame.index ? { ...item, status, audio_url: frame.audio_url } : item));
      setHint(t('@yovoice.live.segmentDone', { n: frame.index + 1 }));
      setHintIsError(false);
    },
    onFinished: audioUrl => {
      setRunState('FINISHED');
      setHasFinal(!!audioUrl);
      setHint(t('@yovoice.live.allDone'));
      setHintIsError(false);
    },
    onError: message => {
      const index = selectedIndexRef.current;
      if (index !== null) setSegments(list => list.map(item => item.index === index ? { ...item, status: 'error' } : item));
      setHint(message);
      setHintIsError(true);
      setStarting(false);
    },
    onRestart: toIndex => {
      resetFrom(toIndex);
      setHint(t('@yovoice.live.restarted', { n: toIndex + 1 }));
      setHintIsError(false);
    },
    onHint: (message, isError) => { setHint(message); setHintIsError(!!isError); },
    onMicChange: active => setMicActive(active),
  };

  const asrState = asrStatus.status ?? 'idle';
  const showAsr = !(asrState === 'disabled' || asrState === 'ready' || asrState === 'idle' || (asrStatus.enabled === false && asrState === 'idle'));
  const asrProgress = Math.max(0, Math.min(100, Number(asrStatus.progress ?? 0) * 100));
  const canStart = !!sessionId && segments.length > 0 && (runState === 'IDLE' || runState === 'ERROR') && !starting;
  const canExport = !!sessionId && (runState === 'FINISHED' || hasFinal) && !exporting;
  const stateClass = runState === 'AI_SPEAKING' ? 'generated' : runState === 'RECORDING' ? 'recorded' : runState === 'FINISHED' ? 'generated' : runState === 'ERROR' ? 'error' : 'pending';

  return <VStack className="live-page" gap={0}>
    <HStack className="live-header" gap={2} vAlign="center">
      <h2 className="grow">{t('@yovoice.nav.live')}</h2>
      <Button size="sm" variant="primary" icon={<Plus />} label={t('@yovoice.live.new')} data-testid="live-new" onClick={() => setNewOpen(true)} />
    </HStack>

    <HStack className="live-toolbar" gap={2} vAlign="center" wrap="wrap">
      <Button size="sm" variant="primary" icon={<Play />} label={t('@yovoice.live.start')} data-testid="live-start" isDisabled={!canStart} isLoading={starting} onClick={() => void handleStart()} />
      <Button size="sm" icon={runState === 'PAUSED' ? <Play /> : <Pause />} label={runState === 'PAUSED' ? t('@yovoice.live.resume') : t('@yovoice.live.pause')} data-testid="live-pause" isDisabled={!driving} onClick={handleTogglePause} />
      {runState === 'RECORDING' ? <Button size="sm" icon={<Square />} label={t('@yovoice.live.finishRecording')} data-testid="live-finish-record" onClick={handleFinishRecording} /> : null}
      <Button size="sm" icon={<Volume2 />} label={previewMode ? t('@yovoice.live.previewStop') : t('@yovoice.live.preview')} data-testid="live-preview" isDisabled={!previewMode && !previewable} onClick={handleTogglePreview} />
      <Button size="sm" icon={<Download />} label={t('@yovoice.live.export')} data-testid="live-export" isDisabled={!canExport} isLoading={exporting} onClick={() => void handleExport()} />
    </HStack>

    <HStack className="live-body" gap={0}>
      <VStack className="live-sessions" gap={1}>
        {sessions.length === 0 ? <p className="muted live-empty">{t('@yovoice.live.empty')}</p> : sessions.map(session => (
          <HStack key={session.session_id} className={`live-session ${session.session_id === sessionId ? 'active' : ''}`} gap={1} vAlign="center">
            <button type="button" className="live-session-link" data-testid={`live-session-${session.session_id}`} onClick={() => void loadSession(session.session_id)}>
              <span className="live-session-title">{session.title || session.first_text || session.session_id}</span>
              <span className="live-session-meta eyebrow">{formatDate(session.created_at)} · {t('@yovoice.live.segmentCount', { count: session.segment_count })}{session.has_final ? ` · ${t('@yovoice.live.finishedMark')}` : ''}</span>
            </button>
            <button type="button" className="live-session-delete" data-testid={`live-session-delete-${session.session_id}`} aria-label={t('@yovoice.action.delete')} onClick={event => void handleDelete(session.session_id, event)}>×</button>
          </HStack>
        ))}
      </VStack>
      <VStack className="live-stage" gap={0}>
        <VStack className="live-segments" gap={1}>
          {segments.map(segment => (
            <HStack key={segment.index} className={`live-seg-row ${segment.index === selectedIndex ? 'selected' : ''}`} gap={2} vAlign="center">
              <button type="button" className="live-seg-main" data-testid={`live-segment-${segment.index}`} onClick={() => selectSegment(segment.index)}>
                <span className="seg-num">{String(segment.index + 1).padStart(2, '0')}</span>
                <span className="seg-speaker">{segment.speaker || t('@yovoice.live.narrator')}</span>
                <span className={`seg-tag ${segment.source}`}>{segment.source === 'live' ? t('@yovoice.live.source.live') : t('@yovoice.live.source.ai')}</span>
                <span className="seg-text">{segment.text}</span>
                <span className={`seg-status live-seg-status ${segment.status}`}>{statusLabel(segment.status)}</span>
              </button>
              {segment.audio_url ? <audio className="live-seg-audio" controls preload="none" src={segment.audio_url} data-testid={`live-segment-audio-${segment.index}`} /> : null}
            </HStack>
          ))}
        </VStack>
        <VStack className="live-edit" gap={1}>
          <TextArea size="sm" label={t('@yovoice.live.editLabel')} isLabelHidden placeholder={t('@yovoice.live.selectHint')} value={editText} isDisabled={selectedIndex === null}
            data-testid="live-edit"
            onChange={value => { setEditText(value); scheduleEditSync(value); }}
            onBlur={flushEditSync} />
          <small className="muted">{selectedIndex === null ? t('@yovoice.live.selectHint') : t('@yovoice.live.editHint')}</small>
        </VStack>
      </VStack>
    </HStack>

    <VStack className="live-status" gap={2}>
      <HStack gap={2} vAlign="center">
        <span className={`seg-status live-state ${stateClass}`}>{stateLabel(runState)}</span>
        {micActive ? <span className="live-mic">{t('@yovoice.live.micActive')}</span> : null}
      </HStack>
      <p className={`live-hint ${hintIsError ? 'error' : ''}`} data-testid="live-hint">{hint}</p>
      {showAsr ? <VStack className="live-asr" gap={1}>
        <span className="live-asr-track"><span className={`live-asr-fill ${asrState === 'error' ? 'error' : ''}`} style={{ width: `${asrProgress}%` }} /></span>
        <small className={asrState === 'error' ? 'error' : 'muted'}>{asrStatus.message || t('@yovoice.live.asr.preparing')}</small>
      </VStack> : null}
      {asrPartial ? <p className="live-partial" data-testid="live-asr-partial">{asrPartial}</p> : null}
      {alignment !== null ? <small className="muted live-alignment">{t('@yovoice.live.alignment', { percent: Math.round(alignment * 100) })}</small> : null}
    </VStack>

    <audio ref={playbackRef} hidden />
    <audio ref={previewRef} hidden />

    {newOpen ? <AppDialog title={t('@yovoice.live.newTitle')} busy={creating} error={createError} onClose={() => setNewOpen(false)} actions={<>
      <Button label={t('@yovoice.action.cancel')} isDisabled={creating} onClick={() => setNewOpen(false)} />
      <Button variant="primary" label={t('@yovoice.live.create')} isLoading={creating} isDisabled={!newText.trim()} data-testid="live-create" onClick={() => void handleCreate()} />
    </>}>
      <TextInput label={t('@yovoice.live.titleLabel')} value={newTitle} onChange={setNewTitle} />
      <TextArea label={t('@yovoice.live.scriptLabel')} placeholder={t('@yovoice.live.scriptPlaceholder')} value={newText} onChange={setNewText} />
    </AppDialog> : null}
  </VStack>;
}
