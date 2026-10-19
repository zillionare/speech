import { draftCopy } from './i18n/draft-copy';
import { seedLocale, type UiLocale } from './i18n/locale';

export type Mode = 'speaker' | 'reference' | 'vector' | 'text';
export interface SynthesisSettings {
  modelOptions?: Record<string, Record<string, string | number | boolean>>; speaker?: string; synthesisLanguage?: string; omniSpeed?: number;
  lyrics?: string; instrumental?: boolean;
  // YuE2 的 ABC 旋律谱：带和弦按谱编曲，只有旋律时按旋律填配器。
  abc?: string;
  voiceMode?: 'design' | 'clone';
  voxMode?: 'design' | 'clone' | 'continuation'; voiceDescription?: string; referenceText?: string; guidanceScale?: number; inferenceSteps?: number;
  modelId: string; voiceId: string | null;
  mode: Mode; emotionVoiceId: string | null; emotionText: string; inferEmotion: boolean;
  emotionStrength: number; emotions: number[]; randomEmotion: boolean; language: string;
  speed: number; temperature: number; topP: number; topK: number; repetitionPenalty: number;
  maxTokens: number; intervalSilenceMs: number; doSample: boolean; numBeams: number; lengthPenalty: number; seed: number | null;
}
export interface SubtitleSpeaker { id: string; sourceName: string; characterId?: string; settings?: SynthesisSettings }
export interface CharacterPerformance { id: string; name: string; settings: SynthesisSettings }
export interface SubtitleCue { performance?: CharacterPerformance; id?: string; start: number; end: number; text: string; speakerId: string }
export interface SubtitleDocument { speakers: SubtitleSpeaker[]; cues: SubtitleCue[] }
export interface AudioClip { id: string; generationId?: string; assetId?: string; start: number; offset: number; duration: number; gainDb?: number }
export interface AudioLane { id: string; name: string; muted: boolean; solo?: boolean; locked?: boolean; gainDb?: number; duckDb?: number; clips: AudioClip[] }
export interface AudioAsset { id: string; name: string; fileName: string; duration: number }
export interface AudioMarker { id: string; time: number; name: string }
export interface AudioTimeline { markers?: AudioMarker[]; regenerateMode?: 'ripple' | 'preserve'; assets?: AudioAsset[]; acceptedGenerations?: string[]; tracks: AudioLane[] }
// 编曲乐谱与 Go 的 schema.Score 一致：位置以小节和拍计，拍从 1 开始，音高为 MIDI 音高。
export interface ScoreNote { bar: number; beat: number; pitch: number; length: number; velocity: number }
export interface ScoreRamp { start: number; end: number; from: number; to: number }
export interface Humanize { velocity?: number; timingMs?: number }
export type ScoreRole = 'melody' | 'piano' | 'strings' | 'bass' | 'drums' | 'pad' | 'arp' | 'other';
export interface ScoreTrack { id: string; name: string; role?: ScoreRole; program: number; drums?: boolean; level?: number; pan?: number; reverb?: number; mute?: boolean; humanize?: Humanize; dynamics?: ScoreRamp[]; notes: ScoreNote[] }
export interface ScoreSection { name: string; start: number; end: number; tempo?: number }
export interface Score { tempo: number; timeSignature: number[]; key?: string; sections?: ScoreSection[]; tracks: ScoreTrack[] }
export type ProjectKind = 'text' | 'story' | 'subtitle' | 'music' | 'score';
export interface Draft extends SynthesisSettings { performance?: CharacterPerformance; id: string; title: string; text: string; kind?: ProjectKind; characterId?: string; createdAt?: string; updatedAt?: string; subtitles?: SubtitleDocument; timeline?: AudioTimeline; score?: Score }
export type ProjectPage = 'text' | 'story' | 'music' | 'score';
// 旧字幕作品沿用原始数据，统一归入故事。
export const projectKind = (draft: Draft): ProjectPage => draft.kind === 'music' || draft.kind === 'score' ? draft.kind : draft.subtitles || draft.kind === 'story' || draft.kind === 'subtitle' ? 'story' : 'text';
export interface CharacterPreview { id: string; fileName: string; duration: number; settings: SynthesisSettings; text: string }
export interface Character { performances?: CharacterPerformance[]; id: string; name: string; settings: SynthesisSettings; demoText: string; preview?: CharacterPreview; createdAt?: string; updatedAt?: string }
export const synthesisSettings = ({ performance: _performance, id: _id, title: _title, text: _text, kind: _kind, characterId: _characterId, createdAt: _createdAt, updatedAt: _updatedAt, subtitles: _subtitles, timeline: _timeline, score: _score, ...settings }: Draft): SynthesisSettings => structuredClone(settings);
// 参数键的序列化顺序不影响试听是否过期。
export const stableJSON = (value: unknown): string => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);
// 与 Go 的可选字段默认值对齐，保留 seed=0 与自动随机种子的区别。
const comparableSettings = (settings: SynthesisSettings) => ({ speaker: '', synthesisLanguage: '', omniSpeed: 0, voiceMode: '', voxMode: '', voiceDescription: '', referenceText: '', guidanceScale: 0, inferenceSteps: 0, modelOptions: {}, ...Object.fromEntries(Object.entries(settings).filter(([, value]) => value !== undefined)) });
export const previewStale = (c: Character) => !!c.preview && (c.demoText !== c.preview.text || stableJSON(comparableSettings(c.settings)) !== stableJSON(comparableSettings(c.preview.settings)));
// 每种演绎保存独立的模型和参数快照。
export function performanceSettings(base: SynthesisSettings, performance?: CharacterPerformance): SynthesisSettings {
  return structuredClone(performance?.settings ?? base);
}
export function cueSettings(draft: Draft, cue: SubtitleCue): SynthesisSettings {
  return performanceSettings(draft.subtitles?.speakers.find(s => s.id === cue.speakerId)?.settings ?? synthesisSettings(draft), cue.performance);
}
export function cueAudioStatus(draft: Draft, history: Generation[], cue: SubtitleCue): 'missing' | 'stale' | 'ready' {
  const ids = new Set(draft.timeline?.tracks.flatMap(t => t.clips.map(c => c.generationId)) ?? []);
  const generation = history.find(g => ids.has(g.id) && g.segment?.cueId === cue.id);
  if (!generation) return 'missing';
  return generation.settings.text === cue.text && generation.segment?.speakerId === cue.speakerId && stableJSON(comparableSettings(synthesisSettings(generation.settings))) === stableJSON(comparableSettings(cueSettings(draft, cue))) ? 'ready' : 'stale';
}
export interface Voice { referenceText?: string; source?: string; sourceGenerationId?: string; id: string; name: string; fileName: string; duration: number }
export interface ModelFile { path: string; size: number; sha256: string }
export interface ModelPackage { voices?: string[]; variant?: string; loadOptions?: Record<string, string>; sessionOptions?: Record<string, string>; files?: ModelFile[]; directory?: string; task?: string; engineMinimum?: string; family: string; id: string; name: string; version: string; precision: string; remotePath: string; size: number; sha256: string }
export interface InstalledModel { id: string; path: string; managed: boolean }
export interface GenerationSegment { cueId: string; speakerId: string; speakerName: string; batchId: string; index: number; targetClipId?: string; placement?: 'ripple' | 'preserve' }
// 本机 Codex CLI 的检测结果：path 为空表示未安装，login 为 chatgpt、apikey 或空（未登录）。
export interface CodexInfo { path?: string; version?: string; login?: 'chatgpt' | 'apikey' }
// 服务日志的一条记录与查询：级别是下限，模块是事件名第一段；结果新的在前。
export type LogLevel = 'INFO' | 'WARN' | 'ERROR';
export interface LogEntry { time: string; level: LogLevel; event: string; fields?: Record<string, unknown> }
export interface LogFilter { level?: string; module?: string; text?: string; since?: string; offset?: number; limit?: number }
export interface LogPage { entries: LogEntry[]; total: number; modules: string[] }
export interface Generation { segment?: GenerationSegment; id: string; title: string; fileName: string; createdAt: string; duration: number; settings: Draft; lyricTimes?: number[] }
/** UI chrome locale. Distinct from Draft.language (TTS). */
export type { UiLocale };
/** Stable product message key. Catalogs and Go constants share this spelling. */
export type MessageCode = `@yovoice.${string}`;
export type MessageParams = Record<string, unknown>;
export interface Activity {
  cueId?: string;
  projectId?: string;
  characterId?: string;
  requestId?: string;
  startedAt?: string | null;
  modelId?: string | null;
  kind: string;
  // melody 为出谱或扒谱。
  task?: 'melody';
  code: MessageCode;
  params: MessageParams | null;
  status: string;
  received: number;
  total: number;
  errorCode: MessageCode | null;
  errorParams: MessageParams | null;
}
// 本机接口的设置与运行状态，由 localApi.* 调用返回，不属于作品状态。
export interface LocalAPIStatus { enabled: boolean; port: number; token: string; running: boolean; url?: string; error?: { code: MessageCode; params?: MessageParams } }
// 存储空间状态与清理结果，由 storage.* 调用返回。
export interface StorageInfo {
  root: string;
  modelsBytes: number;
  outputsBytes: number;
  voicesBytes: number;
  downloadsBytes: number;
  totalBytes: number;
  pendingPath?: string;
}
export interface CleanupReport {
  cleanedCount: number;
  freedBytes: number;
}
export interface Preferences { downloadSource: string; backend: string; modelDirectory: string | null; uiLocale: UiLocale; proxyURL?: string; proxyEnabled?: boolean; aiTimeout?: number; introSeen?: boolean; tourSeen?: boolean; assistantQuiet?: boolean }
export interface State {
  characters: Character[]; previews: CharacterPreview[]; drafts: Draft[]; voices: Voice[]; models: InstalledModel[]; history: Generation[];
  preferences: Preferences; runtimePath: string | null; runtimeBackend: string | null; runtimeVersion: string | null; runtimes?: Record<string, { path: string; version: string }>; activity: Activity | null; modelActivity?: Activity | null;
  aiProviders?: AIProvider[]; aiProviderID?: string; aiTask?: AITask;
  melody?: MelodyResult;
  /** 助手会话的持久化记录与仅广播的实时状态，与 Go 的 schema.AssistantSessions/Live 一致。 */
  assistantSessions?: AssistantSessions; assistantLive?: AssistantLive; assistantMemory?: AssistantMemo[];
  /** state.get 附带的随包 AI 助手运行时可用性：缺失时 AI 写作入口禁用。 */
  assistantMissing?: boolean;
}

// 引擎写出的旋律谱，与 Go 的 schema.MelodyResult 一致：plan 为按歌词出谱，cover 为从歌曲扒谱；只在本次会话有效。
export interface MelodyResult { id: string; projectId: string; source: 'plan' | 'cover'; abc: string; truncated?: boolean; createdAt: string }
/** 后端解析旋律谱的结果；issue 指出第一处问题所在的小节与声部。 */
export interface MelodyIssue { code: string; line?: number; bar?: number; voice?: string; detail?: string }
export interface MelodyInspection { issue?: MelodyIssue; tempo?: number; meter?: [number, number]; key?: string; bars?: number; seconds?: number; chords?: boolean; score?: Score }

// 可选的大模型服务，与 Go 的 schema.AIProvider 一致；密钥不在状态中，只有掩码。
export interface AITest { ok: boolean; code?: string; params?: Record<string, unknown>; at: string }
// 后台 AI 写作任务，与 Go 的 schema.AITask 一致；只在本次会话有效。
export type AITaskKind = 'cast' | 'image';
export interface AITask {
  id: string; kind: AITaskKind; projectId: string; brief: string; status: 'running' | 'completed' | 'failed' | 'cancelled'; stage: 'writing' | 'repairing'; done?: number; total?: number;
  provider: string; model: string; timeout: number; startedAt: string; finishedAt?: string; result?: unknown; errorCode?: string; errorParams?: Record<string, unknown>;
}
export const aiTimeoutDefault = 600;
export interface AIProvider { id: string; preset: string; name: string; format: string; baseURL: string; modelsPath: string; model: string; models?: string[]; keyMask?: string; lastTest?: AITest }

// 助手会话记录与实时状态，与 Go 的 schema.Assistant* 一致。
export type AssistantMessageRole = 'user' | 'assistant';
export type AssistantToolCardState = 'running' | 'done' | 'error';
export interface AssistantToolCard { tool: string; summary?: string; args?: string; state?: AssistantToolCardState; proposal?: AssistantProposal }
/** 助手提议、等用户在卡片上确认的操作，与 Go 的 schema.AssistantProposal 一致。 */
export type AssistantProposalState = 'pending' | 'confirmed' | 'dismissed' | 'failed';
export interface AssistantProposal { id: string; action: string; target?: string; value?: string; label?: string; reason?: string; state: AssistantProposalState; errorCode?: string; errorParams?: Record<string, unknown> }
/** 助手记住的一条用户偏好。 */
export interface AssistantMemo { id: string; text: string; createdAt: string }
/** 助手消息里按出现顺序排列的一块：文本、思考或一次工具调用。 */
export interface AssistantBlock { kind: 'text' | 'thinking' | 'tool'; text?: string; card?: AssistantToolCard }
/** 助手消息：blocks 为按顺序的内容块，text 为文本块拼接；tools 只出现在旧版本记录里。 */
export interface AssistantMessage {
  id: string; role: AssistantMessageRole; text?: string; blocks?: AssistantBlock[]; tools?: AssistantToolCard[];
  status?: 'completed' | 'cancelled' | 'failed'; errorCode?: MessageCode; errorParams?: MessageParams; createdAt: string;
}
/** 拼接内容块里的文本块，段与段之间空一行，与 Go 的 blocksText 一致。 */
export const blocksText = (blocks: AssistantBlock[]) => blocks.filter(block => block.kind === 'text' && block.text?.trim()).map(block => block.text!.trim()).join('\n\n');
/** 一段助手对话：每个会话对应独立的助手时间线。 */
export interface AssistantSession { id: string; title: string; messages: AssistantMessage[]; updatedAt: string }
export interface AssistantSessions { currentId: string; sessions: AssistantSession[] }
export interface AssistantTurn { blocks?: AssistantBlock[]; model?: string; startedAt: string }
export interface AssistantNav { id: string; workId?: string; page?: string }
export interface AssistantUpdate { id: string; workId: string }
export interface AssistantLive { turn?: AssistantTurn | null; queued?: number; nav?: AssistantNav | null; update?: AssistantUpdate | null }
/** assistant.send 附带的界面上下文，与 Go 的 workbench.AssistantContext 一致。 */
export interface AssistantContext {
  page?: string; workId?: string; workTitle?: string; workKind?: string; locale?: UiLocale;
  /** 发送时刻用户在编辑区选中的文字、故事配音里选中的台词与播放位置（秒）。 */
  selection?: string; cue?: AssistantCue; playhead?: number;
}
export interface AssistantCue { index: number; total: number; speaker?: string; text?: string }

/** 标题为空或仍是任一界面语言的“未命名作品”时视为尚未命名，AI 写作可以顺带起名。 */
export const isUntitled = (title: string) => !title.trim() || Object.values(draftCopy).some(copy => copy.untitled === title.trim());

export const createDraft = (example = false, locale: UiLocale = 'zh-CN', modelId = 'index-2.5-q8'): Draft => {
  const copy = draftCopy[seedLocale(locale)];
  return {
  id: crypto.randomUUID().replaceAll('-', ''),
  title: example ? copy.exampleTitle : copy.untitled,
  text: example ? copy.exampleText : '',
  modelId, voiceId: null, mode: 'text', emotionVoiceId: null,
  emotionText: copy.emotionText, inferEmotion: false, emotionStrength: 0.6,
  emotions: [0, 0, 0, 0, 0, 0, 0, 0.5], randomEmotion: false, language: copy.language, speed: 1,
  temperature: 0.8, topP: 0.8, topK: 30, repetitionPenalty: 10, maxTokens: 1500,
  intervalSilenceMs: 200, doSample: true, numBeams: 3, lengthPenalty: 0, seed: null,
};
};
export const emptyState = (): State => ({ characters: [], previews: [], drafts: [createDraft(true)], voices: [], models: [], history: [], preferences: { downloadSource: 'modelscope', backend: 'cpu', modelDirectory: null, uiLocale: 'zh-CN' }, runtimePath: null, runtimeBackend: null, runtimeVersion: null, activity: null, modelActivity: null });
// version 是推荐安装的 audio.cpp 版本，minimum 是当前应用可用的最低版本。
export interface EngineInfo { version: string; minimum: string }
// 服务未报告版本要求（如浏览器预览）时不做限制。
export const versionAtLeast = (value: string | null, minimum: string) => {
  if (!minimum) return true;
  const parse = (v: string | null) => /^v(\d+)\.(\d+)\.(\d+)$/.exec(v ?? '')?.slice(1).map(Number);
  const a = parse(value), b = parse(minimum);
  if (!a || !b) return false;
  const index = a.findIndex((n, i) => n !== b[i]);
  return index < 0 || a[index] > b[index];
};
// 内核需与所选设备一致：低于最低版本时必须更新；不低于最低版本但低于推荐版本时可继续使用，升级可选。
export type RuntimeStatus = 'ready' | 'upgradable' | 'outdated' | 'missing';
export const runtimeStatus = (state: State, engine: EngineInfo): RuntimeStatus =>
  !state.runtimePath || state.runtimeBackend !== state.preferences.backend ? 'missing'
    : !versionAtLeast(state.runtimeVersion, engine.minimum) ? 'outdated'
      : versionAtLeast(state.runtimeVersion, engine.version) ? 'ready' : 'upgradable';
export const runtimeUsable = (status: RuntimeStatus) => status === 'ready' || status === 'upgradable';
// 部分模型（如音乐）需要比应用最低版本更新的内核。
export const modelEngineReady = (state: State, model?: ModelPackage) => !model?.engineMinimum || versionAtLeast(state.runtimeVersion, model.engineMinimum);
export const formatTime = (seconds: number) => `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${Math.floor(seconds % 60).toString().padStart(2, '0')}`;
export const formatSize = (bytes: number) => `${(bytes / 1e9).toFixed(2)} GB`;
export function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  const value = bytes / Math.pow(1024, i);
  if (i === 0) return `${Math.round(value)} B`;
  return `${value.toFixed(2)} ${units[i]}`;
}
export interface Track { id: string; name: string; fileName: string; kind: 'voices' | 'outputs'; subtitle: string; playRequest?: number }
/** 资源页的一行，对应 Go 的 workbench.Asset；bytes 为 -1 表示文件已丢失，users 非空表示正在被引用。 */
export type AssetKind = 'generation' | 'segment' | 'voice' | 'material' | 'image';
export interface Asset { kind: AssetKind; id: string; name: string; media: 'outputs' | 'voices' | 'images'; fileName: string; projectId?: string; projectTitle?: string; projectKind?: ProjectPage; createdAt: string; duration?: number; bytes: number; users?: string[] }
export interface AssetResult { kind: AssetKind; id: string; error?: MessageCode; params?: MessageParams }

export const isKokoroModel = (id: string) => id.startsWith('kokoro-');
// 音乐模型只用于音乐作品，不出现在配音和角色的模型列表里。
export const isMusicModel = (id: string) => id.startsWith('ace-step-') || id.startsWith('yue2-');
/** YuE2 由歌词或旋律谱决定长度，没有时长设置。 */
export const isYuE = (id: string) => id.startsWith('yue2-');
// 设置里模型按用途分页：语音模型、音乐模型（歌曲生成与扒谱）和编曲音色库。
const musicFamilies = new Set(['ace_step', 'yue2', 'sheetsage2']);
export const modelTab = (family: string) => family === 'soundfont' ? 'soundfonts' : musicFamilies.has(family) ? 'music-models' : 'models';
export const modelTabs = ['models', 'music-models', 'soundfonts'];
// 音色库只供编曲作品渲染使用。
export const isSoundFont = (id: string) => id.endsWith('-sf2');
// 扒谱模型只用于从歌曲提取旋律谱。
export const isTranscriber = (id: string) => id.startsWith('sheetsage2-');
export const isVoiceModel = (id: string) => !isMusicModel(id) && !isSoundFont(id) && !isTranscriber(id);
// 同名模型的不同量化归为一组，保持目录顺序。
export function groupModels(models: ModelPackage[]) {
  const groups = new Map<string, ModelPackage[]>();
  for (const model of models) groups.set(model.name, [...(groups.get(model.name) ?? []), model]);
  return [...groups.values()];
}
// 新作品的默认模型：沿用最近一次同类作品用过且已安装的模型，否则取第一个已安装的同类模型，都没有时用 fallback。
export function preferredModel(state: Pick<State, 'models' | 'drafts'>, kind: 'text' | 'story' | 'music' | 'score', fallback: string) {
  const fits = kind === 'music' ? isMusicModel : kind === 'score' ? isSoundFont : isVoiceModel;
  const installed = new Set(state.models.map(model => model.id));
  const recent = state.drafts.filter(d => installed.has(d.modelId) && fits(d.modelId)).sort((a, b) => (b.updatedAt ?? '').localeCompare(a.updatedAt ?? ''))[0];
  return recent?.modelId ?? state.models.find(model => fits(model.id))?.id ?? fallback;
}
export const isVoxModel = (id: string) => id.startsWith('voxcpm2-');
export const isReferenceModel = (id: string) => id.startsWith('omnivoice-') || id.startsWith('qwen3-tts-');
export const isBreezeModel = (id: string) => id.startsWith('breeze-tts-');
export const requiresVoice = (draft: Draft) => isKokoroModel(draft.modelId) || !isVoiceModel(draft.modelId) ? false : draft.modelId.startsWith('qwen3-tts-') ? !/customvoice|voicedesign/.test(draft.modelId) : draft.modelId.startsWith('omnivoice-') || isBreezeModel(draft.modelId) ? draft.voiceMode === 'clone' : !isVoxModel(draft.modelId) || ['clone', 'continuation'].includes(draft.voxMode ?? 'design');

// 旧字幕的身份与 Go 保持一致，编辑后不再依赖数组位置。
export const emptyScore = (): Score => ({ tempo: 100, timeSignature: [4, 4], key: '', sections: [], tracks: [] });

// 打开或修改作品时统一补齐旧数据：台词补上稳定 ID，缺少乐谱的编曲作品补一份空乐谱，避免渲染时崩溃。
export function normalizeDraft(draft: Draft): Draft {
  let next = draft;
  if (next.kind === 'score' && !Array.isArray(next.score?.tracks)) next = { ...next, score: { ...emptyScore(), ...next.score, tracks: [] } };
  if (!next.subtitles || next.subtitles.cues.every(c => c.id)) return next;
  return { ...next, subtitles: { ...next.subtitles, cues: next.subtitles.cues.map((cue, index) => ({ ...cue, id: cue.id ?? `legacy-${next.id}-${index}` })) } };
}
