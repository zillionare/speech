// 在线演播（Live Studio）的 REST 接口封装与共享类型。
// 请求与页面同源，直接用 fetch；WebSocket 协议与麦克风采集见 live-driver.ts。

export type LiveSegmentSource = 'tts' | 'live';
export type LiveSegmentStatus = 'pending' | 'generating' | 'generated' | 'recorded' | 'missing' | 'error';

export interface LiveSegment {
  index: number;
  text: string;
  speaker: string;
  resolved_voice: string;
  source: LiveSegmentSource;
  status: LiveSegmentStatus;
  audio_url?: string | null;
}

export interface LiveSession {
  session_id: string;
  title: string;
  created_at: string;
  segment_count: number;
  has_final: boolean;
  first_text: string;
}

// ASR 状态：/api/live/start 返回的 asr 字段与 ws 的 asr_progress / asr_unavailable 同构。
export interface LiveAsrStatus {
  status?: string;
  progress?: number;
  message?: string;
  error?: string;
  enabled?: boolean;
  ready?: boolean;
}

export interface LiveSessionDetail {
  session_id: string;
  title: string;
  created_at?: string;
  state?: string;
  segments: LiveSegment[];
  has_final?: boolean;
  final_audio_url?: string | null;
}

export interface LiveStartResult {
  session_id: string;
  title: string;
  segments: LiveSegment[];
  live_speakers: string[];
  asr_required: boolean;
  asr: LiveAsrStatus;
}

export interface LiveExportResult {
  session_id: string;
  filename: string;
  audio_url: string;
}

// 后端出错时返回 {detail: "..."}；其余情况按 HTTP 状态码报错。
async function request<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  const body: unknown = await response.json().catch(() => undefined);
  if (!response.ok) {
    const detail = body && typeof body === 'object' && 'detail' in body ? String((body as { detail: unknown }).detail) : '';
    throw new Error(detail || `HTTP ${response.status}`);
  }
  return body as T;
}

export function startLiveSession(text: string, title: string): Promise<LiveStartResult> {
  return request<LiveStartResult>('/api/live/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, title }),
  });
}

export function listLiveSessions(): Promise<LiveSession[]> {
  return request<LiveSession[]>('/api/live/sessions');
}

export function getLiveSession(sessionId: string): Promise<LiveSessionDetail> {
  return request<LiveSessionDetail>(`/api/live/${sessionId}`);
}

export function prefetchLiveTts(sessionId: string): Promise<unknown> {
  return request(`/api/live/${sessionId}/prefetch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({}),
  });
}

export function stopLiveSession(sessionId: string): Promise<unknown> {
  return request(`/api/live/${sessionId}/stop`, { method: 'POST' });
}

export function deleteLiveSession(sessionId: string): Promise<unknown> {
  return request(`/api/live/${sessionId}`, { method: 'DELETE' });
}

export function exportLiveSession(sessionId: string): Promise<LiveExportResult> {
  return request<LiveExportResult>(`/api/live/${sessionId}/export`, { method: 'POST' });
}

export function updateLiveSegment(sessionId: string, index: number, text: string): Promise<unknown> {
  return request(`/api/live/${sessionId}/segment/${index}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
}
