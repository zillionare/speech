# Live Podcast — Interfaces（接口契约）

> 本文档与 `story.md` / `spec.md` / `architecture.md` 编号对齐：`IF-NNN ↔ SPEC-NNN ↔ ST-LP-NNN`。
> 定义所有外部可见的接口契约：HTTP API、WebSocket 协议、Pydantic 模型、错误码、文件系统布局。
> 实现者必须以本文档为依据，不得自行变更接口签名或语义。

---

## 1. HTTP API 端点

### 1.1 Live Session 生命周期

#### `POST /api/podcasts/{project_id}/live/start`

**对应**：SPEC-004 / ST-LP-004

**请求体** `LiveStartRequest`：

```python
class LiveStartRequest(BaseModel):
    live_only: bool = False
    """True 时跳过所有 tts 段，仅录制 live 段（ST-LP-015）"""

    live_speakers_override: Optional[List[str]] = None
    """本次会话的 live speaker 列表，覆盖全局配置（ST-LP-001 可选）"""
```

**成功响应** `201 Created`，`LiveStartResponse`：

```python
class LiveStartResponse(BaseModel):
    session_id: str
    """新创建的会话 ID（UUID v4）"""

    project_id: str
    """关联的项目 ID"""

    state: str
    """初始状态，通常为 "AI_SPEAKING" 或 "WAITING_TRIGGER" """

    segment_count: int
    """项目总段数"""

    live_segment_count: int
    """source == "live" 的段数"""

    asr_enabled: bool
    """ASR 是否已启用"""

    asr_ready: bool
    """ASR 模型是否已加载完成"""
```

**错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 404 | `PROJECT_NOT_FOUND` | `project_id` 不存在 |
| 409 | `SESSION_ALREADY_ACTIVE` | 该项目已有非终态 session |
| 409 | `NO_LIVE_SEGMENTS` | 项目没有任何 `source == "live"` 段（UW-LP-5） |
| 422 | `VALIDATION_ERROR` | 请求体不合法 |

---

#### `POST /api/podcasts/{project_id}/live/{session_id}/stop`

**对应**：SPEC-004 / ST-LP-004

**请求体**：无（空 body 或 `{}`）。

**成功响应** `200 OK`，`LiveStopResponse`：

```python
class LiveStopResponse(BaseModel):
    session_id: str
    state: str  # "FINISHED"
    recorded_segments: List[int]
    """本次会话录制的段索引列表"""
    merged: bool
    """是否已自动触发合并"""
    merge_warnings: List[str] = []
    """合并过程中的警告信息（如段音频缺失）"""
```

**错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 404 | `SESSION_NOT_FOUND` | `session_id` 不存在 |
| 409 | `SESSION_TERMINATED` | session 已处于 `FINISHED` 或 `ERROR`（UW-LP-1） |

---

#### `POST /api/podcasts/{project_id}/live/{session_id}/resume`

**对应**：SPEC-016 / ST-LP-016

**请求体**：无。

**成功响应** `200 OK`，`LiveResumeResponse`：

```python
class LiveResumeResponse(BaseModel):
    session_id: str
    state: str
    """恢复后的状态（通常为 WAITING_TRIGGER）"""
    cursor: int
    """恢复时的 cursor 位置"""
    captured_segments: List[int]
    """已捕获音频的段索引"""
    asr_ready: bool
```

**错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 404 | `SESSION_NOT_FOUND` | 内存和磁盘均无此 session |
| 409 | `SESSION_ALREADY_ACTIVE` | session 已在内存中运行 |
| 409 | `SESSION_TERMINATED` | session 已 FINISHED/ERROR |

---

#### `POST /api/podcasts/{project_id}/live/{session_id}/redo/{index}`

**对应**：SPEC-013 / ST-LP-013

**请求体**：无。

**成功响应** `200 OK`，`LiveRedoResponse`：

```python
class LiveRedoResponse(BaseModel):
    session_id: str
    index: int
    """回退到的段索引"""
    previous_wav_deleted: bool
    """是否删除了旧的 live_NNNN.wav"""
```

**错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 400 | `NOT_LIVE_SEGMENT` | 目标段 `source != "live"` |
| 404 | `SESSION_NOT_FOUND` | session 不存在 |
| 409 | `SESSION_BUSY` | cursor > index 且正在 RECORDING 后续段 |
| 409 | `SESSION_TERMINATED` | session 已终态 |

---

#### `POST /api/asr/warmup`

**对应**：SPEC-003 / ST-LP-003

**请求体**：无。

**成功响应** `200 OK`，`ASRWarmupResponse`：

```python
class ASRWarmupResponse(BaseModel):
    backend: str
    """当前 ASR 后端（"mlx_whisper" 或 "faster_whisper"）"""
    model: str
    """模型名称"""
    warmup_seconds: float
    """预热耗时"""
    ready: bool
    """预热完成后为 True"""
```

**错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 409 | `ASR_DISABLED` | `config.asr.enabled == false` |
| 409 | `ASR_WARMING` | 预热正在进行中 |

---

### 1.2 现有端点行为变更

#### `POST /api/podcasts/{project_id}/segments/{index}/regenerate`

**变更**（SPEC-002 §2.4）：

当目标段 `source == "live"` 时，**不调用 TTS 引擎**，直接返回：

| 状态码 | 错误码 | 说明 |
|--------|--------|------|
| 409 | `LIVE_SEGMENT_NO_REGENERATE` | live 段只能通过 redo 重录，不走 TTS 批量再生路径（ST-LP-013） |

**响应体**（409 时）：

```python
class LiveRegenerateError(BaseModel):
    error_code: str  # "LIVE_SEGMENT_NO_REGENERATE"
    message: str     # "Live segment cannot be regenerated via TTS. Use POST .../live/{sid}/redo/{index} instead."
    index: int
    source: str      # "live"
```

---

#### `DELETE /api/podcasts/{project_id}/segments/{index}`

**变更**（SPEC-014 / ST-LP-014）：

新增行为：删除段后执行紧凑重编号。

**成功响应** `200 OK`（沿用现有 `PodcastProject` 返回，无 schema 变更）。

**新增错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 409 | `SEGMENT_IN_ACTIVE_SESSION` | 当前有活跃 session 涉及该段 |

---

#### `POST /api/podcasts/{project_id}/merge`

**变更**（SPEC-009 / ST-LP-009）：

**请求/响应 schema 不变**。行为扩展：

- 遍历段时根据 `source` 字段选择读取 `seg_NNNN.wav` 或 `live_NNNN.wav`
- live 段 WAV 缺失时插入 1 s 静音，`status` 标为 `"missing"`
- 合并输出采样率 = `min(engine_sample_rate, device_sample_rate)`
- 真人段合并前通过 ffmpeg 下采样到目标采样率

**响应体**中 `merge_warnings` 字段（已有，扩展含义）：

```json
{
  "merged_audio_filename": "project_merged.wav",
  "merge_warnings": [
    "segment 3: live audio missing, substituted silence"
  ]
}
```

---

#### `GET /api/config` / `POST /api/config`

**变更**（SPEC-001 / ST-LP-001）：

`AppConfigResponse` 和 `AppConfigUpdateRequest` 自动包含新增的 `asr` 和 `live` 子配置（见 §3 模型定义）。无需修改端点代码。

---

### 1.3 ASR 状态查询

#### `GET /api/asr/status`

**新增端点**，用于前端判断 ASR 可用性。

**成功响应** `200 OK`，`ASRStatusResponse`：

```python
class ASRStatusResponse(BaseModel):
    enabled: bool
    backend: str
    model: str
    ready: bool
    """模型已加载并可推理"""
    warming: bool
    """模型正在加载中"""
    degraded: bool
    """连续失败 ≥ 5 次"""
    consecutive_failures: int
    last_error: Optional[str]
```

---

### 1.4 Session 取消

#### `DELETE /api/podcasts/{project_id}/live/{session_id}`

**对应**：SPEC-004 / ST-LP-004

**语义**：放弃当前会话。与 `stop`（定稿合并）不同，cancel 不执行合并，已录制的 `live_NNNN.wav` 保留在磁盘（由 24h 清理任务处理）。

**请求体**：无。

**成功响应** `200 OK`，`LiveCancelResponse`：

```python
class LiveCancelResponse(BaseModel):
    session_id: str
    state: str  # "ABANDONED"
    captured_segments: List[int]
    """已录制但未合并的段索引（WAV 文件保留在磁盘）"""
```

**错误响应**：

| 状态码 | 错误码 | 条件 |
|--------|--------|------|
| 404 | `SESSION_NOT_FOUND` | session_id 不存在 |
| 409 | `SESSION_TERMINATED` | session 已 FINISHED / ABANDONED / ERROR |

---

## 2. WebSocket 协议

### 2.1 端点

```
WS /ws/podcasts/{project_id}/live/{session_id}?role=driver|observe
```

- `role=driver`（默认）：可发送音频帧和控制帧
- `role=observe`：仅接收广播帧，不可发送音频（SPEC-017 / ST-LP-017）

### 2.2 连接建立

**握手完成后，服务端立即推送 `audio_info` 帧**：

```json
{
  "type": "audio_info",
  "sample_rate": 24000,
  "channels": 1,
  "bit_depth": 16
}
```

`sample_rate` 从 `BaseEngine` 实例动态获取（TTS 引擎原生采样率），不写死。

**若客户端为 driver，紧接着推送 `client_audio_info` 请求**：

```json
{
  "type": "request_client_audio_info"
}
```

客户端回复：

```json
{
  "type": "client_audio_info",
  "sample_rate": 48000,
  "channels": 1,
  "bit_depth": 16
}
```

服务端据此配置下采样参数和 WAV 落盘采样率。

### 2.3 帧类型总表

#### 服务端 → 客户端

| type | 数据格式 | 触发时机 | 对应 SPEC |
|------|----------|----------|-----------|
| `audio_info` | JSON | 连接建立 | SPEC-006 |
| `request_client_audio_info` | JSON | driver 连接后 | SPEC-006 |
| `state` | JSON | 每次状态转换 | SPEC-004 |
| `segment_start` | JSON | 新段开始处理 | SPEC-004 |
| `segment_audio_begin` | JSON | 客户端就绪，服务端可推二进制帧 | SPEC-005 |
| `segment_end` | JSON | 段处理完成 | SPEC-004 |
| `asr_partial` | JSON | ASR 增量结果（每 chunk） | SPEC-003 |
| `asr_final` | JSON | 一段 ASR 终结 | SPEC-003 |
| `alignment` | JSON | 文本对齐进度更新 | SPEC-008 |
| `asr_warming` | JSON | ASR 模型加载进度 | SPEC-003 |
| `asr_unavailable` | JSON | ASR 未启用 | SPEC-003 |
| `asr_degraded` | JSON | ASR 连续失败 ≥ 5 | SPEC-003 |
| `end_near` | JSON | 对齐分数 ≥ 0.85 | SPEC-008 |
| `end_triggered` | JSON | end / user_skipped 触发 | SPEC-008 |
| `prefetch_status` | JSON | look-ahead 预取状态 | SPEC-005 |
| `waiting_for_driver` | JSON | live 段等待 driver 连接 | SPEC-004 |
| `session_persisted` | JSON | session.json 写入完成 | SPEC-016 |
| `error` | JSON | 不可恢复错误 | SPEC-004 |
| `heartbeat` | JSON | 每 2 s 发送一次 | SPEC-017 |
| **二进制帧** | bytes | TTS 音频 PCM 数据 | SPEC-005 |

#### 客户端 → 服务端

| type | 数据格式 | 说明 | 对应 SPEC |
|------|----------|------|-----------|
| `client_audio_info` | JSON | 客户端录音设备参数 | SPEC-006 |
| `state_ack` | JSON | 客户端确认状态转换完成 | SPEC-004 |
| `client_log` | JSON | 调试日志（可选） | SPEC-006 |
| `claim_driver` | JSON | observer 请求升级为 driver | SPEC-017 |
| **二进制帧** | bytes | 录音 PCM 数据（设备原生采样率） | SPEC-007 |

### 2.4 JSON 帧 Schema

以下列出每个 JSON 帧的完整字段定义。

#### `state`

```python
class StateFrame(BaseModel):
    type: Literal["state"] = "state"
    state: str
    """IDLE | AI_SPEAKING | RECORDING | WAITING_TRIGGER | FINISHED | ABANDONED | ERROR"""
    previous_state: str
    cursor: int
    total_segments: int
```

#### `segment_start`

```python
class StateAckFrame(BaseModel):
    type: Literal["state_ack"] = "state_ack"
    state: str
    """客户端确认收到的新状态值"""


class SegmentStartFrame(BaseModel):
    type: Literal["segment_start"] = "segment_start"
    index: int
    source: Literal["tts", "live"]
    speaker: str
    text: str
```

#### `segment_end`

```python
class SegmentAudioBeginFrame(BaseModel):
    type: Literal["segment_audio_begin"] = "segment_audio_begin"
    index: int
    """即将推送音频的段索引。客户端收到此帧后开始接收二进制帧。"""


class SegmentEndFrame(BaseModel):
    type: Literal["segment_end"] = "segment_end"
    index: int
    source: Literal["tts", "live"]
    duration_seconds: float
    """该段实际时长（TTS 段从 engine 获取，live 段从录制时长计算）"""
```

#### `asr_partial`

```python
class ASRPartialFrame(BaseModel):
    type: Literal["asr_partial"] = "asr_partial"
    text: str
    audio_ms: int
    """该 chunk 对应的音频时长（毫秒）"""
```

#### `asr_final`

```python
class ASRFinalFrame(BaseModel):
    type: Literal["asr_final"] = "asr_final"
    text: str
    matched_ratio: float
    """LCS 对齐分数 = len(lcs) / len(target_text)"""
    confidence: float
    """ASR 置信度 0..1"""
```

#### `alignment`

```python
class AlignmentFrame(BaseModel):
    type: Literal["alignment"] = "alignment"
    matched_chars: int
    total_chars: int
    ratio: float
    """matched_chars / total_chars"""
```

#### `asr_warming`

```python
class ASRWarmingFrame(BaseModel):
    type: Literal["asr_warming"] = "asr_warming"
    progress: float
    """0.0 ~ 1.0，1.0 表示加载完成"""
```

#### `asr_unavailable`

```python
class ASRUnavailableFrame(BaseModel):
    type: Literal["asr_unavailable"] = "asr_unavailable"
    reason: str
    """如 "asr disabled in config" 或 "model not installed" """
```

#### `asr_degraded`

```python
class ASRDegradedFrame(BaseModel):
    type: Literal["asr_degraded"] = "asr_degraded"
    consecutive_failures: int
    last_error: str
```

#### `end_near`

```python
class EndNearFrame(BaseModel):
    type: Literal["end_near"] = "end_near"
    index: int
    alignment_score: float
    """触发时的对齐分数，≥ 0.85"""
```

#### `end_triggered`

```python
class EndTriggeredFrame(BaseModel):
    type: Literal["end_triggered"] = "end_triggered"
    index: int
    trigger: Literal["end", "user_skipped", "force_end"]
    alignment_score: float
    silence_ms: int
```

#### `prefetch_status`

```python
class PrefetchStatusFrame(BaseModel):
    type: Literal["prefetch_status"] = "prefetch_status"
    index: int
    """预取的段索引"""
    status: Literal["started", "ready", "error"]
```

#### `waiting_for_driver`

```python
class WaitingForDriverFrame(BaseModel):
    type: Literal["waiting_for_driver"] = "waiting_for_driver"
    index: int
    """等待 driver 的 live 段索引"""
```

#### `session_persisted`

```python
class SessionPersistedFrame(BaseModel):
    type: Literal["session_persisted"] = "session_persisted"
    cursor: int
    last_save_at: float
    """Unix timestamp"""
```

#### `error`

```python
class ErrorFrame(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    recoverable: bool = False
```

#### `heartbeat`

```python
class HeartbeatFrame(BaseModel):
    type: Literal["heartbeat"] = "heartbeat"
    timestamp: float
```

#### `client_audio_info`（客户端 → 服务端）

```python
class ClientAudioInfoFrame(BaseModel):
    type: Literal["client_audio_info"] = "client_audio_info"
    sample_rate: int
    """设备原生采样率，推荐 48000"""
    channels: int = 1
    bit_depth: int = 16
```

#### `client_log`（客户端 → 服务端）

```python
class ClientLogFrame(BaseModel):
    type: Literal["client_log"] = "client_log"
    level: Literal["debug", "info", "warn", "error"] = "info"
    msg: str
```

#### `claim_driver`（客户端 → 服务端）

```python
class ClaimDriverFrame(BaseModel):
    type: Literal["claim_driver"] = "claim_driver"
```

### 2.5 二进制帧格式

#### 服务端 → 客户端（TTS 音频）

| 属性 | 值 |
|------|-----|
| 格式 | 裸 PCM（无 WAV header） |
| 采样率 | TTS 引擎原生采样率（由 `audio_info` 帧协商） |
| 声道 | 1（mono） |
| 位深 | 16-bit |
| 字节序 | little-endian |
| 切片时长 | ~200 ms |

#### 客户端 → 服务端（录音音频）

| 属性 | 值 |
|------|-----|
| 格式 | 裸 PCM（无 WAV header） |
| 采样率 | 设备原生采样率（由 `client_audio_info` 帧告知，推荐 48 kHz） |
| 声道 | 1（mono） |
| 位深 | 16-bit |
| 字节序 | little-endian |
| 切片时长 | ~1 s（`ScriptProcessor` buffer size 4096 samples） |

### 2.6 WebSocket 关闭码

| 关闭码 | 含义 | 触发条件 |
|--------|------|----------|
| 1000 | 正常关闭 | 用户主动 stop |
| 1001 | 服务端离开 | uvicorn shutdown |
| 1003 | 角色违规 | observer 发送 binary 帧 |
| 1008 | 策略违规 | driver 已存在时第二个客户端尝试 driver；session 已终态 |
| 1011 | 服务端异常 | 不可恢复的内部错误 |
| 1012 | 服务重启 | 服务端即将重启（客户端应尝试重连） |

### 2.7 重连策略

- 客户端断线后服务端暂停 30 s（`WAITING_TRIGGER` → 建议 spec 新增 `PAUSED` 状态以区分断线暂停与正常检测），超时自动 stop
- 客户端重连参数：`reconnect_attempts=5, reconnect_delay_ms=1000, backoff=2.0`
- 重连时携带相同 `session_id`，服务端恢复 driver 绑定

---

## 3. Pydantic 模型定义

### 3.1 新增配置模型

```python
class ASRConfig(BaseModel):
    """ASR 引擎配置（SPEC-003）"""
    enabled: bool = False
    backend: Literal["mlx_whisper", "faster_whisper"] = "mlx_whisper"
    model: str = "mlx-community/whisper-medium-mlx-4bit"
    language: str = "zh"
    chunk_seconds: float = Field(default=1.0, ge=0.1, le=10.0)
    beam_size: int = Field(default=1, ge=1, le=5)
    vad_filter: bool = True
    compute_type: str = "float16"
    device: str = "auto"
    warmup_on_start: bool = False

class LiveConfig(BaseModel):
    """Live Podcast 运行时配置（SPEC-018 等）"""
    max_ai_rephrases_per_segment: int = Field(default=2, ge=0, le=5)
    silence_db_threshold: float = Field(default=-45.0)
    end_near_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    end_alignment_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
    end_silence_ms: int = Field(default=300, ge=100, le=5000)
    force_end_silence_ms: int = Field(default=4000, ge=1000, le=30000)
    debounce_ms: int = Field(default=200, ge=50, le=1000)
    driver_disconnect_timeout_s: int = Field(default=30, ge=5, le=120)
    session_persist_interval_s: int = Field(default=5, ge=1, le=30)
    max_sessions: int = Field(default=5, ge=1, le=20)
    max_observers_per_session: int = Field(default=10, ge=1, le=50)
    audio_buffer_max_mb: int = Field(default=50, ge=10, le=200)
    tts_max_seconds: int = Field(default=60, ge=10, le=300)
    """单段 TTS 文本超过此时长（按字符估算）时自动切分子段（SPEC-005）"""
    tts_timeout_seconds: int = Field(default=30, ge=5, le=120)
    """单段 TTS 生成超时。超时则跳过该段、用静音占位"""
```

### 3.2 现有模型扩展

#### `VoicesConfig`（追加字段）

```python
class VoicesConfig(BaseModel):
    base_dir: str
    default_voice: str
    cache_subdir: str = ".cache"
    bundled_chinese_voices: List[str] = []
    live_speakers: List[str] = []
    """声明为真人演绎的 speaker 名列表（SPEC-001）"""
```

#### `PodcastSegment`（追加字段）

```python
class PodcastSegment(BaseModel):
    # ... 现有字段保持不变 ...
    source: Literal["tts", "live"] = "tts"
    """段的音频来源（SPEC-002）。默认 tts，向后兼容。"""
```

`status` 字段 Literal 扩展：

```python
    status: Literal["pending", "generated", "error", "missing"] = "pending"
    """新增 "missing"：live 段 WAV 文件缺失（SPEC-009）"""
```

#### `PodcastProject`（追加字段）

```python
class PodcastProject(BaseModel):
    # ... 现有字段保持不变 ...
    live_speakers_override: Optional[List[str]] = None
    """项目级 live speaker 覆盖，None 时使用全局 config.voices.live_speakers（SPEC-001 可选）"""
```

#### `AppConfig`（追加字段）

```python
class AppConfig(BaseModel):
    # ... 现有字段保持不变 ...
    asr: ASRConfig = ASRConfig()
    live: LiveConfig = LiveConfig()
```

#### `AppConfigResponse`（追加字段）

```python
class AppConfigResponse(BaseModel):
    # ... 现有字段保持不变 ...
    asr: ASRConfig
    live: LiveConfig
```

#### `AppConfigUpdateRequest`（追加字段）

```python
class AppConfigUpdateRequest(BaseModel):
    # ... 现有字段保持不变 ...
    asr: Optional[ASRConfig] = None
    live: Optional[LiveConfig] = None
```

### 3.3 会话持久化模型

```python
class SessionSnapshot(BaseModel):
    """每 5 s 写入 live_sessions/{session_id}.json（SPEC-016）"""
    session_id: str
    project_id: str
    cursor: int
    state: str
    captured_segments: List[int]
    """已捕获音频的段索引"""
    started_at: float
    last_save_at: float
    live_only: bool = False
    device_sample_rate: Optional[int] = None
    """客户端录音设备采样率（从 client_audio_info 帧获取）"""
```

---

## 4. 错误码体系

### 4.1 HTTP 错误响应格式

所有错误响应遵循统一格式：

```python
class ErrorResponse(BaseModel):
    error_code: str
    """机器可读错误码，全大写 SNAKE_CASE"""
    message: str
    """人类可读错误描述"""
    details: Optional[Dict[str, Any]] = None
    """可选的附加信息"""
```

### 4.2 错误码清单

| 错误码 | HTTP 状态码 | 说明 | 对应 SPEC |
|--------|-------------|------|-----------|
| `PROJECT_NOT_FOUND` | 404 | 项目 ID 不存在 | — |
| `SESSION_NOT_FOUND` | 404 | 会话 ID 不存在 | SPEC-004 |
| `SESSION_ALREADY_ACTIVE` | 409 | 项目已有非终态会话 | SPEC-004 |
| `SESSION_TERMINATED` | 409 | 会话已 FINISHED 或 ERROR | SPEC-004 |
| `SESSION_BUSY` | 409 | 会话正在录制后续段，无法 redo | SPEC-013 |
| `NO_LIVE_SEGMENTS` | 409 | 项目无 live 段，不能启动 live session | SPEC-004 / UW-LP-5 |
| `NOT_LIVE_SEGMENT` | 400 | 目标段不是 live 类型 | SPEC-013 |
| `LIVE_SEGMENT_NO_REGENERATE` | 409 | live 段不能走 TTS regenerate | SPEC-002 |
| `SEGMENT_IN_ACTIVE_SESSION` | 409 | 段被活跃会话占用 | SPEC-014 |
| `ASR_DISABLED` | 409 | ASR 配置未启用 | SPEC-003 |
| `ASR_WARMING` | 409 | ASR 预热进行中 | SPEC-003 |
| `SESSION_ABANDONED` | 409 | 会话已被取消（cancel 后尝试 stop/resume） | SPEC-004 |
| `VALIDATION_ERROR` | 422 | 请求体不合法 | — |

### 4.3 WebSocket 错误帧的 `code` 字段

| code | 含义 | recoverable |
|------|------|-------------|
| `ASR_DOWN` | ASR 推理异常 | true |
| `ASR_DEGRADED` | ASR 连续失败 ≥ 5 | true |
| `TTS_TIMEOUT` | TTS 引擎超时 | true |
| `TTS_ERROR` | TTS 生成失败 | true |
| `TTS_SEGMENT_TOO_LONG` | TTS 段文本超长，已自动切分或跳过 | true |
| `WAV_WRITE_FAILED` | WAV 落盘失败 | true |
| `BUFFER_OVERFLOW` | audio_buffer 超内存限制 | false |
| `ILLEGAL_STATE_TRANSITION` | 非法状态转换 | false |
| `SESSION_EXPIRED` | 会话超时自动终止 | false |

---

## 5. 文件系统契约

### 5.1 目录布局

```
outputs/podcasts/{project_id}/
├── {project_id}.json                     # PodcastProject 序列化
├── seg_{index:04d}.wav                   # TTS 段音频
├── live_{index:04d}.wav                  # 真人段音频（新增）
├── {project_id}_merged.wav               # 合并输出
└── live_sessions/                        # 会话状态快照（新增）
    └── {session_id}.json                 # SessionSnapshot
```

### 5.2 文件命名规则

| 文件类型 | 命名模式 | 采样率 | 位深 | 声道 |
|----------|----------|--------|------|------|
| TTS 段 | `seg_{index:04d}.wav` | 引擎原生 | 16-bit | mono |
| 真人段 | `live_{index:04d}.wav` | 设备原生（推荐 48 kHz） | 16-bit | mono |
| 合并输出 | `{project_id}_merged.wav` | min(引擎, 设备) | 16-bit | mono |
| 会话快照 | `{session_id}.json` | — | — | — |

### 5.3 段编号紧凑（SPEC-014）

`DELETE /api/podcasts/{pid}/segments/{i}` 后执行 `compact_segment_files(project_id)`：

1. 从 `project.segments` 弹出第 i 项
2. 删除对应 `seg_{i:04d}.wav` 或 `live_{i:04d}.wav`
3. 后续所有段文件重编号：`seg_{n:04d}.wav` → `seg_{n-1:04d}.wav`（live 同理）
4. 更新每段的 `audio_filename` 字段
5. 重写 `{project_id}.json`

### 5.4 生命周期清理

| 文件类型 | 过期条件 | 清理动作 |
|----------|----------|----------|
| `live_sessions/*.json` (FINISHED) | finished > 1 h | 删除 JSON |
| `live_sessions/*.json` (orphan) | 无对应内存 session 且 > 24 h | 删除 JSON + 关联 live wav |
| `live_NNNN.wav` | 所属 session 过期清理时 | 随 session 一起删除 |
| `seg_NNNN.wav` | 不自动删除 | 需通过项目级 DELETE 端点 |

---

## 6. 配置 Schema（config.yaml）

### 6.1 新增/扩展字段

```yaml
voices:
  base_dir: "voices"
  default_voice: "default"
  live_speakers:          # 新增（SPEC-001）
    - "Aaron"

asr:                       # 新增（SPEC-003）
  enabled: false
  backend: "mlx_whisper"   # "mlx_whisper" | "faster_whisper"
  model: "mlx-community/whisper-medium-mlx-4bit"
  language: "zh"
  chunk_seconds: 1.0
  beam_size: 1
  vad_filter: true
  compute_type: "float16"
  device: "auto"
  warmup_on_start: false

live:                      # 新增（SPEC-018 等）
  max_ai_rephrases_per_segment: 2
  silence_db_threshold: -45.0
  end_near_threshold: 0.85
  end_alignment_threshold: 0.98
  end_silence_ms: 300
  force_end_silence_ms: 4000
  debounce_ms: 200
  driver_disconnect_timeout_s: 30
  session_persist_interval_s: 5
  max_sessions: 5
  max_observers_per_session: 10
  audio_buffer_max_mb: 50
  tts_max_seconds: 60
  tts_timeout_seconds: 30
```

### 6.2 向后兼容

- `live_speakers` 默认 `[]`（空列表），不影响现有项目
- `asr` 默认 `enabled: false`，不影响现有项目
- `live` 所有字段有默认值，不影响现有项目
- `save_config_to_yaml()` 通过 Pydantic v2 序列化所有字段，不丢失未知字段

---

## 7. 内部模块接口（live 子包）

以下为 `tts_service/live/` 子包内各模块的公开接口。实现者不得在模块外直接访问私有属性。

### 7.1 EmbeddedASR（asr_engine.py）

```python
class EmbeddedASR:
    def __init__(self, cfg: ASRConfig) -> None: ...

    def warmup(self) -> None:
        """用 1 s 静音样本预热模型。阻塞调用。"""

    async def transcribe_chunk(self, pcm_bytes: bytes) -> ASRResult:
        """推理一个 chunk 的 PCM 数据。非阻塞（内部 run_in_executor）。"""

    @property
    def is_ready(self) -> bool:
        """模型已加载并可推理。"""

    @property
    def is_warming(self) -> bool:
        """模型正在加载中。"""

    @property
    def degraded(self) -> bool:
        """连续失败 ≥ 5 次。"""
```

```python
@dataclass(frozen=True)
class ASRResult:
    text: str
    is_final: bool
    audio_ms: int
    confidence: float       # 0..1
    raw_segments: list       # word-level 时间戳
```

### 7.2 LiveSession（session.py）

```python
class LiveSession:
    session_id: str
    project_id: str
    cursor: int
    state: LiveState
    segments: List[PodcastSegment]
    audio_buffer: Dict[int, bytearray]
    alignment_score: float
    last_asr_text: str
    started_at: float
    finished_at: Optional[float]
    ws_clients: Set[WebSocket]
    driver_ws: Optional[WebSocket]
    errors: List[str]

    def transition(self, new_state: LiveState) -> None:
        """执行状态转换。非法转换抛 IllegalStateTransition。"""

    async def start(self) -> None:
        """启动会话，调度首段。"""

    async def stop(self) -> LiveStopResponse:
        """优雅终止，落盘所有缓冲，可选触发 merge。"""

    async def resume(self) -> None:
        """从持久化文件恢复。"""

    async def redo(self, index: int) -> LiveRedoResponse:
        """回退 cursor 到指定段，删除旧 WAV。"""

    async def broadcast(self, frame: BaseModel) -> None:
        """向所有连接的 WS 客户端推送 JSON 帧。"""

    async def push_audio(self, pcm_bytes: bytes) -> None:
        """向 driver 推送二进制音频帧。"""

    def persist_snapshot(self) -> None:
        """写入 session.json。"""
```

### 7.3 LiveSessionRegistry（session.py）

```python
class LiveSessionRegistry:
    def __init__(self, config: AppConfig, engine: BaseEngine,
                 sample_manager: SampleManager,
                 podcast_manager: PodcastManager) -> None: ...

    async def create(self, project_id: str,
                     request: LiveStartRequest) -> LiveSession:
        """创建新会话。超 max_sessions 时抛异常。"""

    def get(self, project_id: str, session_id: str) -> Optional[LiveSession]:
        """获取会话，不存在返回 None。"""

    async def stop(self, project_id: str, session_id: str) -> LiveStopResponse:
        """停止会话。"""

    async def resume(self, project_id: str, session_id: str) -> LiveSession:
        """从磁盘恢复。"""

    def scan_resumable(self, outputs_dir: Path) -> List[SessionSnapshot]:
        """启动时扫描孤儿 session 文件。"""

    def get_active_for_project(self, project_id: str) -> Optional[LiveSession]:
        """获取项目的活跃会话（如有）。"""

    async def cleanup(self) -> None:
        """清理过期会话和文件。"""
```

### 7.4 StreamingTTSProxy（streaming_engine.py）

```python
class StreamingTTSProxy:
    """前置条件：BaseEngine 必须实现 sample_rate 属性（返回引擎原生采样率）。
    各引擎实现值：LocalVibeVoiceEngine -> 24000, QwenRemoteEngine -> 从 WAV 头解析。
    """

    def __init__(self, engine: BaseEngine) -> None: ...

    async def stream_segment(
        self,
        text: str,
        voice: str,
        chunk_ms: int = 200,
    ) -> AsyncIterator[bytes]:
        """生成一段 TTS 音频并以 PCM chunk 流式 yield。"""

    @property
    def engine_sample_rate(self) -> Optional[int]:
        """引擎原生采样率，首次生成后可用。"""
```

### 7.5 EndDetector（end_detector.py）

```python
class Trigger(str, Enum):
    END_NEAR = "end_near"
    END = "end"
    USER_SKIPPED = "user_skipped"

class EndDetector:
    def __init__(
        self,
        target_text: str,
        silence_db_threshold: float = -45.0,
        end_near_threshold: float = 0.85,
        end_alignment_threshold: float = 0.98,
        end_silence_ms: int = 300,
        force_end_silence_ms: int = 4000,
        debounce_ms: int = 200,
    ) -> None: ...

    def update_vad(self, dbfs: float, frame_ms: int = 20) -> Optional[Trigger]:
        """更新 VAD 状态。可能返回 end（仅静音触发）。"""

    def update_asr(self, asr_text: str) -> Optional[Trigger]:
        """更新 ASR 文本对齐。可能返回 end_near / end / user_skipped。"""

    @property
    def alignment_score(self) -> float:
        """当前对齐分数。"""

    @property
    def silence_ms(self) -> int:
        """当前累计静音毫秒。"""
```

### 7.6 WavWriter（wav_writer.py）

```python
class WavWriter:
    def __init__(self, path: Path, sample_rate: int, channels: int = 1) -> None:
        """打开 WAV 文件准备流式写入。"""

    def append(self, pcm_bytes: bytes) -> None:
        """追加 PCM 数据到文件。"""

    def close(self) -> None:
        """关闭文件并执行 peak_normalize。"""

    @property
    def bytes_written(self) -> int:
        """已写入的 PCM 字节数。"""

    @property
    def duration_seconds(self) -> float:
        """已写入的音频时长。"""
```

### 7.7 AudioResampler（audio_resampler.py）

```python
class AudioResampler:
    @staticmethod
    def downsample(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
        """将 PCM 字节从 src_rate 下采样到 dst_rate。
        使用 scipy.signal.resample_poly。
        src_rate == dst_rate 时直接返回原数据。
        """

    @staticmethod
    def upsample(pcm_bytes: bytes, src_rate: int, dst_rate: int) -> bytes:
        """上采样。仅用于合并时理论完备性，正常路径不调用（不上采样原则）。"""
```

### 7.8 Frames（frames.py）

所有 §2.4 中定义的 Pydantic 帧模型集中在此文件，供 WS handler 和 session 使用。

---

## 8. 与现有 API 的兼容性

### 8.1 不变端点

以下端点行为完全不变，不因 Live Podcast 功能产生任何变化：

- `GET /health`
- `GET /api/voices`、`POST /api/voices`、`PUT /api/voices/{speaker}/transcript`
- `POST /api/voices/{speaker}/cache`、`DELETE /api/voices/{speaker}`
- `GET /api/voices/{speaker}/audio`
- `GET /api/generations`、`POST /api/generate`、`POST /api/generate/stream`
- `GET /api/outputs/{filename}`、`POST /api/outputs/prune`
- `GET /v1/voices`、`POST /v1/audio/speech`、`POST /v1/audio/podcast`
- `GET /api/tone-voices/{speaker}`
- `POST /api/podcasts`、`GET /api/podcasts`、`GET /api/podcasts/{id}`、`DELETE /api/podcasts/{id}`
- `PUT /api/podcasts/{id}/gap`、`PUT /api/podcasts/{id}/segments/{i}`
- `POST /api/podcasts/{id}/segments/{i}/insert`
- `POST /api/podcasts/{id}/generate-all`
- `GET /api/bgm`、`POST /api/bgm`、`DELETE /api/bgm/{filename}`、`GET /api/bgm/{filename}`
- `GET /api/podcasts/{id}/audio/{filename}`

### 8.2 行为扩展端点

以下端点 schema 不变，但行为有条件扩展（见 §1.2）：

- `POST /api/podcasts/{id}/segments/{i}/regenerate` — live 段返回 409
- `DELETE /api/podcasts/{id}/segments/{i}` — 新增紧凑重编号
- `POST /api/podcasts/{id}/merge` — 多源合并 + 采样率统一
- `GET /api/config` / `POST /api/config` — 响应体新增 asr/live 字段

### 8.3 新增端点汇总

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/podcasts/{pid}/live/start` | 创建并启动 live session |
| POST | `/api/podcasts/{pid}/live/{sid}/stop` | 停止 live session |
| POST | `/api/podcasts/{pid}/live/{sid}/resume` | 恢复崩溃的 session |
| POST | `/api/podcasts/{pid}/live/{sid}/redo/{index}` | 重录指定 live 段 |
| POST | `/api/asr/warmup` | 手动预热 ASR 模型 |
| GET | `/api/asr/status` | 查询 ASR 状态 |
| DELETE | `/api/podcasts/{pid}/live/{sid}` | 取消（放弃）live session |
| WS | `/ws/podcasts/{pid}/live/{sid}` | 双向 WebSocket 通道 |
