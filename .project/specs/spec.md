# Live Podcast — Spec（技术规范）

> 与 `story.md` 中的编号一一对应：`SPEC-NNN ↔ ST-LP-NNN`。
> 复用原则：所有改动尽量沿用现有 `PodcastManager` / `BaseEngine` / 静态 UI 形态。
> 命名空间：`tts_service/live/` 为新增子包；现有文件尽量通过追加而非重写扩展。
>
> **关键架构决策：**
> - **ASR 完全内嵌到 Python 服务进程**（`tts_service/live/asr_engine.py`）。
>   不再走外部 HTTP API。理由：端到端检测链路（录音→ASR→对齐→切段）任何跨进程跳转都会引入
>   50-150ms 抖动，破坏"贴"的听感。
> - **音频质量分层原则**：录音端和 TTS 生成端各自保持设备/引擎的最高采样率，不作降质；
>   ASR 路径独立下采样到 16 kHz；最终合并时，统一到两者中较低采样率（下采样，不上采样，
>   因为上采样不产生新信息量）。

---

## SPEC-001 Live Speaker 配置扩展

### 1.1 YAML Schema
在 `tts_service/config.py` 的 `VoicesConfig` 中追加：

```python
class VoicesConfig(BaseModel):
    base_dir: str
    default_voice: str
    cache_subdir: str = ".cache"
    bundled_chinese_voices: List[str] = []
    live_speakers: List[str] = []          # 新增：声明为真人演绎的 speaker 名
```

向后兼容：未列出的项目读出为 `[]`。

### 1.2 运行时更新
- `/api/config` 端点（`server.py:179-249`）已支持 Pydantic 模型更新；在 `VoicesConfig` 增字段后会自动允许通过 `AppConfigUpdateRequest` 写入。
- `save_config_to_yaml()`（`config.py:153-164`）无需改动 —— Pydantic v2 默认序列化所有字段。

### 1.3 Per-project 覆盖（可选）
- 在 `tts_service/models.py:131` 的 `PodcastProject` 增加 `live_speakers_override: List[str] | None = None`。
- 解析顺序：`project.live_speakers_override ?? config.voices.live_speakers`。

### 1.4 LiveConfig 定义
在 `tts_service/config.py` 新增：

```python
class LiveConfig(BaseModel):
    """Live Podcast 运行时配置"""
    max_ai_rephrases_per_segment: int = 2
    silence_db_threshold: float = -45.0
    end_near_threshold: float = 0.85
    end_alignment_threshold: float = 0.98
    end_silence_ms: int = 300
    force_end_silence_ms: int = 4000
    debounce_ms: int = 200
    driver_disconnect_timeout_s: int = 30
    session_persist_interval_s: int = 5
    max_sessions: int = 5
    max_observers_per_session: int = 10
    audio_buffer_max_mb: int = 50
    tts_max_seconds: int = 60
    tts_timeout_seconds: int = 30
```

在 `AppConfig` 中挂载：

```python
class AppConfig(BaseModel):
    ...
    live: LiveConfig = LiveConfig()
```

向后兼容：所有字段有默认值，不影响现有项目。

---

## SPEC-002 PodcastSegment.source 字段

### 2.1 Pydantic 模型
在 `tts_service/models.py:131` 的 `PodcastSegment` 中追加：

```python
class SegmentSource(str, Enum):
    TTS = "tts"
    LIVE = "live"

class PodcastSegment(BaseModel):
    text: str
    speaker: str
    tone: Optional[str] = None
    audio_filename: Optional[str] = None
    status: str = "pending"
    pre_pause: float = 0.0
    post_pause: float = 0.0
    bgm_filename: Optional[str] = None
    source: SegmentSource = SegmentSource.TTS   # 新增，默认 TTS
```

### 2.2 旧项目兼容
`podcast_manager.load_project()`（`podcast_manager.py:419`）在读 JSON 后做：

```python
for seg in data.get("segments", []):
    seg.setdefault("source", "tts")
```

### 2.3 自动标记
`PodcastManager._text_to_segments()`（`podcast_manager.py:435-456`）创建每个段时：

```python
src = SegmentSource.LIVE if seg_data["speaker"] in live_speakers_set else SegmentSource.TTS
seg_data["source"] = src
```

`live_speakers_set` 来自 `self.config.voices.live_speakers` 联合 `project.live_speakers_override`。

### 2.4 解析端在 `regenerate_segment` 处的行为
当 `seg.source == "live"`，`regenerate_segment()` 必须立即返回、不调用 `engine.generate_single()`，不创建 `seg_NNNN.wav`，仅保留音频待真人录制。

---

## SPEC-003 ASR 完全内嵌

### 3.1 模块位置
新增 `tts_service/live/asr_engine.py`，提供类：

```python
class EmbeddedASR:
    """进程内 ASR 推理器，不走外部网络"""

    def __init__(self, cfg: ASRConfig):
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    def warmup(self) -> None: ...
    async def transcribe_chunk(self, pcm_bytes: bytes) -> ASRResult: ...

    @property
    def is_ready(self) -> bool: ...
    @property
    def is_warming(self) -> bool: ...
```

### 3.2 数据结构
```python
@dataclass
class ASRConfig:
    enabled: bool = False
    backend: str = "mlx_whisper"     # "mlx_whisper" | "faster_whisper"
    model: str = "mlx-community/whisper-small"
    language: str = "zh"
    chunk_seconds: float = 1.0
    beam_size: int = 1
    vad_filter: bool = True
    compute_type: str = "float16"     # 仅 faster_whisper 用
    device: str = "auto"
    warmup_on_start: bool = False

@dataclass
class ASRResult:
    text: str
    is_final: bool
    audio_ms: int
    confidence: float            # 0..1，标记低置信
    raw_segments: list           # 含 word-level 时间戳，便于 alignment
```

### 3.3 后端实现策略

**`backend == "mlx_whisper"`（Apple Silicon 首选）：**
```python
import mlx_whisper
result = mlx_whisper.transcribe(
    audio_np,             # 16kHz mono float32 numpy
    path_or_hsp=self.cfg.model,
    language=self.cfg.language,
    word_timestamps=True,
)
```
- 模型由 mlx-whisper 在首次调用时懒加载（无需手动 `load_models`）
- Metal/MLX 编译缓存首次推理后保留
- 在 Apple Silicon 上 small 模型 P95 ≈ 200-300 ms / 1 s chunk

**`backend == "faster_whisper"`（跨平台）：**
```python
from faster_whisper import WhisperModel
self._model = WhisperModel(
    self.cfg.model,
    device=self.cfg.device,
    compute_type=self.cfg.compute_type,
)
segments, info = self._model.transcribe(
    audio_np,
    language=self.cfg.language,
    beam_size=self.cfg.beam_size,
    vad_filter=self.cfg.vad_filter,
    word_timestamps=True,
)
```

### 3.4 模型预热
- `warmup()` 跑一次 1 秒静音样本的完整推理
- 预热可以延后到首次 `transcribe_chunk` 调用前自动执行
- 服务启动时不阻塞（默认 `warmup_on_start = false`），避免 uvicorn 启动阻塞
- 提供 `/api/asr/warmup` 手动触发端点

### 3.5 线程模型
- **`EmbeddedASR` 自身是线程安全的**：内部用 `_lock` 保护模型引用，但对 `mlx-whisper` 的实际推理仍串行（同一 Metal stream 不并发）
- **MVP 阶段**：LiveSession 的 ASR 调用串行排队 + `asyncio.run_in_executor` 把同步推理放到线程池，避免阻塞事件循环
- **未来扩展**：若两个会话同时录制且后端是 `faster_whisper`，可以用两套模型实例；不过通常一个 session 只对应一个真人驱动者

### 3.6 失败处理
- 捕获 `RuntimeError` 等异常 → 返回 `ASRResult(text="", confidence=0.0)`
- 连续 5 次失败 → 标记 `degraded=True`，触发 WS 事件 `asr_degraded`，前端可提示用户检查

### 3.7 依赖（追加到 `requirements.txt`）
```
mlx-whisper>=0.4.0           # Apple Silicon 路径（可选）
# faster-whisper>=1.0.0      # 跨平台路径（视部署选其一或并存）
```
由用户根据硬件二选一；两套接口兼容。

### 3.8 配置挂载
在 `tts_service/config.py` 的 `AppConfig` 中加：

```python
class AppConfig(BaseModel):
    ...
    asr: ASRConfig = ASRConfig()
```

不破坏既有字段。

### 3.9 不再使用旧 SPEC 的 web-API 客户端
**SPEC-003 v1 提到的 ASRClient（web API 方式）已废弃**，对应代码模块不再创建。如未来确有需要可在新 spec 中另立 `SPEC-019` 描述 ASR-over-HTTP 适配器。

---

## SPEC-004 LiveSession 状态机

### 4.1 文件位置
`tts_service/live/session.py`。

### 4.2 状态枚举
```python
class LiveState(str, Enum):
    IDLE = "IDLE"
    AI_SPEAKING = "AI_SPEAKING"
    RECORDING = "RECORDING"
    DETECTING = "DETECTING"          # 段刚结束，等待 EndDetector 输出 trigger 决定下一段
    PAUSED = "PAUSED"                # 客户端断线或用户主动暂停，等待外部事件
    FINISHED = "FINISHED"
    ABANDONED = "ABANDONED"          # 用户取消会话，不合并
    ERROR = "ERROR"
```

### 4.3 状态合法性
定义合法转换表：

| 当前状态 | 合法下一状态 |
|---|---|
| IDLE | AI_SPEAKING, ERROR |
| AI_SPEAKING | RECORDING, DETECTING, PAUSED, FINISHED, ERROR |
| RECORDING | DETECTING, PAUSED, ERROR |
| DETECTING | AI_SPEAKING, RECORDING, FINISHED, ERROR |
| PAUSED | AI_SPEAKING, RECORDING, FINISHED, ERROR |
| FINISHED | (无) |
| ABANDONED | (无) |
| ERROR | (无) |

非法转换抛 `IllegalStateTransition`。

### 4.4 数据结构
```python
@dataclass
class LiveSession:
    session_id: str
    project_id: str
    cursor: int                       # next seg index
    state: LiveState
    segments: List[PodcastSegment]
    started_at: float
    finished_at: Optional[float]
    errors: List[str]
    pipeline: SessionAudioPipeline    # 录音 → 检测数据流（注入）
    ws: SessionWSContext               # WS 连接与帧收发（注入）
```

**不持有的资源**（通过注入获得）：
- `audio_buffer` → `SessionAudioPipeline`（§4.9）
- `alignment_score` / `last_asr_text` → `EndDetector`（由 Pipeline 持有）
- `ws_clients` / `driver_ws` → `SessionWSContext`（§4.10）
- `asr` → `LiveSessionRegistry`（§4.8，通过 `registry.get_asr()` 访问）

### 4.5 Session 注册表
`LiveSessionRegistry`（同文件）：
- 单例全局：`self._sessions: Dict[(project_id, session_id), LiveSession]`
- 提供 `create / get / stop / cleanup` 方法
- 后台任务每 30 s 清理已 FINISHED 超 5 分钟的 session

### 4.6 端点
- `POST /api/podcasts/{id}/live/start` → 创建 session，调度首段，返回 `{session_id}`
- `POST /api/podcasts/{id}/live/{session_id}/stop` → 优雅收尾
- `POST /api/podcasts/{id}/live/{session_id}/resume` → 从持久化文件恢复（SPEC-016）
- `DELETE /api/podcasts/{id}/live/{session_id}` → 取消（放弃）会话，不合并，WAV 保留磁盘
- `POST /api/asr/warmup` → 手动预热 ASR 模型

### 4.7 端点位置
在 `tts_service/server.py` 的现有 router 后追加；如不存在，新建 `@app.post("/api/podcasts/{pid}/live/start")` 风格 handler。

### 4.8 ASR 实例化与归属

**ASR 实例属于 Registry，不属于 Session。**

- `LiveSessionRegistry.__init__` 中创建 `self._asr = EmbeddedASR(self._app_cfg.asr)`
- 多 session 共享同一 ASR 实例（推理本身串行）
- `LiveSession` **不持有** `asr` 字段；需要 ASR 时通过 `registry.get_asr()` 获取
- `SessionAudioPipeline.push_pcm()` 内部调 `registry.get_asr().transcribe_chunk()`
- 服务启动钩子中**不**自动 `warmup()`（避免阻塞）

```python
class LiveSessionRegistry:
    _asr: EmbeddedASR          # 全局共享

    def get_asr(self) -> EmbeddedASR:
        return self._asr
```

### 4.9 SessionAudioPipeline
封装录音 → 检测的完整数据流，由 `LiveSession` 持有。

```python
@dataclass
class SessionAudioPipeline:
    audio_buffer: Dict[int, bytearray]   # seg index → captured PCM
    alignment_score: float
    last_asr_text: str
    end_detector: Optional[EndDetector]
    wav_writer: Optional[WavWriter]

    def push_pcm(self, pcm_bytes: bytes, seg_index: int) -> None:
        """接收 PCM 帧：写入 buffer + VAD dbfs + 下采样送 ASR 队列。"""

    def subscribe_triggers(self, callback) -> None:
        """注册 trigger 回调（end_near / end / user_skipped）。"""

    def reset_for_segment(self, seg_index: int, target_text: str) -> None:
        """进入新 RECORDING 段时重置 EndDetector 和 WavWriter。"""

    def flush_wav(self, seg_index: int, sample_rate: int) -> None:
        """end trigger 时将 buffer 写入 live_NNNN.wav + peak_normalize。"""
```

- `push_pcm()` 内部调 `registry.get_asr().transcribe_chunk()` 进行 ASR
- 溢出保护：`audio_buffer` 字节数超 `config.live.audio_buffer_max_mb` 时触发 `BUFFER_OVERFLOW`

### 4.10 SessionWSContext
封装 WS 连接管理和帧收发，由 `LiveSession` 持有。

```python
@dataclass
class SessionWSContext:
    ws_clients: Set[WebSocket]
    driver_ws: Optional[WebSocket]
    observers: Set[WebSocket]

    async def broadcast(self, frame: BaseModel) -> None:
        """向所有客户端推送 JSON 帧。"""

    async def push_audio(self, pcm_bytes: bytes) -> None:
        """向 driver 推送二进制音频帧。"""

    def add_client(self, ws: WebSocket, role: str) -> None:
        """添加客户端。role='driver' 或 'observer'。"""

    def remove_client(self, ws: WebSocket) -> Optional[str]:
        """移除客户端，返回断开的角色（'driver' 或 'observer'）。"""
```

---

## SPEC-005 流式 TTS + 音频队列

### 5.1 改动入口
- 新增 `tts_service/live/streaming_engine.py`，提供 `StreamingTTSProxy`。
- 不直接修改 `OmlxRemoteEngine` / `LocalVibeVoiceEngine`，保持主路径不变。

### 5.2 类接口
```python
class StreamingTTSProxy:
    def __init__(self, engine: BaseEngine):
        self.engine = engine
        self.sr = engine.sample_rate  # BaseEngine 必须实现 sample_rate 属性

    async def stream_segment(
        self,
        text: str,
        voice: str,
        chunk_ms: int = 200,
    ) -> AsyncIterator[bytes]:
        """Bulk-generate 后切片 + 间隔 yield"""
```

### 5.3 实现策略
1. 调用 `engine.generate_single(text, voice)` 拿到完整 WAV bytes
2. 跳过 44 字节 WAV header，对 PCM 部分按 `chunk_ms * sr / 1000 * 2` 字节切片
3. 每个切片 `await asyncio.sleep(0)` 让出事件循环
4. 在调度器层并行启动下一个 seg 的 `stream_segment`（look-ahead=1）

### 5.4 流式 HTTP 取（远期）
若 `OmlxRemoteEngine` 后续支持 SSE/分块返回（参考 `engines/base.py:198`），替换 5.3 第 1 步为流式 URL fetch；接口签名不变。

### 5.5 look-ahead 调度
`LiveSession` 启动时预生成 `cursor + 1` 段的音频至 `self.prefetch: asyncio.Queue[bytes]`，通过 `asyncio.create_task(self._prefetch_worker())`。

### 5.6 关键并发提示
- AI 播放路径（WS 推 PCM）和 ASR 推理路径（CPU/GPU 推理）必须在不同执行上下文：
  - AI 音频生成：在线程池里跑 `generate_single`（CPU 编码），生成完切片后在主事件循环里 yield
  - ASR 推理：单独线程池 `asr_pool`，LIVE 录制期间常驻
  - 两者不共享 Metal/GPU 资源（mlx-whisper vs qwen/VibeVoice 各自独占）

### 5.7 TTS 段超时与降级策略

**问题**：`engine.generate_single` 是全量生成后切片，用户从 `AI_SPEAKING` 开始到听到第一个音节，延迟等于整段 TTS 生成时间。
VibeVoice MLX 生成 30 秒音频约需 15-30 秒，远程 Qwen 可能更长。

**配置**（`config.live`）：
- `tts_max_seconds: int = 60` — 单段文本超过此时长（按字符估算）时自动切分为多个子段
- `tts_timeout_seconds: int = 30` — 单段 TTS 生成超时

**行为**：
1. 进入 `AI_SPEAKING` 前，检查段文本长度。若估算时长 > `tts_max_seconds`，按句号切分为多个子段，每个子段独立生成
2. `engine.generate_single()` 包装在 `asyncio.wait_for(timeout=tts_timeout_seconds)` 中
3. 超时时：跳过该段，用静音占位，发 `error` 帧（code=`TTS_TIMEOUT`），继续下一段
4. 生成失败时：重试 1 次（无退避），仍失败则跳过，发 `error` 帧（code=`TTS_ERROR`）

**对应 Story**：ST-LP-019。配置项见 SPEC-001 §1.4 的 `LiveConfig` 定义。

### 5.8 BaseEngine.sample_rate 属性要求

`BaseEngine` 必须实现 `sample_rate` 属性，返回引擎原生输出采样率：

```python
class BaseEngine(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of this engine."""
        ...
```

各子类实现：
- `LocalVibeVoiceEngine.sample_rate` -> `24000`
- `QwenRemoteEngine.sample_rate` -> 从首次返回的 WAV 头解析，或通过配置指定

此属性被 `StreamingTTSProxy`、`audio_info` 帧、合并采样率协商等多处使用。

---

## SPEC-006 WebSocket 通道

### 6.1 端点
`WS /ws/podcasts/{pid}/live/{sid}?role=driver|observe`

### 6.2 帧协议

**Server → Client** JSON:
```json
{ "type": "state", "state": "AI_SPEAKING" }
{ "type": "segment_start", "index": 3, "source": "tts", "speaker": "Flora", "text": "..." }
{ "type": "asr_partial", "text": "今天我们", "audio_ms": 1200 }
{ "type": "asr_final", "text": "今天我们聊...", "matched_ratio": 0.95 }
{ "type": "alignment", "matched_chars": 42, "total_chars": 48 }
{ "type": "asr_warming", "progress": 0.6 }
{ "type": "asr_unavailable", "reason": "asr disabled" }
{ "type": "asr_degraded", "consecutive_failures": 5 }
{ "type": "audio_info", "sample_rate": "<engine_sample_rate>", "channels": 1, "bit_depth": 16 }
{ "type": "error", "code": "ASR_DOWN", "message": "..." }
```

**Server → Client** Binary:
- TTS 引擎原生采样率 mono PCM（裸字节流，16-bit little-endian，无 WAV header）。采样率由 `audio_info` 帧协商，从 `BaseEngine` 实例动态获取。

**Client → Server** Binary:
- 设备原生采样率 mono PCM（推荐 48 kHz，`{ ideal: 48000 }`，16-bit little-endian）。实际采样率由客户端通过 `audio_info` 帧告知服务端。服务端内部下采样到 16 kHz 供 ASR 使用，同时保留原始采样率 PCM 用于 WAV 落盘。

**Client → Server** JSON:
- `{"type": "client_audio_info", "sample_rate": 48000, "channels": 1, "bit_depth": 16}` — 连接建立后第一条 JSON 帧，告知服务端实际录音采样率
- `{"type": "state_ack", "state": "AI_SPEAKING"}` — 确认状态转换完成，服务端收到后开始推 TTS 二进制帧
- `{"type": "client_log", "level": "info", "msg": "..."}`

### 6.3 实现
在 `tts_service/server.py` 注册：
```python
@app.websocket("/ws/podcasts/{pid}/live/{sid}")
async def ws_live(ws: WebSocket, pid: str, sid: str, role: str = "observe"):
    await ws.accept()
    session = registry.get(pid, sid)
    if not session: await ws.close(1008); return
    if role == "driver":
        if session.driver_ws is not None:
            await ws.close(1008, "driver already connected"); return
        session.driver_ws = ws
    session.ws_clients.add(ws)
    try:
        # 推 frame loop + 收 binary 写 audio_buffer[seg_index]
        ...
    finally:
        session.ws_clients.discard(ws)
        if session.driver_ws is ws:
            session.driver_ws = None
            session.transition(LiveState.WAITING_TRIGGER)
```

### 6.4 重连
客户端用断线重连参数：`reconnect_attempts=5, reconnect_delay_ms=1000`。若断线期间 server 处于 `RECORDING`，server 端 buffer 暂存但**不再接受新 binary**，超时 30 s 自动 `WAITING_TRIGGER`。

---

## SPEC-007 前端录音

### 7.1 文件位置
`tts_service/static/live.js`（新增模块）+ `index.html` 中挂载 `<script src="/static/live.js" defer>`。

### 7.2 录音启动
```js
const stream = await navigator.mediaDevices.getUserMedia({
  audio: { channelCount: 1, sampleRate: { ideal: 48000 }, echoCancellation: true, noiseSuppression: true }
});
const ctx = new AudioContext({ sampleRate: stream.getAudioTracks()[0].getSettings().sampleRate });
const src = ctx.createMediaStreamSource(stream);
const proc = ctx.createScriptProcessor(4096, 1, 1);
proc.onaudioprocess = e => {
  const pcm = e.inputBuffer.getChannelData(0);
  const int16 = new Int16Array(pcm.length);
  for (let i = 0; i < pcm.length; i++) int16[i] = Math.max(-1, Math.min(1, pcm[i])) * 0x7FFF;
  ws.send(int16.buffer);
};
src.connect(proc).connect(ctx.destination);
```
客户端以设备原生采样率（推荐 48 kHz）采集并发送 PCM。服务端在内部下采样到 16 kHz 供 ASR 推理，同时保留原始采样率数据用于 WAV 落盘。

注：用 `ScriptProcessor` 而非 `AudioWorklet`，因为 IE/Safari 兼容更好，且实现简洁；若未来要降延迟迁移到 Worklet，接口形态保持一致。

### 7.3 静音控制
`stream.getAudioTracks().forEach(t => t.enabled = (state === "RECORDING"));`

### 7.4 Pre-roll beep
进入 `RECORDING` 立即 `ctx.createOscillator()` 800 Hz / 250 ms / gain 0.1，连到 `ctx.destination`，结束停振。

### 7.5 Mute 自我监听
录制时 `AI_SPEAKING` 段仍走扬声器，但用户应戴耳机；UI 顶部显示红色提示条 "请佩戴耳机以避免回授"。

---

## SPEC-008 文本对齐 + VAD 讲完检测

### 8.1 模块
`tts_service/live/end_detector.py`

### 8.2 接口
```python
class EndDetector:
    def __init__(self, target_text: str, silence_db_threshold: float = -45.0,
                 end_near_threshold: float = 0.85,
                 end_alignment_threshold: float = 0.98,
                 end_silence_ms: int = 300,
                 force_end_silence_ms: int = 4000,
                 debounce_ms: int = 200):
        self.target = target_text
        self.normalized_target = normalize_text(target_text)
        self.silence_ms = 0
        self.last_trigger = None

    def update_vad(self, dbfs: float, frame_ms: int = 20) -> None: ...

    def update_asr(self, asr_text: str) -> Optional[Trigger]:
        """Returns 'end_near', 'end', 'user_skipped', or None"""
```

### 8.3 文本归一
`normalize_text`：去空白、繁简统一（`zhconv.convert(s, 'zh-cn')`）、去掉标点、转小写。

### 8.4 对齐算法
用 **最长公共子序列（LCS）** 计算匹配率：
```python
ratio = len(lcs) / max(1, len(normalized_target))
```
LCS 优于子串包含：可应对用户跳词、轻微顺序调整。

### 8.5 Trigger 发射
`update_asr` 内部状态机：

| 输入 | 状态 |
|---|---|
| ratio >= 0.85 且未触发 end_near | emit `end_near` |
| ratio >= 0.98 且 silence_ms >= 300 | emit `end` |
| silence_ms >= 800 | emit `end` |
| silence_ms >= 4000 | emit `user_skipped` |

`debounce_ms` 限制同一 trigger 200 ms 内只发一次。

### 8.6 VAD
能量计算：每 20 ms 一帧 RMS → dBFS = `20 * log10(rms + 1e-7)`；低于阈值则累加 silence_ms。

### 8.7 VAD 数据源
VAD 不依赖 ASR，**它直接读 WS 收到的 PCM 帧**（每 20 ms 一个 frame），由 LiveSession 单独开一个线程计算能量 → dbfs → 推进 `EndDetector.silence_ms`。这样 VAD 即便 ASR 故障也能正常判断讲完。

---

## SPEC-009 多源合并

### 9.1 改动
`podcast_manager.py:266-311` 的 `merge_project()`：
- 现有：调用 `_concatenate_audio_segments(audio_parts, ...)`
- 新增：解析每个段的 `source`：
  - `source == "tts"` → 现有路径
  - `source == "live"` → `live_{index:04d}.wav`

### 9.2 缺文件兜底
```python
if seg.source == SegmentSource.LIVE and not live_path.exists():
    audio_parts.append(self._make_silence(1.0))
    seg.status = "missing"
    warnings.append(f"segment {i}: live audio missing, substituted silence")
else:
    audio_parts.append(load(live_path))
```

`_make_silence()` 复用 `engines/base.py:313-323` 的 `anullsrc` 生成函数。

### 9.3 现状兼容
所有段仍为 `tts` 的项目，合并路径不变 → 零回归。

### 9.4 采样率统一
合并输出采样率统一到 TTS 引擎原生采样率与真人录音采样率中**较低者**。真人段在合并前通过 ffmpeg 下采样到该目标采样率。上采样不产生新信息量，因此不作升频。

---

## SPEC-010 真人段 WAV 落盘

### 10.1 落盘位置
`outputs/podcasts/{project_id}/live_{index:04d}.wav`，与 `seg_{index:04d}.wav` 同目录。

### 10.2 WAV 头生成
```python
def write_wav(path: Path, pcm_bytes: bytes, sample_rate: int, channels: int = 1):
    import wave
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)   # 设备原生采样率，推荐 48 kHz
        w.writeframes(pcm_bytes)
```
`sample_rate` 由客户端通过 WS `audio_info` 帧告知，写入 WAV 头保持一致。

### 10.3 归一化
落盘后调用 `tts_service/sample_manager.peak_normalize()`（已存在，`sample_manager.py:23-118` 内）将峰值归到 0.95。

**采样率兼容性**：`peak_normalize()` 接收 `np.ndarray` 和 `sample_rate` 参数，归一化操作不依赖特定采样率（仅计算 peak amplitude 并缩放），因此对任意采样率的 live 段均可安全调用。

### 10.4 流式写
为减少内存峰值，使用 `wave.open` 以 append 模式持续 `writeframes(binary_chunk)`；每帧（audio_frame）落入时不整体加载到内存。

### 10.5 不自动清理
只在 session `FINISHED` 后保留；abandoned session 的 wav 文件由 SPEC-016 的 24 h 后台任务清理。

---

## SPEC-011 前端实时播放

### 11.1 AudioContext 链
```js
const audioCtx = new AudioContext({ sampleRate: engineSampleRate });  // 由 audio_info 帧协商
const gain = audioCtx.createGain();
gain.connect(audioCtx.destination);

ws.addEventListener('message', async ev => {
  if (ev.data instanceof ArrayBuffer) {
    const buf = await audioCtx.decodeAudioData(ev.data);
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(gain);
    src.start(nextStartTime);
    nextStartTime = buf.duration;
  }
});
```

### 11.2 Jitter buffer
```js
const JITTER_MIN_MS = 600, JITTER_MAX_MS = 1500;
let bufferMs = 0;
// 在每帧 decodeAudioData 后 bufferMs += frame.duration * 1000
// 当 bufferMs >= JITTER_MIN_MS 时启用播放；溢出则丢帧
```

### 11.3 静音扬声器
录制时：
```js
gain.gain.value = 0;  // 静音本地监听，避免回授
```
但仍在 UI 显示可视化波形（来自 `MediaRecorder` 的电平表）。

---

## SPEC-012 ASR 中间结果 UI（可选）

### 12.1 DOM 结构
```html
<div id="live-panel" hidden>
  <div class="live-target">{segment.text}</div>
  <div class="live-asr"></div>
  <button id="live-start">Start Live</button>
  <button id="live-stop">Stop</button>
  <div class="live-vu"><canvas id="vu"></canvas></div>
</div>
```

### 12.2 高亮算法
收到 `asr_partial` 文本后：
1. 在 `live-target` 上跑 LCS 对齐（与 SPEC-008 同样的逻辑，前端实现一次）
2. 把 matched 字符 `<span class="hit">` 包裹，命中区绿色
3. 未命中灰色；ASR 中多余内容折叠到 `live-asr` 行

---

## SPEC-013 重录单段

### 13.1 端点
`POST /api/podcasts/{pid}/live/{sid}/redo/{index}`
- 限制：仅 `live` 段可 redo；其他段返回 400。
- 行为：把 cursor 移到该 index；若 `live_{index:04d}.wav` 已存在则删除。

### 13.2 冲突保护
若 cursor > index 且当前正在 `RECORDING` 后续段，返回 409 "session busy"。

### 13.3 UI
段编辑器（`app.js:1013-1327` 现有段编辑 UI）加 "Re-record" 按钮：仅当 `seg.source == "live"` 且当前无活跃 session 时启用。

---

## SPEC-014 Live 段删除

### 14.1 端点
`DELETE /api/podcasts/{pid}/segments/{i}`

### 14.2 行为
1. 删除 `seg_{i:04d}.wav` 或 `live_{i:04d}.wav`
2. 从 `project.segments` 弹出第 i 项
3. **重新编号**：所有 seg N+1..end 的 audio_filename 同步更新
4. 把 audio 文件重命名：`seg_{n:04d}.wav` → `seg_{n-1:04d}.wav`（live 同理）
5. 重写 `project.json`

### 14.3 保护
若当前有活跃 session 涉及该段，返回 409。

### 14.4 工具函数
`podcast_manager.compact_segment_files(project_id)`：返回 `(success, error_list)`。

---

## SPEC-015 Live-only 录制模式

### 15.1 启动参数
```python
class LiveStartRequest(BaseModel):
    live_only: bool = False
```

### 15.2 行为
开启后：
- `tts` 段被跳过，不调度 TTS 生成
- 合并时插入 1 s 静音代替（用 `_make_silence`）
- cursor 仍按 segment 顺序推进

### 15.3 UI
"Start Live (Skip AI)" 复选框 → 启动时 `live_only: true`。

---

## SPEC-016 错误恢复

### 16.1 持久化
每 5 s `LiveSession` 序列化到：
`outputs/podcasts/{project_id}/live_sessions/{session_id}.json`
字段：
```json
{
  "cursor": 3,
  "state": "RECORDING",
  "captured_segments": [0, 1, 2],
  "started_at": 1700000000.0,
  "last_save_at": 1700000123.0
}
```

### 16.2 启动时扫描
服务启动 hook（`tts_service.cli.main()` 或 `create_app` 末尾）：
```python
for f in outputs_dir.glob("podcasts/*/live_sessions/*.json"):
    registry.register_resumable(load_json(f))
```

### 16.3 恢复端点
`POST /api/podcasts/{pid}/live/{sid}/resume`
- 若当前 session 不在内存中，从 JSON 重建（不带音频 buffer）
- 重连 driver WS 后从 cursor 继续

### 16.4 自动过期
`registry.cleanup()` 每小时一次：finished > 1 h / orphan > 24 h → 删除 JSON + live wav。

### 16.5 异常恢复时的 ASR
恢复时也复用同一 `EmbeddedASR` 实例；模型已在内存中，无需重新加载。

### 16.6 LiveSession ↔ PodcastManager 同步协议

**两层写入策略**：

| 层级 | 触发 | 写入目标 | 实现 |
|---|---|---|---|
| 实时层 | 每 5s + 状态转换 | `live_sessions/{sid}.json` | `persist_snapshot()` |
| 定稿层 | `stop()` / `cancel()` | `{pid}.json` + 段文件 | `PodcastManager` |

- 正常运行时 `LiveSession` 不调用 `PodcastManager._save_project()`（避免频繁 IO）
- `stop()` 时遍历已录制段，更新 status，通过 `PodcastManager` 写回 `project.json`
- `cancel()` 时不更新 `project.json`，`live_NNNN.wav` 保留等 24h 清理
- `redo()` 只删 `live_NNNN.wav`，不改 `project.json`

---

## SPEC-017 多客户端观察模式

### 17.1 连接角色
WS URL 参数 `?role=observe`：注册为 observer，只接收 server → client 帧。
非 driver 试图发送 binary → server close(1003, "observe role cannot send audio")。

### 17.2 角色提升
`{"type": "claim_driver"}` 帧：当前 driver 5 s 内无响应 → 升级为 driver。
实现：driver 心跳每 2 s 一次 `{"type": "heartbeat"}`；连续 3 次丢失触发升级窗口。

### 17.3 限制
`{project_id, session_id}` 同时只允许 1 个 driver；observer 上限 10。

---

## SPEC-018 实时 AI 段落续接

### 18.1 触发
`LiveSession` 处于 `AI_SPEAKING` 状态 ≥ 8 s 且下一段为 `tts` 时，触发续接。

### 18.2 行为
1. 给当前段 prepend 一个 1.5 s 静音（用 `_make_silence(1.5)`）
2. 重新调 `engine.generate_single(text, voice, instructions="<温和平静>")` 生成新版
3. 用新版替换原有 chunk 流（通过中止当前 `StreamingTTSProxy` task 并启动新的）

### 18.3 限制
单段最多重试 2 次；都失败则用首次结果。

### 18.4 关闭开关
`config.live.max_ai_rephrases_per_segment: int = 2`，设为 0 关闭。

---

## 依赖与模块清单

| 新增模块 | 路径 | 行数估计 |
|---|---|---|
| Live session 状态机 | `tts_service/live/session.py` | ~250 |
| ASR 内嵌引擎 | `tts_service/live/asr_engine.py` | ~200 |
| 流式 TTS 代理 | `tts_service/live/streaming_engine.py` | ~100 |
| 讲完检测 | `tts_service/live/end_detector.py` | ~120 |
| WAV 写盘 | `tts_service/live/wav_writer.py` | ~60 |
| WebSocket handler | 追加 `tts_service/server.py` | ~120 |
| 配置扩展 | 追加 `tts_service/config.py` | ~30 |
| 数据模型扩展 | 追加 `tts_service/models.py` | ~15 |
| 前端实时模块 | `tts_service/static/live.js` | ~400 |
| 前端样式 | 追加 `tts_service/static/styles.css` | ~80 |
| 单测 | `tests/test_live_*.py` × 4 | ~400 |

`requirements.txt` 追加：`mlx-whisper>=0.4.0`（Apple Silicon）或 `faster-whisper>=1.0.0`（跨平台）。

**总计约 1750 行新增/改动。**

---

## ASR 延迟预算（验证 SPEC-003 性能目标）

| 阶段 | 时间 | 备注 |
|---|---|---|
| 浏览器录音 → WS 上行 | ≤ 100 ms | LAN 内 localhost |
| 服务端收到 PCM → VAD | < 1 ms | 简单能量计算 |
| 服务端收到 PCM → ASR 输入队列 | < 1 ms | 含 48k→16k 下采样 |
| ASR 推理（whisper-small, 1 s chunk） | ≤ 250 ms | mlx / faster-whisper GPU |
| ASR → LCS 对齐 | ≤ 5 ms | 文本 ≤ 100 字 |
| 对齐 → WS 下行 frame | ≤ 1 ms | |
| 客户端刷新 UI | ≤ 16 ms | 一帧 |
| **单 tick 端到端** | **≤ 400 ms** | 满足"贴"听感 |

VAD 路径独立，端到端 ≤ 50 ms（仅 PCM 解析 + 能量计算），作为对齐失败兜底。

---

## 采样率架构

```
录音层（设备原生采样率）    →  WAV 落盘保持原始采样率，保留原始质量
  ├─ 下采样 → 16 kHz       →  ASR 推理（whisper 原生 16k）
  └─ 下采样 → 合并目标采样率 →  合并时统一到较低的采样率

TTS 层（引擎原生采样率）    →  引擎原生输出，不重采样
  └─ 播放 → 引擎原生采样率  →  前端 AudioContext 跟随引擎采样率

合并层                       →  真人段下采样到较低者，统一混流
```

**原则：**
- 录音端和 TTS 端各自保持最高采样率，不作降质
- ASR 独立下采样路径，不影响录制质量
- 合并时按两者中**较低者**下采样，不上采样（上采样不产生新信息量）
- 不对任何采样率写死具体数值——由引擎和设备的实际能力决定

---

## 实施顺序（推荐 6 个里程碑）

| 里程碑 | 内容 | 验收 |
|---|---|---|
| **M1** | SPEC-001, 002, 010 — 配置 + 模型 + WAV 落盘 | 旧项目零回归；新建 live speaker 项目 metadata 正确 |
| **M2** | SPEC-004 — LiveSession 状态机 + 端点（含 cancel） | `/api/.../live/start/stop` + `DELETE .../live/{sid}` 在 mock 数据下走通 |
| **M3** | SPEC-005, 006, 011, ST-LP-019 — 流式 TTS + WS + 前端播放 + TTS 超时降级 | 浏览器看到 AI 段音频在 ≤ 1 s 内开始；TTS 超时后自动跳过 |
| **M4** | SPEC-007, 010 — 前端录音 + 落盘 | 浏览器录音 → 服务器端 WAV 文件落盘验证 |
| **M5** | SPEC-003, 008 — ASR 引擎 + 讲完检测 | 在 mlx-whisper 加载完成后 end_near / end 触发正确 |
| **M6** | SPEC-009, 013-018 — 多源合并 + 重录 + 观察 + 错误恢复 | 全流程 e2e demo 通 |

每个里程碑结束跑：
- `python -m unittest tests/test_live_*.py`
- `python regression_test.py`（确保 0 回归）
- M5 起跑 `tests/test_asr_latency.py`：1 s chunk 推理 P95 ≤ 300 ms