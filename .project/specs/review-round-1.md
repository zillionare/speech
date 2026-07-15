# Architecture & Interface Review — Round 1

> 审查对象：`story.md`（ST-LP-001 ~ ST-LP-018）和 `spec.md`（SPEC-001 ~ SPEC-018）
> 审查日期：2026-07-15

---

## 1. 架构问题

### 1.1 `LiveSession` 职责过重（God Object）

SPEC-004 §4.4 的 `LiveSession` dataclass 同时持有：

- session 状态（`state`, `cursor`, `started_at`, `finished_at`）
- ASR 引擎引用（`asr: EmbeddedASR`）
- WS 客户端集合（`ws_clients: Set[WebSocket]`）
- 音频 buffer（`audio_buffer: Dict[int, bytearray]`）
- 对齐状态（`alignment_score`, `last_asr_text`）

这是 God object 的早期信号。随着功能增长，该类将变得难以测试和维护。

**建议拆分为三个独立模块：**

| 模块 | 职责 |
|------|------|
| `LiveSession` | 纯状态机 + cursor + segment 快照，不持有 IO 资源 |
| `SessionAudioPipeline` | 管理 PCM buffer、VAD 计算、ASR 调用、EndDetector、WAV 流式写入，封装录音→检测的完整数据流 |
| `SessionWSContext` | 管理 WS 连接、driver/observer 角色、binary/JSON 帧收发 |

每个模块可独立单元测试，`LiveSession` 通过注入获得 `SessionAudioPipeline` 和 `SessionWSContext`。

---

### 1.2 `asr` 字段归属矛盾

- SPEC-004 §4.4 在 `LiveSession` 上放了 `asr: EmbeddedASR`
- SPEC-004 §4.8 说 ASR 实例在 `LiveSessionRegistry` 中全局共享、多 session 复用

这两处矛盾。ASR 实例应属于 Registry，session 通过 registry 间接引用，不直接持有。

```python
# 建议
class LiveSession:
    ...
    # 不持有 asr 字段
    pass

class LiveSessionRegistry:
    _asr: EmbeddedASR          # 全局共享
    _sessions: Dict[...]

    def get_asr(self) -> EmbeddedASR:
        return self._asr
```

---

### 1.3 `WAITING_TRIGGER` 语义过载

当前 `WAITING_TRIGGER` 同时覆盖两种场景：

| 场景 | 含义 |
|------|------|
| 段刚结束，等待 EndDetector 输出 trigger 决定下一段 | 正常调度决策中 |
| 客户端断线，等待重连 | 暂停 |

两种场景的恢复逻辑完全不同：前者是自动推进，后者是等待外部事件。建议拆分：

```
WAITING_TRIGGER  →  DETECTING   （正常：等待 ASR/VAD 检测结果）
                 →  PAUSED      （异常：断线、用户暂停）
```

状态转换表需相应更新：

| 当前状态 | 合法下一状态 |
|---|---|
| `RECORDING` | `DETECTING`, `ERROR` |
| `DETECTING` | `AI_SPEAKING`, `RECORDING`（redo）, `ERROR` |
| `AI_SPEAKING` | `RECORDING`（下一段是 live）, `PAUSED`, `FINISHED`, `ERROR` |
| `PAUSED` | `AI_SPEAKING`, `RECORDING`, `FINISHED`, `ERROR` |

---

### 1.4 TTS "全量生成→切片" 的延迟上限未量化

SPEC-005 §5.3 是先调 `engine.generate_single(text, voice)` 拿到完整 WAV，再切 200ms 片 yield。这导致用户从 `AI_SPEAKING` 开始到听到第一个音节，延迟等于整段 TTS 生成时间。

VibeVoice MLX 生成 30 秒音频约需 15-30 秒，远程 Qwen 可能更长。如果首段 TTS 需要 20 秒生成，用户会看到 20 秒的静默。

**建议：**
- 在 spec 中明确 TTS 段的最大长度限制（如 `max_tts_segment_chars`），超过则自动切分成多个子段，缩短首段生成延迟
- 定义 TTS 生成超时（如 30 秒），超时则跳过该段、用静音占位，避免阻塞整个 session
- 在 story 中增加 **ST-LP-019**：TTS 段超时与降级策略

---

### 1.5 `BaseEngine` 缺少 `sample_rate` 属性

SPEC-006 说 `audio_info` 帧的 `sample_rate` 从 `BaseEngine` 实例动态获取，但现有 `BaseEngine`（`engines/base.py:97`）没有此属性。

需要在 `BaseEngine` 中新增：

```python
class BaseEngine(ABC):
    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Native output sample rate of this engine."""
        ...
```

各子类实现：
- `LocalVibeVoiceEngine.sample_rate` → 24000
- `QwenRemoteEngine.sample_rate` → 从首次返回的 WAV 头解析，或通过配置指定

---

## 2. 接口问题

### 2.1 `audio_info` 握手不对称

当前只有 Server→Client 的 `audio_info` 帧（SPEC-006 §6.2）。Client→Server 也需要发送 `audio_info` 告知实际录音采样率。

`{ ideal: 48000 }` 不保证被浏览器采纳——实际可能是 44100 或 48000。服务端需要知道真实值才能正确写 WAV 头。

**缺失帧：**

```json
// Client → Server，连接建立后第一条 JSON 帧
{"type": "audio_info", "sample_rate": 44100, "channels": 1, "bit_depth": 16}
```

服务端据此设定 WAV 落盘采样率和 ASR 下采样比例。

---

### 2.2 LiveSession ↔ TTS Engine 的接口未定义

`LiveSession` 需要在 `AI_SPEAKING` 时触发 TTS 生成、获取 engine `sample_rate`、look-ahead 预生成。但 spec 没有定义 session 如何获得 engine 引用。

**建议：** 在 `start` handler 中创建 `StreamingTTSProxy` 并注入到 `LiveSession`：

```python
# server.py
@app.post("/api/podcasts/{pid}/live/start")
async def live_start(pid: str):
    session = registry.create(pid)
    engine = get_engine(config, sample_manager)
    proxy = StreamingTTSProxy(engine, engine.sample_rate)
    session.set_tts_proxy(proxy)
    await session.start()
    return {"session_id": session.session_id}
```

---

### 2.3 组件间数据流未明确

四个核心组件之间的数据流缺乏定义：

```
WS Handler ──→ audio_buffer ──→ VAD thread ──→ EndDetector.silence_ms
            │                 │
            │                 └──→ ASR (48k→16k) ──→ EndDetector.update_asr()
            │
            └──→ WAV writer (streaming append)

TTS Scheduler ──→ StreamingTTSProxy ──→ WS sender (binary chunks)
```

需要明确的问题：
- VAD 和 ASR 是否共享同一个 PCM ring buffer？锁策略是什么？
- audio_buffer 是否有容量上限？溢出时的行为？
- WAV writer 和 ASR 是否在同一个线程？如果不是，如何保证 WAV 写入原子性？

**建议：** 在 spec 中增加一个数据流架构图（`DataFlow` 章节），明确线程模型和 buffer 边界。

---

### 2.4 `segment_start` 到首帧音频的时序缺口

`segment_start` JSON 帧后紧跟二进制音频帧。客户端收到 `segment_start` 后需要时间初始化 `AudioBufferSourceNode`。如果音频帧立即到达，首帧可能丢失。

**建议：** 在 `segment_start` 后增加一个显式帧或延迟：

```
Option A: {"type": "segment_audio_begin", "index": N}  →  客户端准备就绪  →  开始推二进制帧
Option B: segment_start 后服务端 sleep 100ms 再推音频帧
```

推荐 Option A，更可靠。

---

### 2.5 `RECORDING → AI_SPEAKING` 转换时的双向数据竞态

当 `end` 触发、状态从 `RECORDING` 切到 `AI_SPEAKING` 时：
- 客户端可能还在发送最后一帧录音 PCM（1 秒 timeslice 的残帧）
- 服务端可能已经开始推送 TTS chunk

WS 没有帧优先级，JSON 控制帧和二进制音频帧在同一个 socket 上串行传输。如果 TTS 二进制帧在 `state: "AI_SPEAKING"` JSON 帧之前到达，客户端可能误解。

**建议：** 状态转换时，服务端先发送 `state` JSON 帧，等待客户端 ACK（`{"type": "state_ack", "state": "AI_SPEAKING"}`），再开始推送 TTS 二进制帧。ACK 超时 500ms 则直接推送（降级）。

---

### 2.6 `LiveSession` 与 `PodcastManager` 的职责边界模糊

`LiveSession` 快照了 segments 列表（SPEC-004 §4.4），但修改 segment 状态（如 `status = "recorded"`）应该通过 `PodcastManager` 还是直接操作内存对象？

如果直接操作，`PodcastManager._save_project()` 不会被调用，磁盘 project.json 不同步。如果通过 `PodcastManager`，每次状态变更都需要 IO。

**建议：** 明确为两层写入策略：
- 实时状态变更（cursor 推进、status 变更）→ 由 `LiveSession` 写入内存快照 + 周期性持久化（SPEC-016 的 5 秒 JSON）
- 最终定稿（stop/merge）→ 通过 `PodcastManager` 写回 `project.json` 和合并音频

---

### 2.7 缺少 `cancel`/`abort` 端点

只有 `start` 和 `stop`。`stop` 的行为是"定稿合并"，与"放弃"语义不同。

**建议新增：**

```
DELETE /api/podcasts/{pid}/live/{sid}
```

行为：立即终止 session，不合并，已录制的 `live_NNNN.wav` 保留在磁盘（由 24h 清理任务处理），session 状态标记为 `ABANDONED`。

---

### 2.8 `peak_normalize` 的多采样率兼容性

SPEC-010 §10.3 说用 `sample_manager.peak_normalize()` 归一化。该函数（`sample_manager.py`）接收 `np.ndarray` 和 `sample_rate` 参数，不依赖特定采样率，兼容性没问题。但 spec 应确认此点。

---

## 3. 缺失的接口/Story

| 缺失项 | 优先级 | 说明 |
|--------|:------:|------|
| `BaseEngine.sample_rate` 属性 | P0 | 引擎必须暴露原生采样率，否则 WS 协商、合并重采样都无法获取 |
| Client→Server `audio_info` 帧 | P0 | 客户端告知服务端实际录音采样率 |
| `DELETE /api/podcasts/{pid}/live/{sid}` cancel 端点 | P1 | 放弃当前 session |
| `LiveSession` 与 `PodcastManager` 的同步协议 | P0 | 明确状态变更何时写回 project.json |
| 数据流架构图（线程模型 + buffer 边界） | P1 | VAD/ASR/WAV writer 之间的共享内存和锁策略 |
| TTS 段最大长度限制 + 超时降级策略 | P1 | 避免首段生成延迟过长 |
| `state_ack` 帧协议 | P1 | 解决状态转换竞态 |
| `segment_audio_begin` 帧 | P2 | 解决首帧丢失 |
| TTS 生成失败的重试策略 | P1 | 几次重试？退避策略？降级行为？ |

---

## 4. 建议的模块依赖图

```
server.py (WS handler)
  │
  ├── LiveSessionRegistry (单例)
  │     ├── EmbeddedASR (全局共享，所有 session 复用)
  │     └── Dict[(pid, sid), LiveSession]
  │           ├── EndDetector (per recording segment)
  │           ├── SessionAudioPipeline
  │           │     ├── PCM ring buffer
  │           │     ├── VAD calculator (独立线程)
  │           │     └── WAV writer (streaming append)
  │           └── cursor → segments (快照)
  │
  ├── StreamingTTSProxy (注入到 session)
  │     └── BaseEngine (现有)
  │
  └── PodcastManager (project CRUD, merge)
        └── 最终定稿时写回 project.json
```

**关键约束：**
- `LiveSession` 不直接操作 `project.json`——通过 `PodcastManager` 读写
- `StreamingTTSProxy` 由 handler 创建并注入，session 不从内部自行获取
- `EmbeddedASR` 属于 Registry，不属于 session
- `SessionAudioPipeline` 封装录音→检测的完整数据流，对外暴露 `push_pcm(bytes)` 和 `subscribe_triggers(callback)`

---

## 5. 建议的 spec 新增章节

### 5.1 数据流架构（`DataFlow`）

```
[浏览器] ──WS binary──→ [server.py]
                            │
                    ┌───────┴────────┐
                    ▼                ▼
            SessionAudioPipeline  StreamingTTSProxy
              │    │                    │
              ▼    ▼                    ▼
            VAD   ASR               BaseEngine
              │    │               (generate_single)
              ▼    ▼
           EndDetector
                │
                ▼
         LiveSession (状态转换决策)
```

### 5.2 线程模型

| 线程 | 职责 | 阻塞容忍 |
|------|------|----------|
| 主事件循环 | WS 收发、状态转换、JSON 帧处理 | 不允许阻塞 > 50ms |
| `asr_pool`（线程池） | `EmbeddedASR.transcribe_chunk`（同步） | 允许 250ms |
| `tts_pool`（线程池） | `engine.generate_single`（同步） | 允许 30s |
| `vad_thread` | 从 ring buffer 读 PCM、计算 dBFS、更新 EndDetector | 不允许阻塞 |

---

## 6. 总结

| 维度 | 评价 |
|------|------|
| 功能完整性 | 覆盖了核心流程，缺少 cancel、TTS 超时降级、双端 `audio_info` 协商 |
| 接口清晰度 | WS 帧协议定义较好，但组件间内部接口（Session↔Engine↔Pipeline）模糊 |
| 职责分离 | `LiveSession` 过重，需要拆分为 state/pipeline/WS context 三层 |
| 线程安全 | 提及了 ASR 和 TTS 路径分离，但 VAD/ASR/WAV 共享 buffer 的策略未定义 |
| 可测试性 | 拆分后每个模块可独立单测；当前 God object 形态难以测试 |