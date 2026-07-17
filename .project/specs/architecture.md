# Live Podcast - Architecture（架构设计）

> 本文档与 `story.md` / `spec.md` 编号对齐：`ARCH-NNN ↔ SPEC-NNN ↔ ST-LP-NNN`。
> 目的：定义模块边界、组件依赖、数据流、状态机、并发模型，作为 interface.md 与实现的依据。
>
> **复用原则**：现有 `tts_service/server.py::create_app()` 是闭包工厂（无 router/blueprint），
> 所有路由通过 `@app.xxx` 在闭包内注册，可直接访问 `engine` / `sample_manager` / `podcast_manager` / `config`
> 闭包变量。Live 子系统**追加在同一闭包内**，不引入新的依赖注入机制。

---

## 1. 顶层模块划分

```
tts_service/
├── server.py                # 闭包工厂 + 路由（追加 live 路由 + WS handler）
├── config.py                # Config Pydantic 模型（追加 ASRConfig / LiveConfig）
├── models.py                # API Pydantic 模型（扩展 PodcastSegment.source / 新增 Live* 模型）
├── podcast_manager.py       # 播客项目持久化（扩展 merge_project / regenerate_segment 早返回）
├── engines/
│   └── base.py              # TTS 引擎基类（不修改，仅复用 _generate_silence / _concatenate_audio_segments）
└── live/                    # 【新增子包】Live Podcast 运行时
    ├── __init__.py
    ├── session.py           # LiveSession 纯状态机 + LiveSessionRegistry（不持有 IO 资源）
    ├── audio_pipeline.py    # SessionAudioPipeline — PCM ring buffer / VAD / ASR 调度 / EndDetector / WavWriter
    ├── ws_context.py        # SessionWSContext — WS 连接管理 / driver-observer 角色 / binary-JSON 帧收发
    ├── asr_engine.py        # EmbeddedASR 进程内推理
    ├── streaming_engine.py  # StreamingTTSProxy 流式切片
    ├── end_detector.py      # LCS 对齐 + VAD 讲完检测
    ├── wav_writer.py        # 流式 WAV 落盘 + 归一化
    ├── audio_resampler.py   # 采样率转换（48k -> 16k for ASR；落盘保持原采样率）
    └── frames.py            # WS 帧的 Pydantic 模型 + 序列化
```

### 1.1 模块职责矩阵

| 模块 | 职责 | 不做 |
|---|---|---|
| `live/session.py` | 纯状态机（LiveState 枚举 + transition）、cursor 推进、段调度决策、协调 Pipeline 和 WSContext | 不持有 IO 资源、不做 ASR/TTS、不管理 WS 连接 |
| `live/audio_pipeline.py` | PCM ring buffer 管理、VAD 能量计算、ASR chunk 调度（run_in_executor）、EndDetector 驱动、WavWriter 流式写入 | 不做状态转换决策、不管理 WS |
| `live/ws_context.py` | WS 连接集合管理、driver/observer 角色分配、JSON/binary 帧收发、断线检测 | 不做 ASR/TTS、不操作 audio buffer |
| `live/asr_engine.py` | 加载/预热 mlx-whisper 或 faster-whisper、PCM -> 文本 | 不做对齐、不做 VAD |
| `live/streaming_engine.py` | 调 `engine.generate_single` -> 切 200ms 帧 -> yield | 不做并发调度（look-ahead 由 session.py 负责） |
| `live/end_detector.py` | 文本归一 + LCS + VAD 累计 -> Trigger | 不读取 PCM（由 session 喂数据） |
| `live/wav_writer.py` | 流式 WAV 落盘 + peak_normalize | 不做采样率转换 |
| `live/audio_resampler.py` | PCM 上下采样（用 scipy.signal.resample_poly） | 不做格式转换 |
| `live/frames.py` | WS 帧 Pydantic 模型定义 | 不做协议逻辑 |

---

## 2. 组件依赖图

```
                    ┌─────────────────────────────────────────────┐
                    │              server.py::create_app          │
                    │                                             │
                    │  ┌──────────┐  ┌────────────┐  ┌─────────┐│
                    │  │  config  │  │   engine   │  │ sample_ ││
                    │  │  (Config)│  │(BaseEngine)│  │ manager ││
                    │  └────┬─────┘  └─────┬──────┘  └────┬────┘│
                    │       │              │              │      │
                    │       ▼              │              │      │
                    │  ┌──────────────────┐│              │      │
                    │  │ podcast_manager  ││              │      │
                    │  └────────┬─────────┘│              │      │
                    │           │          │              │      │
                    │  ┌────────▼──────────▼──────────────▼────┐│
                    │  │       LiveSessionRegistry (单例)       ││
                    │  │  ┌──────────────────────────────────┐ ││
                    │  │  │      EmbeddedASR (共享单例)       │ ││
                    │  │  └──────────────────────────────────┘ ││
                    │  │  ┌──────────────────────────────────┐ ││
                    │  │  │      LiveSession (per session)    │ ││
                    │  │  │  ├─ SessionAudioPipeline            │ ││
                    │  │  │  │   ├─ PCM ring buffer (lock-free) │ ││
                    │  │  │  │   ├─ EndDetector (per live seg)  │ ││
                    │  │  │  │   └─ WavWriter (per live seg)    │ ││
                    │  │  │  ├─ SessionWSContext                 │ ││
                    │  │  │  │   ├─ ws_clients / driver_ws      │ ││
                    │  │  │  │   └─ frame send/recv             │ ││
                    │  │  │  └─ StreamingTTSProxy (1 个)         │ ││
                    │  │  └────────────────────────────┘ ││
                    │  └───────────────────────────────────────┘│
                    └─────────────────────────────────────────────┘
```

**关键依赖关系**：
- `LiveSessionRegistry` 在 `create_app` 内创建，持有 `config`、`engine`、`sample_manager`、`podcast_manager` 的引用
- `EmbeddedASR` 由 registry 持有（**不属于 session**），所有 session 共享（推理串行，复用 Metal/GPU）
- `StreamingTTSProxy` 由 handler 创建并注入到 session（每 session 独立，避免流状态串扰）
- `SessionAudioPipeline` 由 session 持有，封装 PCM buffer / VAD / EndDetector / WavWriter
- `SessionWSContext` 由 session 持有，封装 WS 连接管理和帧收发
- `LiveSession` 本身是纯状态机，通过注入获得 Pipeline 和 WSContext

---

## 3. 数据流（端到端）

### 3.1 AI 段播放流（AI_SPEAKING 状态）

```
LiveSession 调度器
    │
    ▼
StreamingTTSProxy.stream_segment(text, voice)
    │
    ├── 调 engine.generate_single(text, voice) ── 同步阻塞 ── 返回 WAV bytes
    │       └── 实际是 OmlxRemoteEngine._call_remote() 或 LocalVibeVoiceEngine._generate_one()
    │
    ├── sf.info(audio_bytes) 读取采样率
    │
    ├── 跳过 44 字节 WAV header
    │
    ├── 按 chunk_ms=200ms 切片
    │
    └── async yield bytes ──► LiveSession.audio_out_queue (asyncio.Queue)
                                    │
                                    ▼
                              WS handler binary frame ──► 浏览器 AudioContext
```

**关键约束**：
- `engine.generate_single` 是**同步阻塞**调用（remote HTTP timeout=300s 或 local MLX），必须在 `run_in_executor` 中执行
- 切片后每个 chunk `await asyncio.sleep(0)` 让出事件循环，避免阻塞 ASR 路径
- look-ahead：当 cursor 段开始播放时，并行启动 cursor+1 段的 `stream_segment`，结果入 prefetch queue

### 3.2 真人录制流（RECORDING 状态）

```
浏览器 MediaRecorder / ScriptProcessor
    │
    │  binary frame (设备原生采样率 PCM 16-bit)
    ▼
WS handler
    │
    ├── 1. 写入 session.audio_buffer[cursor]（原始采样率，落盘用）
    │
    ├── 2. audio_resampler.downsample(pcm, src_sr, 16000) ──► ASR 输入队列
    │       └─ 单独 asr_pool 线程消费
    │
    └── 3. 计算 dbfs ──► EndDetector.update_vad(dbfs, frame_ms=20)
                                │
                                ▼
                          silence_ms 累计
                                │
        ┌───────────────────────┴────────────────────────┐
        │                                                │
        ▼                                                ▼
  ASR 线程返回文本                              VAD 路径独立触发
        │                                                │
        ▼                                                │
  EndDetector.update_asr(text)                          │
        │                                                │
        ├──► 触发 'end_near' ──► 启动下一段 TTS prefetch  │
        ├──► 触发 'end'      ──► 落盘 + cursor++ + 切段 ◄┘
        └──► 触发 'user_skipped' ──► 强制切段
```

**关键约束**：
- WS 收到 binary 的回调里**不能直接跑 ASR 推理**（会阻塞 WS 读循环）
- PCM 写 buffer、下采样、VAD dbfs 计算都在事件循环 tick 内同步完成（< 1ms）
- ASR 推理通过 `asyncio.run_in_executor(asr_pool, ...)` 异步排队
- VAD 路径**不依赖 ASR**，即便 ASR 故障也能切段（兜底）

### 3.3 多源合并流（FINISHED 后）

```
podcast_manager.merge_project(project_id)
    │
    ▼
遍历 segments[]
    │
    ├── source == "tts"   ──► 读 seg_NNNN.wav
    ├── source == "live"  ──► 读 live_NNNN.wav
    │                          ├── 不存在 ──► _generate_silence(1.0) 占位 + status="missing"
    │                          └── 存在    ──► ffmpeg 下采样到目标采样率
    │
    ▼
_concatenate_audio_segments(audio_parts, ..., base_gap=gap_seconds)
    │
    ▼
{project_id}_merged.wav
```

---

## 4. 状态机

### 4.1 LiveSession 状态转换图

```
                  ┌──────────────────────────────────────────────┐
                  │                                              │
                  ▼                                              │
              ┌───────┐                                          │
              │ IDLE  │                                          │
              └───┬───┘                                          │
                  │ start()                                      │
                  │ 首段 tts: 调度 stream_segment                 │
                  │ 首段 live: 等 driver WS                       │
                  ▼                                              │
        ┌─────────────────┐                                     │
        │  AI_SPEAKING    │◄────────┐                           │
        │  TTS 流式推送    │         │                           │
        └────┬────────┬───┘         │                           │
             │        │             │                           │
   TTS 结束  │        │ 下一段 live  │ 下一段 tts                 │
             ▼        ▼             │                           │
     ┌────────────┐  ┌──────────┐   │                           │
     │WAITING_    │  │RECORDING │   │                           │
     │TRIGGER     │  │ 真人+ASR │   │                           │
     │决策下一段   │──┘  VAD+对齐│   │                           │
     └─────┬──────┘    └────┬────┘   │                           │
           │                │        │                           │
           │ end_trigger    │        │                           │
           └────────────────┘        │                           │
                  │                  │                           │
                  ▼                  │                           │
            ┌─────────────┐          │                           │
            │  FINISHED   │          │                           │
            │  或 ERROR   │          │                           │
            └─────────────┘          │                           │
                                     │                           │
                  ┌──────────────────┘                           │
                  │  driver 断线 30s                              │
                  └──────────────────────────────────────────────┘
                              (回到 WAITING_TRIGGER)
```

### 4.2 状态转换矩阵（合法转换）

| 当前状态 | 触发条件 | 下一状态 | 副作用 |
|---|---|---|---|
| IDLE | `start()` 调用 | AI_SPEAKING 或 WAITING_TRIGGER | 首段 tts: 启动 stream_segment；首段 live: 等 driver |
| AI_SPEAKING | TTS stream 结束 | WAITING_TRIGGER | 推进 cursor；判断下一段 source |
| AI_SPEAKING | driver WS 断开 | WAITING_TRIGGER | 暂停 TTS 推送（保留 prefetch） |
| AI_SPEAKING | 所有段处理完 | FINISHED | 触发 merge（可选） |
| WAITING_TRIGGER | 下一段 tts 就绪 | AI_SPEAKING | 从 prefetch queue 取音频推 WS |
| WAITING_TRIGGER | 下一段 live + driver 在线 | RECORDING | 启动 EndDetector、WavWriter、ASR 订阅 |
| RECORDING | EndDetector `end` / `user_skipped` | WAITING_TRIGGER | 落盘 wav、清 audio_buffer、推进 cursor |
| RECORDING | driver WS 断开 | WAITING_TRIGGER | 暂存 audio_buffer 30s 等重连 |
| 任意 | 不可恢复错误 | ERROR | 落盘已录内容、写 session.json |

非法转换抛 `IllegalStateTransition`。

### 4.3 决策点：WAITING_TRIGGER 的内部逻辑

```
进入 WAITING_TRIGGER:
    cursor += 1
    if cursor >= len(segments):
        state = FINISHED
        return

    next_seg = segments[cursor]
    if next_seg.source == "tts":
        # 从 prefetch queue 取（look-ahead 已生成）或现场启动
        state = AI_SPEAKING
        推送 segment_start 帧
        从 prefetch 取音频流推 WS
    elif next_seg.source == "live":
        if driver_ws is not None:
            state = RECORDING
            推送 segment_start 帧 + state 帧
            启动 EndDetector(target=next_seg.text)
            启动 WavWriter(live_{cursor:04d}.wav)
        else:
            # driver 未连，保持 WAITING_TRIGGER 等待
            推送 "waiting_for_driver" 帧
```

---

## 5. 并发模型

### 5.1 执行上下文分层

```
┌─────────────────────────────────────────────────────────┐
│  主事件循环 (uvicorn asyncio)                           │
│                                                          │
│  ┌─────────────────┐  ┌─────────────────┐              │
│  │ WS 收发协程      │  │ TTS 推送协程     │              │
│  │ (per session)   │  │ (per session)   │              │
│  └────────┬────────┘  └────────┬────────┘              │
│           │                     │                       │
│           │ binary frame        │ yield from            │
│           │                     │ StreamingTTSProxy     │
│           ▼                     ▼                       │
│  ┌──────────────────────────────────────────┐           │
│  │      LiveSession 状态机（同步逻辑）        │           │
│  └──────────────────────────────────────────┘           │
└────────┬─────────────────────────────────┬──────────────┘
         │                                 │
         │ run_in_executor                 │ run_in_executor
         ▼                                 ▼
┌─────────────────────┐         ┌─────────────────────┐
│  tts_pool (2 线程)   │         │  asr_pool (1 线程)   │
│                      │         │                      │
│  engine.generate_    │         │  mlx_whisper.transc │
│  single (同步阻塞)    │         │  ribe (同步阻塞)     │
└─────────────────────┘         └─────────────────────┘
```

**关键约束**：
- `tts_pool` 和 `asr_pool` **物理隔离**，避免 TTS 长 HTTP 调用阻塞 ASR 推理
- `asr_pool` 只 1 个线程：mlx-whisper 推理本身串行（同一 Metal stream 不并发），多线程无收益
- `tts_pool` 2 个线程：当前段 + look-ahead 段可并行生成
- 主事件循环**不阻塞**：所有同步 CPU/GPU 工作都 `run_in_executor`

### 5.2 asyncio.Queue 拓扑

每个 LiveSession 内部维护：

| Queue | 容量 | 生产者 | 消费者 |
|---|---|---|---|
| `audio_out_queue` | 50 chunks (10s@200ms) | StreamingTTSProxy.stream_segment | WS 推送协程 |
| `prefetch_queue` | 1 segment | look-ahead worker | session 调度器（进入 AI_SPEAKING 时取） |
| `asr_in_queue` | 20 chunks (20s@1s) | WS 收帧协程（下采样后） | asr_pool 线程 |

**背压策略**：
- `audio_out_queue` 满 -> StreamingTTSProxy await 阻塞（自然限速）
- `asr_in_queue` 满 -> 丢弃最旧 chunk（ASR 是辅助检测，丢一两个 chunk 不影响）

### 5.3 取消与中止

- `stop_session()` 调用 -> `session.cancel_event.set()` -> 所有 worker 协程检查 cancel_event 退出
- driver WS 断开 -> TTS 推送协程继续完成（已生成的 chunk 推完），但 RECORDING 状态转入 WAITING_TRIGGER
- ASR 推理中的任务不可中断（mlx-whisper 不支持 cancel），等其完成后丢弃结果

---

## 6. 采样率架构

### 6.1 采样率分布

```
                  设备原生采样率           TTS 引擎原生采样率
                  （推荐 48 kHz）         （由 engine 决定，如 24 kHz）
                        │                        │
                        ▼                        ▼
              ┌─────────────────┐      ┌─────────────────┐
              │  WS 上行 PCM     │      │  engine.generate │
              │  (16-bit PCM)   │      │  _single 返回    │
              └────────┬────────┘      └────────┬────────┘
                       │                        │
            ┌──────────┼──────────┐             │
            │          │          │             │
            ▼          ▼          ▼             ▼
       落盘保持    下采样     下采样        WS 下行 PCM
       原采样率    到 16 kHz   到合并       （引擎采样率）
                  for ASR    目标采样率
                              │
                              ▼
                          ┌────────┐
                          │ merge  │
                          │ 统一   │
                          └────────┘
```

### 6.2 采样率协商

- **TTS 侧**：`StreamingTTSProxy` 在首次生成后通过 `sf.info(audio_bytes).samplerate` 读出引擎采样率，缓存为 `engine_sample_rate`，通过 WS `audio_info` 帧告知前端
- **录音侧**：前端 `AudioContext` 用 `stream.getAudioTracks()[0].getSettings().sampleRate` 读设备采样率，通过 WS `client_audio_info` 帧告知服务端
- **合并侧**：`merge_project` 时取 `min(engine_sample_rate, device_sample_rate)` 作为目标采样率，对真人段 ffmpeg 下采样

### 6.3 不写死采样率

代码中**不出现** `24000`、`48000` 等魔法数字（除注释外）。所有采样率从 `sf.info` / `getSettings` / engine 配置动态获取。这与现有 `_concatenate_audio_segments(sample_rate=24000)` 的写死风格不同，新代码必须动态获取。

---

## 7. 持久化与生命周期

### 7.1 文件布局

```
outputs/podcasts/{project_id}/
├── {project_id}.json                 # 项目元数据（PodcastProject 序列化）
├── seg_0000.wav                      # TTS 段音频
├── seg_0001.wav
├── live_0002.wav                     # 真人段音频
├── live_0004.wav
└── live_sessions/
    └── {session_id}.json             # 会话状态快照（5s 周期写）
```

### 7.2 生命周期

| 阶段 | 操作 | 触发 |
|---|---|---|
| 创建 | `POST /api/podcasts/{id}/live/start` -> 创建 session.json | 用户点 Start Live |
| 录制中 | 每 5s 覆盖写 session.json | 后台定时任务 |
| 结束 | `POST .../stop` -> session.json 标 FINISHED + 触发可选 merge | 用户点 Stop |
| 异常 | 服务重启 -> 扫描 live_sessions/*.json -> 注册为 resumable | 启动钩子 |
| 过期 | FINISHED > 1h / orphan > 24h -> 删 JSON + 删 live wav | 后台清理任务 |

---

## 8. 错误恢复策略

### 8.1 单点失败与降级

| 失败点 | 降级行为 | 用户感知 |
|---|---|---|
| ASR 推理异常 | 返回空文本，连续 5 次发 `asr_degraded` 事件 | UI 提示"ASR 暂不可用，仅用静音检测" |
| ASR 模型未加载 | `asr_warming` 事件 + 跳过 ASR 走 VAD-only | UI 显示"ASR 预热中..." |
| TTS engine 超时 | 当前段标 error，跳过到下一段 | UI 显示"TTS 生成失败，已跳过" |
| driver WS 断开 | 暂存 30s，超时 stop | UI 提示"连接断开，尝试重连" |
| WavWriter 写盘失败 | 该段标 missing，merge 时插静音 | merge 警告"段 N 音频缺失" |
| 服务进程崩溃 | 重启后扫描 live_sessions -> 提供 resume 端点 | UI 列出"可恢复会话" |

### 8.2 不变性

- **cursor 永不回退**（除 `redo` 端点显式回退）
- **audio_buffer 在 end 触发时必须落盘**（即便不完整）
- **session.json 每 5s 必写**（即便状态未变，记录 last_save_at 心跳）

---

## 9. 安全与资源限制

| 资源 | 限制 | 实现 |
|---|---|---|
| 单 session 内存 audio_buffer | ≤ 50 MB（约 5 分钟 48k 16-bit mono） | 超限触发 force end |
| prefetch_queue | 1 segment | asyncio.Queue(maxsize=1) |
| audio_out_queue | 50 chunks | asyncio.Queue(maxsize=50) |
| 同时活跃 session | 5 | registry.create() 检查 |
| observer 连接数 | 10 / session | WS accept 时计数 |
| ASR 推理超时 | 5s / chunk | asr_pool 任务 + asyncio.wait_for |
| TTS 生成超时 | 沿用 engine 现有（remote 300s） | 不新增 |

---

## 10. 与现有代码的边界

### 10.1 不修改的现有模块

| 模块 | 理由 |
|---|---|
| `engines/base.py::BaseEngine` | 接口稳定，live 通过 StreamingTTSProxy 包装而非修改 |
| `engines/qwen_remote.py` | 远程 TTS 调用逻辑独立 |
| `engines/local_vibevoice.py` | 本地 MLX 逻辑独立 |
| `sample_manager.py::SampleManager` | 仅复用 `resolve_or_default` 和 `peak_normalize` |
| `segmentation.py` | 文本分段逻辑独立 |

### 10.2 修改的现有模块（最小侵入）

| 模块 | 改动 | 影响范围 |
|---|---|---|
| `config.py` | 追加 `ASRConfig` / `LiveConfig` 类，挂到 `Config` | 向后兼容（默认值） |
| `models.py` | `PodcastSegment.source` 字段 + `status` Literal 扩展 + 新增 Live* 模型 | 旧 JSON `setdefault("source","tts")` 兼容 |
| `podcast_manager.py` | `regenerate_segment` 早返回 + `merge_project` 多源 + `compact_segment_files` | 纯 tts 项目零回归 |
| `server.py` | 在 `create_app` 闭包内追加 live 路由 + WS handler + registry 实例化 | 不动现有路由 |
| `static/index.html` / `app.js` | 追加 live panel UI + `<script src="live.js">` | 不动现有 UI |

### 10.3 复用的现有工具

| 工具 | 位置 | 复用方式 |
|---|---|---|
| `_generate_silence` | `engines/base.py:313` | live 段缺失时插静音 |
| `_concatenate_audio_segments` | `engines/base.py:357` | 多源合并 |
| `peak_normalize` | `sample_manager.py` 内 | 真人段落盘后归一化 |
| `resolve_or_default` | `sample_manager.py:337` | speaker 解析（live 段也走这个，但跳过 TTS） |
| `_parse_tagged_dialogue` | `engines/base.py:26` | 脚本解析（已支持 `Speaker: text`） |

---

## 11. 部署视图

### 11.1 进程模型

```
单个 uvicorn 进程
├── 主事件循环
│   ├── HTTP 路由处理
│   ├── WebSocket 连接处理
│   └── 后台任务（session 清理、session.json 持久化）
├── tts_pool（2 线程）
│   └── engine.generate_single 调用
└── asr_pool（1 线程）
    └── mlx_whisper.transcribe 调用
```

不引入多进程或多容器。ASR 与 TTS 共享同一 Python 进程的 numpy/scipy/soundfile 依赖，避免 IPC 开销。

### 11.2 依赖追加

| 包 | 用途 | 必需性 |
|---|---|---|
| `mlx-whisper>=0.4.0` | Apple Silicon ASR | 二选一 |
| `faster-whisper>=1.0.0` | 跨平台 ASR | 二选一 |
| `zhconv>=1.4.0` | 繁简归一（end_detector 文本归一） | 必需 |
| `scipy` (已存在) | `signal.resample_poly` 下采样 | 已有 |

不引入 `torch`、`torchaudio`、`webrtcvad` 等重依赖。VAD 用能量阈值（numpy 即可）。

---

## 12. 数据流架构（线程模型 + Buffer 边界）

### 12.1 线程与数据流全景

```
[Browser] --WS binary--> [server.py WS handler]
                              |
                    +---------+----------+
                    |                    |
                    v                    v
          SessionAudioPipeline     StreamingTTSProxy
            |    |                      |
            |    | push_pcm()            | stream_segment()
            |    |   +- write buffer     |   +- engine.generate_single
            |    |   +- VAD (dbfs)       |       (tts_pool thread)
            |    |   +- downsample       |
            |    |       to 16k          |
            |    |       |               |
            |    |       v               |
            |    |   asr_in_queue        |
            |    |       |               |
            |    |       v               |
            |    |   ASR (asr_pool       |
            |    |    thread)            |
            |    |       |               |
            |    v       v               |
            |  EndDetector               |
            |    |                       |
            |    v                       |
            |  trigger callback          |
            |    |                       |
            v    v                       v
         LiveSession (state transition) audio_out_queue
                                        |
                                        v
                                   WS binary frames
                                        |
                                        v
                                   [Browser playback]
```

### 12.2 PCM Ring Buffer 策略

`SessionAudioPipeline` 内部使用 `collections.deque(maxlen=N)` 作为 PCM ring buffer：

| 参数 | 值 | 说明 |
|---|---|---|
| 容量 | `audio_buffer_max_mb` (默认 50 MB ≈ 5 min @48k 16-bit mono) | 超限触发 `BUFFER_OVERFLOW` 错误 |
| 写入 | WS handler 收到 binary 帧时同步写入 | 无锁（事件循环单线程写） |
| 读取 | VAD 在同一 tick 内计算 dbfs（< 1ms，不阻塞） | 同步读 |
| ASR 消费 | 下采样后放入 `asr_in_queue`，由 `asr_pool` 线程异步消费 | Queue 解耦 |
| WAV 落盘 | `WavWriter.append()` 在 `end` trigger 触发时批量写入 | 不在每帧写 |

**溢出行为**：当 `audio_buffer` 字节数超过 `audio_buffer_max_mb * 1024 * 1024`，触发 `BUFFER_OVERFLOW` 错误帧，自动 force_end 当前段。

### 12.3 VAD 与 ASR 的关系

VAD 和 ASR **共享同一 PCM 数据源**，但消费路径独立：

```
WS binary frame
    |
    +-> audio_buffer[cursor].extend(pcm)     # for WAV persistence
    |
    +-> VAD: dbfs = calc_rms(pcm) -> dBFS    # sync, < 1ms
    |       +-> EndDetector.update_vad(dbfs)
    |
    +-> ASR: downsample(pcm, src_sr, 16000)  # sync, < 1ms
            +-> asr_in_queue.put(resampled)   # async
                    +-> asr_pool: transcribe_chunk()
                            +-> EndDetector.update_asr(text)
```

- VAD 路径不依赖 ASR，ASR 故障时 VAD 仍然正常工作
- 两者不共享锁：VAD 在事件循环内同步完成，ASR 通过 Queue 解耦到独立线程池

### 12.4 WAV Writer 线程归属

`WavWriter` 在 `end` trigger 时由 `SessionAudioPipeline` 在事件循环内同步调用：

1. `trigger == end / user_skipped` -> `pipeline.flush_wav()` -> `wav_writer.append(buffer)` -> `wav_writer.close()`
2. 全部在主事件循环内完成（`buffer` 是 bytearray，无 IO 等待）
3. `peak_normalize` 在 close 时执行（numpy 操作，< 10ms for 1 min audio）

WAV writer **不在独立线程**：写入量小（一次性 flush buffer）、不阻塞事件循环。

### 12.5 线程模型汇总

| 线程/上下文 | 职责 | 阻塞容忍 |
|---|---|---|
| 主事件循环 | WS 收发、状态转换、JSON 帧处理、VAD 计算、WAV flush | 不允许 > 50ms |
| `tts_pool` (2 线程) | `engine.generate_single` 同步调用 | 允许 30s |
| `asr_pool` (1 线程) | `transcribe_chunk` 同步推理 | 允许 250ms |

---

## 13. LiveSession ↔ PodcastManager 写入策略

### 13.1 两层写入

Live Podcast 采用**实时内存快照 + 最终定稿写回**的两层策略：

| 层级 | 触发时机 | 写入目标 | 实现 |
|---|---|---|---|
| 实时层 | 每 5s 定时 + 状态转换时 | `live_sessions/{session_id}.json` | `LiveSession.persist_snapshot()` |
| 定稿层 | `stop()` / `cancel()` 时 | `{project_id}.json` + 段文件 | 通过 `PodcastManager` 写回 |

### 13.2 实时层（LiveSession 直接操作）

- cursor 推进、state 变更 -> 写入内存中的 LiveSession 对象
- 每 5s 定时序列化 -> `live_sessions/{session_id}.json`（SessionSnapshot）
- `end` trigger 时 -> `WavWriter` 写 `live_NNNN.wav`（LiveSession 通过 Pipeline 操作）
- **不**调用 `PodcastManager._save_project()`（避免频繁 IO）

### 13.3 定稿层（通过 PodcastManager）

`stop()` 执行时：
1. `LiveSession` 遍历已录制段，更新 `PodcastSegment.status` 为 `"generated"`
2. 调用 `podcast_manager._save_project(project)` 写回 `{project_id}.json`
3. 可选触发 `podcast_manager.merge_project(project_id)` 生成合并音频

`cancel()` 执行时：
1. `LiveSession` 状态设为 `ABANDONED`
2. **不**更新 `project.json`（段状态保持原样）
3. **不**触发合并
4. 已录制的 `live_NNNN.wav` 保留在磁盘，由 24h 清理任务处理

### 13.4 一致性保证

- 正常运行中 `project.json` 只在 `stop()` 时更新
- 崩溃恢复时从 `session.json` 重建 LiveSession，`project.json` 不受影响
- `redo()` 操作只删 `live_NNNN.wav` 文件，不改 `project.json`

---

## 14. 架构决策记录（ADR 摘要）

### ADR-1: ASR 内嵌而非 web API

**决策**：ASR 完全在 Python 进程内推理，不通过 HTTP 调用外部 ASR 服务。
**理由**：端到端检测链路（录音->ASR->对齐->切段）任何跨进程跳转引入 50-150ms 抖动，破坏"贴"的听感。内嵌后单 tick 端到端 ≤ 400ms。
**代价**：服务进程内存占用增加（whisper-medium-mlx-4bit 模型），但 Apple Silicon unified memory 可接受。

### ADR-2: 闭包工厂而非 router/blueprint

**决策**：live 路由直接在 `create_app` 闭包内追加，不引入 APIRouter。
**理由**：现有 30+ 路由全部在闭包内，直接访问 `engine`/`sample_manager`/`podcast_manager`/`config`。引入 router 反而需要重新设计依赖注入，破坏一致性。
**代价**：`server.py` 文件会变长（+120 行），但可读性优于混合两种模式。

### ADR-3: StreamingTTSProxy 包装而非修改 BaseEngine

**决策**：新增 `StreamingTTSProxy` 包装现有 `BaseEngine.generate_single`，不修改 BaseEngine 接口。
**理由**：`generate_single` 是同步阻塞返回完整 bytes 的接口，被多处调用。改为流式会破坏所有调用方。包装层在生成后切片模拟流式，接口稳定。
**代价**：首 chunk 延迟 = 完整生成时间（remote ~2-5s），但 look-ahead 缓解。

### ADR-4: VAD 用能量阈值而非 silero-vad

**决策**：VAD 用简单 RMS -> dBFS 能量阈值，不引入 silero-vad。
**理由**：silero-vad 需要额外 torch 依赖（~500MB），而能量阈值在 16-bit PCM 上 < 1ms 计算，对单人语音录制场景足够。
**代价**：噪声环境下可能误判，但 `echoCancellation: true` + `noiseSuppression: true` 已在浏览器侧处理。

### ADR-5: 采样率动态协商而非写死

**决策**：所有采样率从 `sf.info` / `getSettings` 动态获取，不写死 24000/48000。
**理由**：现有 `_concatenate_audio_segments(sample_rate=24000)` 是历史包袱，新代码不应继承。TTS 引擎采样率可能因 model 不同而变，设备采样率因硬件而变。
**代价**：代码稍复杂，但避免未来采样率变更时的隐性 bug。
