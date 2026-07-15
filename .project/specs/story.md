# Live Podcast（真人+AI 混录）— Story 文档

> 采用 **EARS（Easy Approach to Requirements Syntax）** 五种语法记法：
> - **U**biquitous（无前置条件，恒成立）
> - **E**vent-driven（当触发事件…）
> - **S**tate-driven（处于…状态时）
> - **O**ptional（可选功能）
> - **U**nwanted（明确禁止的负面行为）
>
> 每个 Story 有唯一编号 `ST-LP-NNN`。优先级：`P0` 必备 / `P1` 推荐 / `P2` 远期。
>
> 关键架构决策：
> - **ASR 完全内嵌到 Python 服务进程内**，不走外部 web API。
>   端到端检测链路（录音→ASR→对齐→切段）任何额外跨进程跳转都会引入 50-150ms 延迟，破坏"贴"的听感。
> - **音频质量分层原则**：录音端和 TTS 生成端各自保持设备/引擎的最高采样率，不作降质；
>   ASR 路径独立下采样到 16 kHz；最终合并时，统一到两者中较低采样率（下采样，不上采样，因为上采样不产生新信息量）。
>
原始 Story:

当前这个项目已经有了生成播客的能力。它目前做的是，根据 speaker tag，找到对应的声音样本，离线合成声音。

我现在想实现真人与 AI 混录。即指定一个 speaker(比如 Aaron）为真人，其它为 AI，比如 speaker-tag 中，有一个 flora。 当前的段落的speaker 是 flora，于是它读取文本，发声（实时生成并播放）。在她说完之后，紧接着的 speaker 是 Aaron，于是我真人发声，通过录音设备实时收音，系统录制，同时系统通过 ASR 实时识别，进行跟踪。当发现 Aaron 快讲完（根据文本对照），或者已讲完（停顿）时，找到下一个 speaker，用它的声音念对应的文本。

当 AI 念文本时，允许有一定的发挥，但基本忠实于文本内容。

我们该如何设计来实现新版的播客生成？

---

## ST-LP-001 [P0] 配置文件支持 Live Speaker 声明

**U. 应用运行时，系统应** 在 `config.yaml` 中维护一个 `live_speakers` 列表，每个条目是一个 speaker 名称（例如 `Aaron`），表示该 speaker 在 live podcast 中由真人演绎。

**U. 配置加载器应** 在 `voices` 段下接受新增的 `live_speakers` 字段（字符串列表），默认值为空列表。

**E. 当 `config.yaml` 被加载或 `/api/config` 被更新时，系统应** 通过 `save_config_to_yaml()` 把 `live_speakers` 写回磁盘，不丢失其他字段。

**O. 系统可额外** 在 `PodcastProject` 内部支持 `live_speakers_override`，仅在该会话内覆盖全局配置。

---

## ST-LP-002 [P0] PodcastSegment 数据模型扩展 `source` 字段

**U. 对每个 `PodcastSegment`，系统应** 持久化一个 `source` 字段，枚举值之一：
- `"tts"`（默认，向后兼容所有现有项目）— 由 TTS 引擎合成
- `"live"` — 由真人在播客录制时实时录音

**U. 当迁移磁盘上已存在的旧播客项目时，系统应** 把缺失的 `source` 字段默认填充为 `"tts"`，确保历史 JSON 文件继续可用。

**U. 当从脚本创建一个段时，系统应** 若解析得到的 speaker 在全局 `live_speakers` 列表中，则自动把 `source` 设为 `"live"`，否则为 `"tts"`。

**E. 当 `resolve_or_default()` 为一个段解析出 speaker 时，系统应** 同时把解析出的 source 写回该段，避免下游（调度器、合并器）重复查询。

---

## ST-LP-003 [P0] ASR 完全内嵌：进程内推理

**U. 系统应** 把 ASR 模块实现为 `tts_service/live/asr_engine.py`，提供 `class EmbeddedASR`，**不通过网络 HTTP 调用**，直接在同一 Python 进程内完成推理。

**U. `EmbeddedASR` 配置应在 `config.asr` 中包含如下字段：**
- `enabled: bool`（默认 `false`）
- `backend: str`（默认 `"mlx_whisper"`，枚举：`mlx_whisper` / `faster_whisper`）
- `model: str`（默认 `"mlx-community/whisper-small"`）
- `language: str`（默认 `"zh"`）
- `chunk_seconds: float`（默认 1.0，上传分片时长）
- `beam_size: int`（默认 1，单 beam 优先速度）
- `vad_filter: bool`（默认 `true`）
- `compute_type: str`（默认 `"float16"`，仅 `faster_whisper` 用）
- `device: str`（默认 `"auto"`）
- `warmup_on_start: bool`（默认 `false`，可启动后第一次推理才加载模型）

**U. 服务进程启动后第一次调用 ASR 时，系统应** 在后台线程加载模型（`mlx_whisper.load_models` 或 `WhisperModel` 构造），加载期间使用 `asr_warming` 状态提示前端，避免冷启动阻塞。

**U. 应用运行期间，`EmbeddedASR` 应** 把每段 `chunk_seconds` 的 16 kHz mono PCM（由录音 48 kHz 下采样得到）通过 `asr_engine.transcribe_chunk(pcm_bytes) -> ASRResult` 接口处理，返回识别文本与时间戳。下采样在服务端 `LiveSession` 中完成，不增加前端负担。

**U. ASR 推理延迟应** 满足如下要求：
- chunk 大小为 1.0 s 时，单段推理 P95 ≤ 300 ms（mlx-whisper / faster-whisper medium 模型，本地推理）
- VAD 过滤后的音频段不进入 whisper 推理，进一步节省时间

**E. 当用户启动 live session 时 `asr.enabled = false`，系统应** 通过 WS 事件 `asr_unavailable` 告警，并降级为 VAD-only 结束检测（ST-LP-008），但不应拒绝启动。

**U. ASR 引擎应** 实现模型预热（warm-up）：服务首次调用 transcribe 前，先用一个 1 秒的静音样本跑一次完整推理，验证模型加载并预热编译器/Metal 缓存，避免首次直播时延迟尖峰。

**U. ASR 引擎应** 优雅处理推理失败：捕获 `RuntimeError` 等异常后返回空文本，附带 `confidence < 0.3` 标记，避免单次失败中断整条 pipeline。

---

## ST-LP-004 [P0] Live Session 状态机与编排器

**U. 系统应** 定义一个 `LiveSession` 对象，表示一次用户录制完整播客项目的会话。保存在服务器内存中，以 `session_id` 为键，对外暴露以下状态：
- `IDLE`（已创建，尚未开始）
- `AI_SPEAKING`（正在播放 AI 合成的段）
- `RECORDING`（真人段录制中，含 ASR 实时转写）
- `WAITING_TRIGGER`（上一段刚结束，等待调度决策）
- `FINISHED`（所有段处理完毕）
- `ERROR`（不可恢复错误，仅当重试耗尽）

**U. 当 `LiveSession` 处于活动状态，系统应** 维护一个 `cursor` 指向下一个要执行的 `PodcastSegment.index`，单调前进，禁止回退。

**E. 当调用 `POST /api/podcasts/{id}/live/start` 时，系统应** 创建一个 `LiveSession`，快照项目段列表，若首段为 `tts` 则后台启动其生成，状态转入 `AI_SPEAKING`，返回 `session_id`。

**E. 当调用 `POST /api/podcasts/{id}/live/{session_id}/stop` 时，系统应** 把会话转到 `FINISHED`，定稿合并的音频文件（参见 ST-LP-009），释放内存资源。

**U. 当会话处于 `FINISHED` 或 `ERROR` 状态，系统应** 拒绝所有对该 `session_id` 的后续控制命令，返回 HTTP 409。

---

## ST-LP-005 [P0] 流式 TTS 输出 + 音频队列

**U. 当会话处于 `AI_SPEAKING` 状态，系统应** 把生成的音频以小 PCM/WAV 切片（~200 ms）流入 `asyncio.Queue`，由 WebSocket sender（ST-LP-006）消费，避免整段音频生成完成才开始播放的等待。

**E. 当 TTS 引擎只支持一次性返回（现有的 `OmlxRemoteEngine` 与 `LocalVibeVoiceEngine`），流式代理应** 在生成完成后切分成固定时长切片，再逐渐 yield，最多叠加 `chunk_seconds` 延迟。

**E. 当下一段是 `tts` 且当前段是 `live`（或接近结束）时，会话调度器应** 立即把下一段生成派发给 worker 池，等到 ST-LP-008 触发 end 时队列里已经有缓冲好的音频。

**U. TTS 调度器应** 实现 1 段 look-ahead 缓冲，超过 next-next 段的音频自动丢弃，控制内存上限。

---

## ST-LP-006 [P0] WebSocket 双向通道 `/ws/podcasts/{id}/live/{session_id}`

**U. WebSocket 连接应** 在同一 socket 上同时承载：
- 一个方向：JSON 控制帧
- 另一个方向：二进制音频帧

**服务端 → 客户端二进制帧：** TTS 引擎原生采样率（由引擎实际输出决定），16 位 PCM，mono，little-endian。连接建立时通过 JSON 帧 `{"type": "audio_info", "sample_rate": <engine_rate>, "channels": 1, "bit_depth": 16}` 协商参数，其中 `<engine_rate>` 从 `BaseEngine` 实例动态获取。

**客户端 → 服务端二进制帧：** 录音设备原生采样率（推荐 48 kHz，`{ ideal: 48000 }`），16 位 PCM，mono，little-endian。服务端在其内部下采样到 16 kHz 供 ASR 使用，同时保留原始 48 kHz 数据用于 WAV 落盘。

**服务端 → 客户端帧：**
- `{"type": "state", "state": "AI_SPEAKING|RECORDING|..."}` — 状态变更
- `{"type": "segment_start", "index": N, "source": "tts|live", "speaker": "...", "text": "..."}`
- `{"type": "asr_partial", "text": "...", "audio_ms": N}` — ASR 增量结果
- `{"type": "asr_final", "text": "...", "matched_ratio": 0.95}` — 一段 ASR 终结
- `{"type": "alignment", "matched_chars": 42, "total_chars": 48}` — 文本对齐进度
- `{"type": "asr_warming", "progress": 0.6}` — 模型加载进度
- `{"type": "error", "code": "...", "message": "..."}`

**客户端 → 服务端 JSON 帧：**
- `{"type": "client_log", "level": "info", "msg": "..."}`（调试可选）

**U. 连接建立期间，服务端应** 在 TTS chunk 就绪 500 ms 内把音频推到客户端。

**E. 当客户端意外断开，服务端应** 暂停会话到可恢复状态（cursor 不变，进入 `WAITING_TRIGGER`）最多 30 s，超时自动 stop。

---

## ST-LP-007 [P0] 前端录音：MediaRecorder 实时上传

**U. 前端应在** 进入 `RECORDING` 状态时调用 `getUserMedia({ audio: { channelCount: 1, sampleRate: { ideal: 48000 } } })`，以设备支持的最高采样率采集音频。实际采样率通过 WebSocket 的 `audio_info` 帧告知服务端。

**E. 当处于 `RECORDING` 状态，客户端应** 以 1 秒 timeslice 捕获 PCM 数据，通过 `AudioContext` 以设备原生采样率将 `Float32Array` 转为 `Int16Array`，并以二进制帧发到 WebSocket。服务端接收后做两件事：① 保留原始采样率 PCM 用于 WAV 落盘；② 下采样到 16 kHz 喂给 ASR。

**U. 当处于 `IDLE` / `AI_SPEAKING` 状态，前端应** 静音本地麦克风（`track.enabled = false`），防止 VAD 误触发。

**E. 当用户点击 "Hold Space to Talk" 或 `RECORDING` 开始时，前端应** 同时通过扬声器播一段 250 ms 的 800 Hz beep 作为"你正在直播"的清晰提示音，并给下游回声消除留出收敛时间。

---

## ST-LP-008 [P0] 文本对齐 + VAD 双触发讲完检测

**U. 当处于 `RECORDING` 状态，检测器应** 对每次 ASR 更新计算两个分数：
- `alignment_score` = `len(longest_common_substring(asr_text, target_text)) / len(target_text)`
- `silence_ms` = 持续低能量帧累计毫秒（VAD，能量阈值通过 `silence_db_threshold` 配置，默认 -45 dBFS）

**触发规则：**
- **End-near 触发（前置预生成）：** E. 当 `alignment_score >= 0.85`，系统应** 触发 `end_near` 事件，若下一段尚未缓冲则启动预取/预生成。
- **End 触发（真正切换）：** E. 当 (`alignment_score >= 0.98` 且 `silence_ms >= 300`) 或 (`silence_ms >= 800`)，系统应** 触发 `end`，定稿所录 WAV（参见 ST-LP-010），cursor 推进，派发下一段。
- **Stuck 触发（兜底）：** E. 当 `silence_ms >= 4000`，系统应** 强制 `end`（即便对齐分数低），视作用户放弃当前段，并发 `user_skipped_segment` 事件。

**U. 检测器应** 对 `end` 触发做 200 ms 去抖，避免瞬态触发的重复触发。

---

## ST-LP-009 [P0] 多源（tts / live）音频段顺序合并

**U. 当会话结束（或按需通过 `POST /api/podcasts/{id}/merge`）执行合并步骤，系统应** 按原始顺序遍历段，每个段向 concat 列表写入：
- 若 `source == "tts"` 且 `status == "generated"`：`seg_NNNN.wav`
- 若 `source == "live"`：`live_NNNN.wav`（由 ST-LP-010 落盘）

**U. 合并器应** 与现有纯 TTS 合并路径（`engines/base.py:_concatenate_audio_segments`）一致地遵守每段的 `pre_pause`、`post_pause` 和项目级 `gap_seconds`。

**E. 当一个 live 段在合并时 WAV 文件缺失（例如会话中途中止），系统应** 用 1 秒静音片段占位，把段 `status` 标为 `"missing"`，继续合并。

**U. 合并应** 输出一个 `{project_id}_merged.wav`，采样率统一到 TTS 引擎原生采样率与真人录音采样率中**较低者**。真人段在合并前通过 ffmpeg 下采样到该目标采样率，TTS 段保持原生采样率不变。上采样不产生新信息量，因此不做升频。总时长等于段时长之和加上段间隔，且不论段来源，TTS 与真人声音的响度配置保持一致。

---

## ST-LP-010 [P0] 真人段 WAV 落盘与持久化

**U. 当处于 `RECORDING` 状态，服务端应** 不停地把到达的 PCM 字节（设备原生采样率，推荐 48 kHz，mono）追加到该会话的内存缓冲中，并在 `end` 触发时把这些缓冲刷新写入 `outputs/podcasts/{project_id}/live_{index:04d}.wav`，文件头为标准 16 位 PCM WAV header，采样率与录制时一致。

**E. 当一段因为任一触发（对齐 / 静音 / 用户中止 / 断线）结束时，系统应** 立即写入已捕获的音频，确保局部片段被保留。

**U. 录制的 WAV 应** 通过 `sample_manager` 中已有的 `peak_normalize` 辅助函数将峰值归一化到 0.95，使真人录制段在与 TTS 段合混时听感音量一致。

**U. 系统应** 永不自动删除 `live_NNNN.wav`；手动清理需通过现有的项目级 DELETE 端点（将在 ST-LP-014 新增）。

---

## ST-LP-011 [P0] 前端实时播放器：排队 PCM 帧

**U. 当 WebSocket 二进制帧到达，前端应** 以 TTS 引擎原生采样率（由 `audio_info` 帧协商）初始化 `AudioContext`，将帧送入 `AudioBufferSourceNode` 链，播放延迟控制在 300 ms 以内。

**U. 播放器应** 使用小型 jitter buffer（目标 600 ms，最大 1500 ms）；只有当缓冲超过最大值才丢帧，并发出 `underrun` 遥测事件。

**E. 当处于 `RECORDING` 状态，本地播放链应** 对 AI 段静音输出，以防扬声器—麦克风开环反馈；只支持通过耳机本地监听（兼容浏览器 autoplay policy）。

---

## ST-LP-012 [P1] ASR 中间结果展示与校对

**O. 当处于 `RECORDING` 状态，前端可** 渲染一个双行面板：
- 上方：实时 ASR partial 文本（每次 ASR tick 更新）
- 下方：目标脚本文本，匹配的字符高亮绿色，未匹配灰色

**O. 系统可额外** 允许用户点击"Adjust alignment"手动标记某目标字符为已念（处理 ASR 乱码但实际上人正确念出的情况）。

---

## ST-LP-013 [P1] 重录单段：仅替换真人段

**E. 当调用 `POST /api/podcasts/{id}/segments/{i}/regenerate` 且 `source == "live"` 时，服务端应** 返回 HTTP 409 和提示信息——live 段只能通过再次进入 `RECORDING` 重录，不能走 TTS 批量再生路径。

**O. 当调用 `POST /api/podcasts/{id}/live/{session_id}/redo/{index}`（新端点）时，系统可** 把 cursor 退到该 index，删除先前录制的 `live_{index:04d}.wav`，下次 `RECORDING` 时覆盖该段。

---

## ST-LP-014 [P1] Live 段删除与项目再平衡

**E. 当调用 `DELETE /api/podcasts/{id}/segments/{i}` 时，系统应** 从项目中删除该段，删除其 `seg_NNNN.wav` 或 `live_NNNN.wav`，对后续段文件做紧凑重编号，确保合并输出不留空隙。

**U. 紧凑过程中，系统应** 保持合并结果一致：文件名按段编号同步重排，但播放顺序保留不变。

---

## ST-LP-015 [P1] 配置：Live-only 录制模式（不跑 TTS）

**O. 在 Live-only 模式下，会话可** 跳过所有 `source == "tts"` 段，可用于"shadow narration"工作流（人做全部配音）。

**U. 该模式应** 由 `/api/podcasts/{id}/live/start` 请求中的 `live_only: bool` 字段控制。开启时，TTS 段在合并时被替换为 1 秒静音占位。

---

## ST-LP-016 [P1] 错误恢复与可恢复中断

**U. 系统应** 把 `LiveSession` 状态（cursor、各段 source、已捕获音频偏移）每 5 秒持久化到 `outputs/podcasts/{project_id}/live_sessions/{session_id}.json`。

**E. 当服务端在会话进行中重启，系统应** 在下次启动时扫描到孤儿 session 文件，并通过 `POST /api/podcasts/{id}/live/{session_id}/resume` 端点允许用户从上一个保存的 cursor 继续会话，已捕获的 `live_NNNN.wav` 文件保留。

---

## ST-LP-017 [P2] 多客户端同步观看模式

**O. 当 live 会话活动时，多个浏览器客户端可** 以 `observe` 角色（通过 `?role=observe` 查询参数）连接到同一 `session_id`，它们应** 收到同样的 WS 广播，但不允许发送音频帧。

**U. 服务端应** 强制每会话角色归属：第一个连接的客户端成为 `driver`，其余默认 `observe`，除非通过显式的 claim 帧升级。

---

## ST-LP-018 [P2] 实时 AI 段落续接

**O. 当 AI 段播放结束但真人还没开始说话，系统可** 注入一段 1.5 秒的"思考停顿"音频，然后用稍高 temperature 重生成该段以增加自然变化。

**U. 重生成应** 限定每 `tts` 段最多 2 次；都失败就回退到第一次的结果原样。

---

## ST-LP-019 [P1] TTS 段超时与降级策略

**E. 当 TTS 引擎生成单段音频超过 `tts_timeout_seconds`（默认 30 s）时，系统应** 跳过该段，用静音占位，通过 WS 发送 `error` 帧（code=`TTS_TIMEOUT`），并继续处理下一段，不阻塞整个会话。

**U. 当单段文本估算时长超过 `tts_max_seconds`（默认 60 s）时，系统应** 按句号自动切分为多个子段，每个子段独立生成并按序播放，缩短首段生成延迟。

**E. 当 TTS 生成失败（非超时）时，系统应** 重试 1 次（无退避），仍失败则跳过该段，发送 `error` 帧（code=`TTS_ERROR`）。

**U. 系统应** 在 `config.live` 中提供 `tts_max_seconds` 和 `tts_timeout_seconds` 配置项，允许用户根据实际 TTS 引擎性能调整。

---

## Unwanted Behaviors（明确禁止）

- **UW-LP-1. 当会话处于 `FINISHED` 或 `ERROR`，服务端不应** 接受任何后续命令。
- **UW-LP-2. 当 `AI_SPEAKING` 时，服务端不应** 在客户端请求麦克风权限（OS 层麦克风应静音）。
- **UW-LP-3. 当录制中，系统不应** 把原始 WAV 字节回传给客户端；只暴露 `asr_partial` / `asr_final` 文本帧。
- **UW-LP-4. 当把一个 speaker 配置为 live，系统不应** 让同一 speaker 名同时映射到 `voices/{name}.wav` 合成引用——live 标志胜出（对该 speaker 忽略 TTS reference）。
- **UW-LP-5. 系统不应** 在项目没有任何 `source == "live"` 段时启动 live session。
- **UW-LP-6. 当流式推音频给客户端，服务端不应** 在单个 chunk 上阻塞事件循环超过 50 ms（必须 yield）。
- **UW-LP-7. 系统不应** 在用户没有显式合并前保留录制的音频；孤立文件在会话放弃 24 小时后被删除。
- **UW-LP-8. ASR 推理不应** 阻塞 AI 音频播放路径——必须跑在独立线程/事件循环，避免拖慢 TTS chunk 推送。
- **UW-LP-9. 当 TTS 生成超时，系统不应** 阻塞整个会话——必须跳过并继续，不得无限等待。

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
