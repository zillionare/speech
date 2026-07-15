# Live Podcast — Acceptance Criteria（验收标准）

> 目标：为每个功能点建立可追踪的验收标准编号（ACC-NNN-MMM），测试代码在注释中引用 ACC 编号。
> 映射关系：ACC-NNN-MMM ↔ SPEC-NNN ↔ ST-LP-NNN。

---

## 1. 配置与数据模型

### ACC-001-1 全局 Live Speaker 配置
**对应**：SPEC-001 / ST-LP-001
**标准**：`Config.voices.live_speakers` 可读写，`save_config_to_yaml` 后 YAML 保留该字段。

### ACC-001-2 每项目 Live Speaker 覆盖
**对应**：SPEC-001 / ST-LP-001
**标准**：`PodcastProject.live_speakers_override` 存在，解析 speaker source 时优先于全局配置。

### ACC-002-1 Segment Source 字段
**对应**：SPEC-002 / ST-LP-002
**标准**：`PodcastSegment.source` 为 `tts` 或 `live`，默认 `tts`；旧项目 JSON 无该字段时加载为 `tts`。

### ACC-002-2 自动标记 Live Segment
**对应**：SPEC-002 / ST-LP-002
**标准**：从脚本创建 segment 时，speaker 在 live_speakers 列表中的段自动标记为 `live`。

### ACC-002-3 Live 段跳过 TTS
**对应**：SPEC-002 / ST-LP-002
**标准**：`regenerate_segment` 遇到 `source == live` 时不调用 engine，状态保持 `pending`。

---

## 2. ASR 引擎

### ACC-003-1 ASR 配置挂载
**对应**：SPEC-003 / ST-LP-003
**标准**：`Config.asr` 为 `ASRConfig`，包含 backend / model / language 等字段。

### ACC-003-2 ASR 预热接口
**对应**：SPEC-003 / ST-LP-003
**标准**：`EmbeddedASR.warmup()` 可被调用，`is_warming` 在预热期间返回 True。

### ACC-003-3 ASR 推理接口
**对应**：SPEC-003 / ST-LP-003
**标准**：`EmbeddedASR.transcribe_chunk(pcm_bytes)` 返回 `ASRResult`，包含 text / confidence / audio_ms。

### ACC-003-4 ASR 失败降级
**对应**：SPEC-003 / ST-LP-003
**标准**：ASR 推理异常时返回空文本且 confidence=0，连续失败触发 degraded 状态。

---

## 3. Live Session 状态机

### ACC-004-1 状态枚举与转换
**对应**：SPEC-004 / ST-LP-004
**标准**：`LiveState` 包含 IDLE / AI_SPEAKING / RECORDING / DETECTING / PAUSED / FINISHED / ABANDONED / ERROR；非法转换抛 `IllegalStateTransition`。

### ACC-004-2 Session 注册表
**对应**：SPEC-004 / ST-LP-004
**标准**：`LiveSessionRegistry` 可创建、获取、停止、取消 session；终端状态不再接受命令。

### ACC-004-3 Session 生命周期端点
**对应**：SPEC-004 / ST-LP-004 / IF-001
**标准**：HTTP `POST /api/podcasts/{id}/live/start|stop|resume` 行为符合接口契约。

---

## 4. 流式 TTS

### ACC-005-1 流式切片
**对应**：SPEC-005 / ST-LP-005
**标准**：`StreamingTTSProxy.stream_segment(text, voice)` 以 ~200ms 切片 async yield PCM bytes。

### ACC-005-2 Look-ahead 缓冲
**对应**：SPEC-005 / ST-LP-005
**标准**：Session 调度器在当前段播放时预生成下一段 TTS。

---

## 5. WebSocket 协议

### ACC-006-1 WS 端点与角色
**对应**：SPEC-006 / ST-LP-006 / IF-006
**标准**：`WS /ws/podcasts/{id}/live/{sid}?role=driver|observe` 可连接；driver 唯一，observer 只收不发。

### ACC-006-2 帧类型完整
**对应**：SPEC-006 / ST-LP-006 / IF-006
**标准**：server/client 帧构造函数返回正确的 `type` 字段，边界值校验通过。

### ACC-006-3 音频流传输
**对应**：SPEC-006 / ST-LP-006
**标准**：服务端能向客户端发送二进制 PCM 帧，客户端能向服务端发送二进制 PCM 帧。

---

## 6. 前端录音

### ACC-007-1 MediaRecorder 采集
**对应**：SPEC-007 / ST-LP-007
**标准**：`live.js` 能在 RECORDING 状态时打开麦克风并以 1s 切片发送 PCM。

### ACC-007-2 状态跟随
**对应**：SPEC-007 / ST-LP-007
**标准**：非 RECORDING 状态时麦克风 track.enabled=false，避免误触发 VAD。

---

## 7. 讲完检测

### ACC-008-1 文本对齐
**对应**：SPEC-008 / ST-LP-008
**标准**：`EndDetector` 用 LCS 计算对齐率，触发 `end_near` / `end` / `user_skipped`。

### ACC-008-2 VAD 独立触发
**对应**：SPEC-008 / ST-LP-008
**标准**：VAD 通过 dbfs 累计 silence_ms，不依赖 ASR 也可触发 end。

### ACC-008-3 去抖
**对应**：SPEC-008 / ST-LP-008
**标准**：同一 trigger 在 debounce 窗口内不重复触发。

---

## 8. 多源合并

### ACC-009-1 TTS + Live 顺序合并
**对应**：SPEC-009 / ST-LP-009
**标准**：`PodcastManager.merge_project` 按 source 选择 `seg_NNNN.wav` 或 `live_NNNN.wav`。

### ACC-009-2 缺失 Live 段兜底
**对应**：SPEC-009 / ST-LP-009
**标准**：live 段 WAV 缺失时插入 1s 静音，状态标为 `missing`，merge 继续。

---

## 9. 真人段 WAV 落盘

### ACC-010-1 流式 WAV 写入
**对应**：SPEC-010 / ST-LP-010
**标准**：`WavWriter` 以 append 模式写入 `live_{index:04d}.wav`，文件头正确。

### ACC-010-2 峰值归一化
**对应**：SPEC-010 / ST-LP-010
**标准**：落盘后 peak_normalize 到 0.95。

---

## 10. 前端实时播放

### ACC-011-1 AudioContext 播放链
**对应**：SPEC-011 / ST-LP-011
**标准**：`live.js` 用 `AudioBufferSourceNode` 排队播放服务端推送的 PCM，控制延迟。

---

## 11. 重录与删除

### ACC-013-1 Live 段重录
**对应**：SPEC-013 / ST-LP-013
**标准**：`POST /api/podcasts/{id}/live/{sid}/redo/{index}` 仅允许 live 段，回退 cursor 并删除旧 WAV。

### ACC-014-1 Live 段删除与重编号
**对应**：SPEC-014 / ST-LP-014
**标准**：`DELETE /api/podcasts/{id}/segments/{i}` 删除 live 段文件并紧凑重编号。

---

## 12. Live-only 模式

### ACC-015-1 跳过 TTS 段
**对应**：SPEC-015 / ST-LP-015
**标准**：`live_only=true` 启动时，tts 段被跳过，merge 时以静音占位。

---

## 13. 错误恢复

### ACC-016-1 Session 状态持久化
**对应**：SPEC-016 / ST-LP-016
**标准**：session 每 5s 写 `live_sessions/{sid}.json`，服务重启后可 resume。

---

## 14. 多客户端观察

### ACC-017-1 Observer 模式
**对应**：SPEC-017 / ST-LP-017
**标准**：observer 角色只接收广播，不能发送音频帧；driver 断线后可 claim。

---

## 15. AI 段落续接

### ACC-018-1 停顿续接
**对应**：SPEC-018 / ST-LP-018
**标准**：AI_SPEAKING ≥ 8s 且真人未开始时，可重生成当前段，最多 2 次。

---

## 未 Wanted 行为

### ACC-UW-1 终端状态拒绝命令
**对应**：UW-LP-1 / SPEC-004
**标准**：FINISHED / ABANDONED / ERROR 状态下控制命令返回 409。

### ACC-UW-2 AI 段不收音
**对应**：UW-LP-2 / SPEC-006
**标准**：AI_SPEAKING 时客户端麦克风静音。

### ACC-UW-3 不泄漏 raw WAV
**对应**：UW-LP-3 / SPEC-006
**标准**：录制期间服务端不向客户端发送原始 WAV bytes。

### ACC-UW-4 Live Speaker 不合成
**对应**：UW-LP-4 / SPEC-002
**标准**：live speaker 忽略其 voices/ 下的 reference，不用于 TTS。

### ACC-UW-5 无 Live 段拒启动
**对应**：UW-LP-5 / SPEC-004
**标准**：项目无 live 段时 `start` 返回 409。

### ACC-UW-6 不阻塞事件循环
**对应**：UW-LP-6 / SPEC-005
**标准**：每个 chunk 处理让出事件循环，单 tick < 50ms。

### ACC-UW-7 孤儿文件清理
**对应**：UW-LP-7 / SPEC-016
**标准**：abandoned session 超过 24h 删除其 live wav 与 session.json。
