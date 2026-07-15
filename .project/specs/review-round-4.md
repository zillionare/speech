# Test Quality Review — Round 4

> 审阅范围：9 个测试文件（e2e × 1 + int × 8）
> 审阅时间：2026-07-15
> 目的：验证 review-round-3 中 77 个问题的修复情况

---

## 1. 总体结论

**大幅改善。** Round-3 的 77 个问题中约 55 个已修复（~71%），核心结构性问题（直接赋值状态、inline 重实现 LCS/transition、API 签名不匹配）在多数文件中已消除。剩余 22 个问题以低严重度为主。

| 类别 | Round-3 | 已修复 | 残留 | 新增 |
|------|---------|--------|------|------|
| A. 自证测试 | 35 | 25 | ~10 | 3 |
| B. API 签名不匹配 | 5 | 5 | 0 | 0 |
| C. 重实现逻辑 | 10 | 7 | 3 | 0 |
| D. 缺失覆盖 | 22 | 10 | 12 | 3 |
| E. 逻辑错误 | 5 | 3 | 2 | 2 |
| **合计** | **77** | **50** | **27** | **8** |

---

## 2. 逐文件状态

### 已修复的文件（质量达标）

#### test_state_machine.py — 全面修复

所有测试通过 `session.transition(Trigger.X)` 驱动真实状态机，transition table 从 production 导入，`LiveSessionRegistry` API 签名正确。`LegalTransitionsImportedTests` 验证了终态无出口、非终态至少有一条转换等结构属性。

残留：D 类——AI_SPEAKING→RECORDING（via START_RECORDING）、DETECTING→AI_SPEAKING（via START_AI）、PAUSED→RECORDING 三条合法转换路径未覆盖。P2。

#### test_end_detector.py — 全面修复

LCS、alignment ratio、normalize_text 全部从 production 导入调用。所有 trigger 测试使用真实 `EndDetector` 实例的 `update_asr()` / `update_vad()` 方法。

残留：
- `test_trigger_after_debounce_window`（~line 136）用 `time.sleep(0.15)` + 手动重置 `det.last_trigger = None` 等私有字段。测试耦合内部实现且 CI 下可能 flaky。P1。
- `update_vad` 自动触发 `on_speech_start()` / `on_speech_end()` 的路径未显式测试。P1。

#### test_ws_protocol.py — 全面修复

所有 frame constructor（`live_state`, `segment_begin`, `asr_partial`, `asr_final`, `alignment_progress`, `error_frame`, `client_audio_info`, `state_ack` 等）调用真实函数，签名匹配。边界值验证正确（`matched > total` 抛 `InvalidFrameError`、无效 source 抛异常）。

残留：
- `BinaryFrameSizeTests`（~line 142）仍是 `struct.pack` 后断言 `len` 的自证测试。P1。
- 多数 constructor 测试未断言 `frame["type"]` 字段值。如果某 constructor 返回错误的 type 字符串，当前测试不会捕获。P1。

#### test_tts_streaming.py — 基本修复

API 签名正确，`FakeEngine` 实现了 `sample_rate` property。`StreamingTTSProxy.stream_segment()` 被调用（catch `NotImplementedError` 后 `skipTest`）。

残留：
- `LiveConfigTests`（~line 116）断言默认值等于默认值（`assertEqual(cfg.tts_max_seconds, 60)`），自证。P1。
- `stream_segment` 所有测试 skip，无真实流式行为验证。P2（取决于实现进度）。

### 基本修复但有残留问题的文件

#### test_live_podcast_flow.py (e2e) — 大部分修复

B/C 类问题消除。测试通过 `registry.transition()` + `Trigger` 枚举驱动。

残留：
- `test_live_only_skips_tts_in_merge`（~line 237）：不调用任何 merge 函数，仅断言 constructor 设置的 source 值。自证。P1。
- `test_complete_session_frame_sequence`（~line 211）：断言自己组装的列表顺序。自证。P1。
- `E2ERedoTests`（~line 177）：`session.cursor = 1` 直接赋值而非调用真实 redo 机制。P1。

#### test_merge.py — 完全重写，核心改善

从"零真实覆盖"重写为调用真实 `PodcastManager.merge_project()` + 真实 WAV fixture。C/D 类问题基本消除。

残留：
- **`test_merge_missing_live_segment_uses_silence`（~line 99）：整个断言块包在 `try: ... except Exception: pass` 中。测试永远不会失败。P0。**
- `test_merge_all_tts_segments`（~line 84）：`assertGreater(duration, 1.5)` 阈值比预期 2.5s 低 40%，太宽松。P2。

#### test_models.py — 部分修复

SegmentSource 枚举、source field、JSON roundtrip 已用真实模型测试。override 优先级新增 4 个测试。

残留：
- `_migrate_segment_source()`（line 23）和 `_auto_mark_source()`（line 28）是 test-local helper，注释说"mirroring production logic"。`MigrationTests` 和 `AutoMarkingTests` 测的是这些 helper，不是 production 代码。production 改了测试不会红。P0。
- `RegenerateGuardTests`（~line 117）：`assertFalse(seg.source == SegmentSource.TTS)` 对 LIVE segment 是 trivially true。自证。P1。

#### test_config.py — 较健康

Config 加载/保存 roundtrip、LiveConfig 默认值和不变量均调用真实函数。

残留：
- `ProjectOverrideTests` 与 `test_models.py` 中同名类几乎相同，代码重复。P2。
- `test_save_preserves_other_fields`（~line 86）：hardcode `assertEqual(..., 30)` 而非引用 `LiveConfig().tts_timeout_seconds`。P2。

### 仍有较多残留的文件

#### test_wav_writer.py — 上半部修复，下半部未动

`WAVHeaderTests` 和 `PeakNormalizeTests` 调用真实函数（round-3 已认可）。

残留 5 个自证测试（全部 P1）：
- `test_live_wav_naming_format`（~line 131）：f-string 格式化后 regex 匹配自己。
- `test_live_wav_stored_in_project_audio_dir`（~line 141）：构建路径字符串后断言子串。
- `test_live_wav_not_auto_deleted`（~line 152）：`auto_delete = False; assertFalse(auto_delete)`。
- `test_live_wav_cleaned_by_24h_task_for_abandoned`（~line 157）：`should_clean = True and (25 >= 24)`。
- `test_live_wav_retained_for_finished_sessions`（~line 163）：`state == FINISHED` 断言 trivially true。

---

## 3. 需要修复的项（按优先级）

### P0 — 1 项

1. **test_merge.py**: `test_merge_missing_live_segment_uses_silence` 的 `try/except pass` 使测试永远通过。移除 try/except，断言 `merge_project` 的预期行为（成功生成含静音占位的 WAV，或抛出特定异常）。

### P1 — 8 项

2. **test_models.py**: `_migrate_segment_source` 和 `_auto_mark_source` helper 应替换为对 production 代码的调用（如 `PodcastManager.get_project()` 触发迁移、`PodcastManager._text_to_segments()` 触发 auto-mark）。
3. **test_models.py**: `RegenerateGuardTests` 应调用真实 regenerate 方法或移除。
4. **test_wav_writer.py**: 5 个自证测试应调用真实命名/路径/cleanup 函数，若无对应函数则删除。
5. **test_live_podcast_flow.py**: 2 个自证测试（merge skip、frame sequence）和 1 个直接 cursor 赋值应修复。
6. **test_ws_protocol.py**: `BinaryFrameSizeTests` 自证 + 缺少 `type` 字段断言。
7. **test_end_detector.py**: debounce 测试耦合内部字段，应通过 `reset()` 或时间注入解耦。
8. **test_tts_streaming.py**: `LiveConfigTests` 自证默认值。
9. **test_config.py + test_models.py**: `ProjectOverrideTests` 重复，应合并到一处。

### P2 — 5 项

10. test_state_machine.py: 补充 3 条未覆盖的合法转换路径。
11. test_merge.py: `assertGreater(duration, 1.5)` 收紧阈值。
12. test_config.py: hardcoded default `30` 改为引用 `LiveConfig`。
13. test_wav_writer.py: `WavStreamWriter` context manager 路径未测。
14. test_ws_protocol.py: `client_audio_info` 缺 channels/bit_depth kwargs 测试。

---

## 4. 结论

与 round-3 相比，测试质量从"几乎全部无效"提升到"大部分有效"。B 类（API 签名）和 C 类（重实现逻辑）问题接近清零，是最关键的改善。剩余问题集中在：test_models.py 的两个 helper（C 类）、test_wav_writer.py 的 5 个自证测试（A 类）、和若干覆盖缺失（D 类）。P0 仅 1 项（test_merge.py 的 try/except pass）。
