# Test Quality Review — Round 3

> 审阅范围：tests/e2e/ (1 file) + tests/int/ (8 files), 共 9 文件 ~2200 行
> 审阅时间：2026-07-15
> 背景：Round-2 之后另一个 AI 修改了源代码（LiveConfig、SegmentSource、sample_rate、peak_normalize 等）并调整了测试，本文是对修改后版本的二次审阅。

---

## 1. 总体结论

源代码层面的改善使 API 签名（B 类）的修复路径更清晰，但**测试代码本身的 5 类问题几乎没有修正**。核心问题依然是：绝大多数测试不调用真实业务代码，只验证 Python 属性赋值和 inline 算术。

| 类别 | P0 | P1 | P2 | 合计 |
|------|----|----|-----|------|
| A. 自证测试 | 19 | 14 | 2 | 35 |
| B. API 签名不匹配 | 2 | 2 | 1 | 5 |
| C. 重实现逻辑 | 6 | 3 | 1 | 10 |
| D. 缺失覆盖 | 4 | 14 | 4 | 22 |
| E. 逻辑错误 | 1 | 3 | 1 | 5 |
| **合计** | **32** | **36** | **9** | **77** |

---

## 2. 逐文件审阅

### 2.1 tests/e2e/test_live_podcast_flow.py

**问题数：27 (P0: 14, P1: 12, P2: 1)**

#### A. 自证测试 (8 个, 全部 P0)

所有 happy path 测试用 `session.state = X; assertEqual(session.state, X)` 驱动，从未调用 `transition()`。

```python
# 现状 (Line 82-84)
session.state = LiveState.AI_SPEAKING
self.assertEqual(session.state, LiveState.AI_SPEAKING)

# 应改为
registry.transition("ai_finished")  # 或其他 trigger
self.assertEqual(session.state, LiveState.AI_SPEAKING)
```

受影响测试：`test_full_setup_flow`, `test_ai_speaking_to_recording_flow`, `test_record_and_merge_flow`, `test_disconnect_and_resume`, `test_cancel_mid_session_abandons`, `test_full_multi_speaker_podcast`。

#### B. API 签名不匹配 (2 个)

- **B1 [P1]**: 所有测试调用 `registry.create(project_id, segments, LiveState.IDLE)`。真实 API 是 `registry.start_live_session(project_id, wav_path, sample_rate, asr_engine=None)`。
- **B2 [P2]**: 从不调用 `stop_live_session()` 或 `registry.get()`。

#### C. 重实现逻辑 (4 个)

- **C1 [P0]** Line 111-116: ASR 对齐用 `len(partial)/len(target_text)` 而非 `EndDetector.update_asr()`。真实代码用 LCS，不是长度比。
- **C2 [P0]** Line 309-325: 同上，`len(normalize_text(asr_text))/len(normalized_target)` 代替 LCS。
- **C3 [P1]** Line 147-155: 音频合并用 raw bytes 拼接，未调用任何 merge/peak_normalize 函数。
- **C4 [P1]** Line 183-190: live-only 过滤用 inline for-loop 实现。

#### D. 缺失覆盖 (10 个, 3 个 P0)

P0: EndDetector 从未实例化、cancel 端点从未调用、`transition()` 合法性从未验证。
P1: `client_audio_info`、`state_ack`、`segment_audio_begin` WS frame 无测试；`SessionAudioPipeline`、`SessionWSContext`、`LiveConfig`、`peak_normalize` 无测试。

#### E. 逻辑错误 (3 个)

- **E1 [P0]** Line 309-315: ASR ratio 公式用总字符数比而非 LCS 匹配字符数。完全无关但等长的文本会给出 ratio=1.0（误触发结束）。
- **E2 [P1]** Line 171: `should_merge = session.state != LiveState.ABANDONED` 是对手动设的字段的 tautological 布尔表达式。
- **E3 [P1]** Line 209-220: disconnect/resume 测试保存再恢复 `session.state`，从未调用 `transition()`，cursor 断言是 vacuously true。

---

### 2.2 tests/int/test_state_machine.py

**问题数：16 (P0: 8, P1: 5, P2: 3)**

#### A. 自证测试 (9 个, 6 个 P0)

- `test_start_with_tts_first_segment_enters_ai_speaking` (~line 185): `session.state = LiveState.AI_SPEAKING` 后断言。
- `test_start_with_live_first_segment_enters_recording` (~line 200): 同上。
- `test_stop_transitions_to_finished` (~line 215): 手动设 `FINISHED` + `finished_at`，从不调用 `stop_live_session()`。
- `test_cancel_transitions_to_abandoned` (~line 232): 手动设 `ABANDONED`。
- `test_finished_or_error_rejects_commands_409` (~line 247): 测试 `state in terminal_states` — 纯 Python 集合成员检查。
- `test_all_illegal_transitions_raise` (~line 100): 名字说"raise"但从未调用 `transition()`，只比对自己 hardcode 的 dict。
- `test_cursor_advances_monotonically` (~line 138): 手动设 cursor 值再读回。

#### B. API 签名不匹配 (2 个, 全部 P0)

- **B1**: `registry.create(project_id, segments, start_state)` 和 `registry.get(project_id, session_id)` 和 `registry.stop(project_id, session_id)` — 真实 API 是 `start_live_session()` / `stop_live_session()` / `transition()`。
- **B2**: 从未用 trigger 字符串调用 `transition()`（真实 API：`registry.transition("ai_finished")` 等）。

#### C. 重实现逻辑 (2 个, 全部 P0)

- **C1**: `_LEGAL_TRANSITIONS` 状态转换表在 `LegalStateTransitionsTests` 和 `IllegalStateTransitionTests` 中各 hardcode 一次，从未从 `session.py` 导入。
- **C2**: start/stop/cancel 逻辑用 inline if-statement 和直接属性赋值重实现。

#### D. 缺失覆盖 (5 个, 1 个 P0)

P0: `LiveSession.transition()` 从未被调用。
P1: `ABANDONED` 终态行为、`cancel` trigger、`client_disconnect`/`client_reconnect` triggers、`force_stop` trigger 均无测试。

#### E. 逻辑错误 (1 个)

- **E1 [P1]**: `test_cancel_transitions_to_abandoned` 将 "cancel"（DETECTING→RECORDING 循环 trigger）与 session abandonment 混淆。

---

### 2.3 tests/int/test_tts_streaming.py

**问题数：15 (P0: 6, P1: 6, P2: 3)**

#### A. 自证测试 (9 个, 6 个 P0)

15 个测试中 9 个是纯算术：

```python
# test_full_audio_sliced_into_chunks (~line 40)
chunk_count = int(24000 * 2.0) // int(24000 * 0.2)
self.assertEqual(chunk_count, 10)  # 测的是乘法
```

- `test_chunk_size_matches_spec` (~line 47): 同一表达式算两次比相等。
- `test_last_chunk_may_be_smaller` (~line 53): `5000 // 4800 == 1`。
- `test_wav_header_skipped_before_slicing` (~line 61): `44 + x - 44 == x`。
- `test_look_ahead_prefetches_next_segment` (~line 71): `2 + 1 == 3`。
- `test_long_segment_auto_split_by_period` (~line 128): inline 重实现分割算法。
- `test_timeout_error_frame` (~line 143): 构建 dict 再断言 dict 值。
- `test_tts_error_retry_once_then_skip` (~line 148): while 循环计数测试。

#### C. 重实现逻辑 (2 个, 全部 P0)

- **C1**: chunk slicing 数学完全 inline，`StreamingTTSProxy.stream_segment` 从未被调用。
- **C2**: long-segment splitting 算法 inline 重实现。

#### D. 缺失覆盖 (4 个)

P0: `StreamingTTSProxy.stream_segment` 从未用真实音频调用。
P1: `SessionAudioPipeline` 无测试；TTS→WavStreamWriter 数据路径无测试。

---

### 2.4 tests/int/test_end_detector.py

**问题数：12 (P0: 4, P1: 7, P2: 1)**

#### A. 自证测试 (3 个, 1 个 P0)

- `test_end_detector_state_transitions` (~line 165): 手动设状态再断言。

#### C. 重实现逻辑 (4 个, 全部 P0)

- **C1** `LCSAlignmentTests`: 完整 LCS 算法在测试中重写，`EndDetector._longest_common_subsequence` 从未调用。
- **C2** `TriggerRuleTests`: trigger 状态机在测试中用 dict hardcode 重实现。
- **C3** `VADIntegrationTests`: VAD 静默检测用 inline 计时器重实现。
- **C4** `DebounceTests`: debounce 计数逻辑 inline 重实现。

#### D. 缺失覆盖 (5 个)

P1: 多 trigger 组合、极端文本（空串/纯标点/超长文本）、`look_ahead_chars` 窗口、`min_speech_s` 最小语音段、`reset()` 后状态清零 均无测试。

---

### 2.5 tests/int/test_merge.py

**问题数：10 (P0: 3, P1: 6, P2: 1)**

#### A. 自证测试 (2 个, 全部 P1)

- `test_sample_rate_mismatch_detection` (~line 135): 手动比较两个数字。
- `test_merge_handles_missing_wav_gracefully` (~line 190): 构建 bool 再断言。

#### C. 重实现逻辑 (5 个, 全部 P0)

全部 9 个测试用 inline Python 算术算 expected value，**从未调用任何 production merge 函数**。

- `merge_project()` 从未被调用
- `traverse_segments()` 从未被调用
- 音频合并逻辑从未被调用
- 时间戳/间隔计算全部 inline

**这个文件提供零真实覆盖，需要完全重写。**

#### D. 缺失覆盖 (3 个)

P1: 真实 WAV fixture + `PodcastManager.merge_project()` 调用、并发写入、大文件性能 均无测试。

---

### 2.6 tests/int/test_models.py

**问题数：4 (P0: 1, P1: 2, P2: 1)** — **最健康的文件**

#### 改善

- `SegmentSource` 枚举和 source field 测试已添加。
- `live_speakers_override` 拼写错误已修复（之前是 `live_spekers_overrride`）。

#### 残留问题

- **A1 [P1]**: `test_auto_mark_source_on_creation` 仍用手动赋值模拟 auto-marking。
- **D1 [P1]**: `PodcastProject.live_speakers_override` 与 `VoicesConfig.live_speakers` 的优先级逻辑无测试。
- **D2 [P1]**: 模型验证（Pydantic validator）无测试。

---

### 2.7 tests/int/test_config.py

**问题数：5 (P0: 1, P1: 3, P2: 1)** — **较健康**

#### 改善

- `LiveConfig` 三个字段的加载/保存/覆盖已有测试，且调用了真实的 `load_config` / `save_config` 函数。

#### 残留问题

- **A1 [P1]**: resolve 优先级测试 (`test_live_overrides_project_voice`) 手动构建 dict 比较，应调用真实 resolve helper（如果存在）。
- **D1 [P1]**: `LiveConfig` 字段边界值（负数、零、超大值）无测试。
- **D2 [P1]**: `save_config_to_yaml` roundtrip 后字段完整性无测试。

---

### 2.8 tests/int/test_wav_writer.py

**问题数：9 (P0: 0, P1: 7, P2: 2)**

#### 亮点

`WAVHeaderTests` 和 `PeakNormalizeTests` 是整个测试套件中**最强的测试**——调用真实函数、传入真实数据、验证真实输出。

#### 残留问题

A 类 5 个（全部 P1）：naming format、stored path、auto_delete、cleanup task、retention 均为手动布尔表达式。
D 类 2 个：`SessionAudioPipeline` 无测试；`peak_normalize` 未测 int16 输入。

---

### 2.9 tests/int/test_ws_protocol.py

**问题数：11 (P0: 2, P1: 6, P2: 3)**

#### A. 自证测试 (6 个, 1 个 P0)

- `test_disconnect_during_recording_enters_paused` (~line 155): `if disconnected: state = PAUSED` — inline if-statement 实现断连逻辑。
- `test_server/client_to_server_binary_is_pcm_int16`: 用 `struct.pack` 再断言 `len`。
- `test_binary_frame_no_wav_header`: 零字节不以 `b"RIFF"` 开头 — trivially true。
- `test_json/binary_frame_detected`: 测试 Python `isinstance`。

#### B. API 签名不匹配 (1 个, P1)

用通用 `WSFrame(payload, direction)` 代替 spec 定义的 `live_state()` / `error()` / `segment_begin()` / `client_audio_info()` / `state_ack()` 构造函数。

#### D. 缺失覆盖 (5 个)

P1: `live_state`、`client_audio_info`、`state_ack` 作为独立构造函数无测试；`SessionWSContext` 无测试。

---

## 3. 修复优先级

### P0 — 必须修（影响测试有效性）

1. **所有文件**: 用 `session.transition(trigger)` 替换 `session.state = X`。加 negative test 验证非法转换抛 `IllegalStateTransition`。
2. **test_end_detector.py**: 用 `EndDetector.update_asr(text)` / `on_speech_start()` / `on_speech_end()` 替换 inline LCS 和 trigger 状态机。从 `end_detector.py` 导入 `_LEGAL_TRANSITIONS` 而非 hardcode。
3. **test_merge.py**: 完全重写。生成真实 WAV fixture，调用 `PodcastManager.merge_project()`，验证输出文件。
4. **test_tts_streaming.py**: 用真实音频数据调用 `StreamingTTSProxy`（即使 `stream_segment` 当前抛 `NotImplementedError`，也应测试接口契约）。
5. **test_live_podcast_flow.py**: 修复 ASR ratio 公式（用 LCS 而非长度比），或改为调用 `EndDetector.update_asr()`。

### P1 — 应该修（覆盖缺失）

6. 补充 WS frame 构造函数测试：`live_state()`、`error()`、`segment_begin()`、`client_audio_info()`、`state_ack()`。
7. 补充 `SessionAudioPipeline` 和 `SessionWSContext` 集成测试。
8. 补充 `LiveConfig` 边界值测试。
9. 补充 `cancel`、`client_disconnect`、`client_reconnect`、`force_stop` trigger 测试。
10. 补充 `ABANDONED` 终态行为测试。

### P2 — 可以后续修

11. `peak_normalize` int16 输入测试。
12. WS 并发写入测试。
13. 大文件 merge 性能测试。

---

## 4. 修复原则

1. **一个测试只验证一件事**。不要用 inline 算术算 expected value——调用真实函数获取。
2. **Import, don't reimplement**。如果测试里出现了 LCS 实现、状态转换表、chunk slicing 算法的副本，说明方向错了。
3. **用 public API 驱动**。测试应模拟真实调用者的行为：`registry.start_live_session()` → `registry.transition()` → `registry.stop_live_session()`。
4. **Negative test 同等重要**。验证合法转换成功的同时，必须验证非法转换抛异常。
5. **Fixture 用真实数据**。WAV 文件用 `wave` 模块生成真实文件，不要用空 bytes。
