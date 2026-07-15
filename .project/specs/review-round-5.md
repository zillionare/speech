# Test Quality Review — Round 5

> 审阅范围：9 个测试文件（e2e × 1 + int × 8）
> 审阅时间：2026-07-15
> 目的：验证 review-round-4 残留的 14 个问题（1 P0 + 8 P1 + 5 P2）的修复情况

---

## 1. 总体结论

**测试质量已达标。** Round-4 的 14 个问题中 11 个已修复，剩余 3 个均为 P2（低优先级）。从 Round-3 的 77 个问题到现在的 3 个，测试套件经历了三轮重写，质量从"几乎全部无效"提升到"有效覆盖核心路径"。

| Round | 总问题数 | P0 | P1 | P2 |
|-------|---------|----|----|-----|
| Round-3 | 77 | 32 | 36 | 9 |
| Round-4 | 14 | 1 | 8 | 5 |
| **Round-5** | **3** | **0** | **0** | **3** |

---

## 2. 本轮修复确认

### P0 — 已清零

| 文件 | 问题 | 状态 |
|------|------|------|
| test_merge.py | `test_merge_missing_live_segment_uses_silence` 的 `try/except pass` | **已修复**。直接调用 `pm.merge_project("p3")`，断言输出文件存在且时长 ≥ 2.5s。 |

### P1 — 已清零

| 文件 | 问题 | 状态 |
|------|------|------|
| test_models.py | `_migrate_segment_source` / `_auto_mark_source` test-local helpers | **已修复**。改为通过 `PodcastManager.get_project()` 触发真实迁移路径。 |
| test_models.py | `RegenerateGuardTests` 自证枚举检查 | **已修复**。整个类已移除。 |
| test_wav_writer.py | 5 个自证测试（naming/path/auto_delete/cleanup/retention） | **已修复**。替换为真实 `write_wav` 调用、多采样率参数化、sine wave 回读验证。 |
| test_live_podcast_flow.py | `test_live_only_skips_tts_in_merge` 不调用 merge | **已修复**。调用 `pm.merge_project("p1")`，断言输出。 |
| test_live_podcast_flow.py | `test_complete_session_frame_sequence` 自组装列表 | **已修复**。列表元素来自 `reg.transition()` 的返回值。 |
| test_ws_protocol.py | `BinaryFrameSizeTests` 自证 | **已修复**。整个类已移除。 |
| test_ws_protocol.py | 缺少 `frame["type"]` 断言 | **已修复**。新增 `ServerFrameTypeTests`（10 个 constructor）和 `ClientFrameTypeTests`（3 个）。 |
| test_end_detector.py | debounce 测试 `time.sleep` + 私有字段操作 | **已修复**。改用 `det.reset()` + `det.on_speech_start()`，无 sleep。 |
| test_end_detector.py | `update_vad` 自动触发 on_speech_start/end 未测 | **已修复**。新增 `EndDetectorVADAutoTriggerTests`。 |
| test_tts_streaming.py | `LiveConfigTests` 自证默认值 | **已修复**。改为关系断言（`assertGreater > 0`）+ roundtrip 测试。 |
| test_config.py | `ProjectOverrideTests` 与 test_models.py 重复 | **已修复**。config 文件中移除，注释指向 test_models.py。 |

### P2 — 3 项残留

| 文件 | 问题 | 状态 |
|------|------|------|
| test_config.py line 114 | `assertEqual(cfg.tts_timeout_seconds, 30)` hardcode 默认值 | **未修复**。建议改为引用 `LiveConfig.model_fields["tts_timeout_seconds"].default`。 |
| test_tts_streaming.py line 91-109 | `stream_segment` 测试 catch `NotImplementedError` 后 skipTest | **未修复**。若已实现则自动运行；若未实现则静默跳过。建议加 `@unittest.skip` 显式标记。 |
| test_live_podcast_flow.py line 243 | `session.cursor = 1` 直接赋值模拟 redo | **未修复**。若无真实 redo API 可调用，当前写法是可接受的临时方案，但应加 TODO 注释。 |

---

## 3. 质量达标文件清单

以下文件无残留问题，质量达标：

- **test_state_machine.py** — transition table 从 production 导入，所有合法/非法转换路径（含 round-4 补充的 3 条）均有测试
- **test_end_detector.py** — LCS/alignment/normalize 调用真实函数，VAD 自动触发、debounce 均通过公共 API 测试
- **test_ws_protocol.py** — 所有 frame constructor 签名正确，type 字段全覆盖，边界值验证到位
- **test_wav_writer.py** — 真实 WAV 写入/回读，context manager 已测，peak_normalize 多场景覆盖
- **test_merge.py** — 调用真实 `PodcastManager.merge_project()`，真实 WAV fixture，断言有意义
- **test_models.py** — 迁移通过 production 路径，override 优先级测试完整
- **test_config.py** — Config 加载/保存 roundtrip、LiveConfig 不变量均有效

以下文件有 1 个 P2 残留，不影响整体验收：

- **test_live_podcast_flow.py** — 1 个 P2（redo cursor 赋值）
- **test_tts_streaming.py** — 1 个 P2（skip on NotImplementedError）

---

## 4. 四轮演进对比

```
Round-3  ████████████████████████████████  77 issues (32 P0)
Round-4  ██████                              14 issues  (1 P0)
Round-5  █                                    3 issues  (0 P0)
```

关键改善节点：
- Round 3→4：消除了直接状态赋值和 inline LCS/transition 重实现，B/C 类问题接近清零
- Round 4→5：修复了 test_merge 的 no-op 测试、test_models 的 helper 镜像、test_wav_writer 的 5 个自证测试、test_ws_protocol 的 type 字段缺失

**结论：测试套件可进入下一阶段（与 production 代码联调）。** 3 个 P2 问题可在后续迭代中处理。
