# Architecture & Interface Review — Round 2

> 审查对象：`story.md` 和 `spec.md`（Round 1 修改后版本）
> 审查日期：2026-07-15

---

## Round 1 问题解决情况

| Round 1 问题 | 状态 | 说明 |
|---|---|---|
| 1.1 LiveSession 职责过重 | ⚠️ 部分 | 注释标注了拆分意图，但 `audio_buffer` 仍在 dataclass 中，`SessionAudioPipeline`/`SessionWSContext` 未定义 |
| 1.2 asr 字段归属矛盾 | ✅ 已解决 | SPEC-004 §4.8 明确 ASR 属于 Registry，附带代码示例 |
| 1.3 WAITING_TRIGGER 语义过载 | ✅ 已解决 | 拆分为 `DETECTING` + `PAUSED`，新增 `ABANDONED` |
| 1.4 TTS 延迟上限 | ✅ 已解决 | 新增 ST-LP-019 + SPEC-005 §5.7 |
| 1.5 BaseEngine.sample_rate | ✅ 已解决 | SPEC-005 §5.8 定义抽象属性 |
| 2.1 audio_info 不对称 | ⚠️ 部分 | ST-LP-007 提及客户端发送，但 SPEC-006 帧列表未列出 |
| 2.2 LiveSession↔Engine 接口 | ⚠️ 部分 | Proxy 构造函数改为接收 engine，但注入方式未定义 |
| 2.3 组件间数据流 | ❌ 未处理 | 无 DataFlow 章节 |
| 2.4 segment_start 时序 | ❌ 未处理 | 无 `segment_audio_begin` 帧 |
| 2.5 RECORDING→AI_SPEAKING 竞态 | ❌ 未处理 | 无 `state_ack` 机制 |
| 2.6 LiveSession↔PodcastManager 同步 | ✅ 已解决 | SPEC-016 §16.6 两层写入策略 |
| 2.7 cancel 端点 | ✅ 已解决 | `DELETE .../live/{sid}` 新增 |
| 2.8 peak_normalize 兼容性 | ❌ 未处理 | 未在 spec 中确认 |

**解决率：7/13（54%）**，其中 4 项完全解决，4 项部分解决。

---

## 本轮新发现的问题

### 3.1 `SessionAudioPipeline` 和 `SessionWSContext` 只存在于注释中

SPEC-004 §4.4 的注释写道：

```
# audio_buffer: removed -- moved to SessionAudioPipeline
# ws_clients: removed -- moved to SessionWSContext
```

但 `audio_buffer: Dict[int, bytearray]` 字段仍在 dataclass 中定义（第 249 行），与注释矛盾。且整个 spec 没有任何地方定义 `SessionAudioPipeline` 或 `SessionWSContext` 的接口。

**建议**：要么从 `LiveSession` 中移除这两个字段并新增 SPEC 章节定义新模块的接口，要么撤回拆分、保留在 `LiveSession` 中。

---

### 3.2 `IDLE → DETECTING` 转换不合理

状态转换表（SPEC-004 §4.3）允许 `IDLE → DETECTING`。但 `DETECTING` 的含义是"段刚结束，等待 EndDetector 输出 trigger"。从 `IDLE`（会话刚创建，尚未开始）直接进入 `DETECTING` 没有意义——没有段刚刚结束，也没有 EndDetector 在运行。

**建议**：移除 `IDLE → DETECTING` 转换。

---

### 3.3 `PAUSED → DETECTING` 转换不合理

`PAUSED` 表示断线或用户暂停。恢复时应该回到断线前的状态（`AI_SPEAKING` 或 `RECORDING`），而不是进入 `DETECTING`（上一段刚结束）。

**建议**：移除 `PAUSED → DETECTING`，改为恢复时回到断线前状态。

---

### 3.4 SPEC-005 §5.7 与 ST-LP-019 的矛盾

SPEC-005 §5.7 写道"不引入新的 story"，但 story.md 中已新增 ST-LP-019。这是遗留注释，需要删除。

---

### 3.5 引用不存在的 `interfaces.md`

SPEC-005 §5.7 写道"配置项已在 interfaces.md 的 LiveConfig 中定义"，但项目中不存在 `interfaces.md` 文件。

---

### 3.6 `LiveConfig` 未在任何地方定义

ST-LP-019 引用 `config.live` 中的 `tts_max_seconds` 和 `tts_timeout_seconds`，SPEC-005 §5.7 引用 `config.live.max_ai_rephrases_per_segment`。但没有任何 SPEC 章节定义 `LiveConfig` Pydantic 模型或 `config.live` YAML 段。

**建议**：新增一个小节，在 `config.py` 中定义 `LiveConfig`：

```python
class LiveConfig(BaseModel):
    tts_max_seconds: int = 60
    tts_timeout_seconds: int = 30
    max_ai_rephrases_per_segment: int = 2
```

---

### 3.7 `LiveSession` 中残留的冗余字段

当前 `LiveSession` dataclass（SPEC-004 §4.4）同时包含：
- `alignment_score: float` — 应属于 `EndDetector`
- `last_asr_text: str` — 应属于 `EndDetector` 或 `SessionAudioPipeline`
- `errors: List[str]` — 可用于 session 级别，但具体含义未定义

如果注释说 `audio_buffer` 和 `ws_clients` 已移到独立模块，这些字段也应该随之迁移。

---

### 3.8 模块依赖图未更新

Round 1 建议的模块依赖图未纳入 spec 文档。当前 spec 提到 `SessionAudioPipeline` 和 `SessionWSContext` 但不定义它们，提到 `LiveSessionRegistry` 但不说明与其他模块的关系。

---

### 3.9 里程碑划分未反映新增内容

实施顺序（SPEC-013 末尾）仍为 6 个里程碑，未包含 ST-LP-019（TTS 超时降级）和 cancel 端点。这些应归入某个里程碑。

---

## 仍需处理的问题（按优先级）

| 优先级 | 问题 | 建议 |
|:------:|------|------|
| **P0** | `SessionAudioPipeline` / `SessionWSContext` 未定义 | 新增 SPEC-004A/004B 定义接口，或撤回拆分 |
| **P0** | `LiveConfig` 未定义 | 在 SPEC-001 或新建小节中定义 |
| **P0** | `IDLE → DETECTING` 无效转换 | 删除 |
| **P0** | `audio_buffer` 字段与注释矛盾 | 删除字段或删除注释，二选一 |
| **P1** | SPEC-005 §5.7 "不引入新的 story" 遗留注释 | 删除 |
| **P1** | `interfaces.md` 引用不存在 | 删除引用或创建文件 |
| **P1** | `PAUSED → DETECTING` 不合理转换 | 删除 |
| **P1** | 组件间数据流未文档化 | 新增 DataFlow 章节 |
| **P1** | Client→Server `audio_info` 帧未列入 SPEC-006 | 添加到帧列表 |
| **P2** | `segment_start` 首帧时序缺口 | 新增 `segment_audio_begin` 帧 |
| **P2** | RECORDING→AI_SPEAKING 竞态 | 新增 `state_ack` 机制 |
| **P2** | 里程碑未包含 ST-LP-019 | 更新里程碑表 |

---

## 总结

| 维度 | Round 1 | Round 2 |
|------|:-------:|:-------:|
| 功能完整性 | 85% | 90%（新增 TTS 超时/cancel） |
| 接口清晰度 | 模糊 | 改善（ASR 归属、同步协议已明确），但新模块只存在于注释中 |
| 职责分离 | 差 | 改善（状态机已拆分），但实现不彻底 |
| 内部一致性 | 中等 | 下降（dataclass 字段与注释矛盾、引用不存在文件） |
| 可实施性 | 可开工 | 可开工，但 `SessionAudioPipeline` 等新模块需要先定义才能动手 |

**核心建议**：要么投入精力完整定义 `SessionAudioPipeline` 和 `SessionWSContext` 接口（新增 2-3 个 SPEC 小节），要么撤回拆分、保持 `LiveSession` 为中等大小的 dataclass（当前 ~15 个字段是合理的）。半拆半留的注释状态是最差的——既增加了阅读混淆，又没提供实际指导。**推荐撤回拆分**，将 `LiveSession` 作为单一 truth source，待 M3 实现后根据实际复杂度再决定是否拆分。