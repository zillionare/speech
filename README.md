# TTS Web Service

基于 mlx-audio 和 VibeVoice 的文本转语音 Web Service，支持单 speaker 和多 speaker 播客生成。

## 特性

- **单 Speaker TTS**: 兼容 OpenAI TTS API 格式
- **多 Speaker 播客生成**: 支持 "Speaker X: 文本" 格式
- **声音样本管理**: 自动扫描 {speaker}.wav 文件
- **CLI 管理**: start/stop/restart/status 命令
- **配置化**: YAML 配置文件，CLI 可覆盖
- **支持长文本**: 最大 64KB 输入文本

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 配置

编辑 `config.yaml`:

```yaml
model:
  name: "mlx-community/VibeVoice-Realtime-0.5B-4bit"

defaults:
  voice: "zh-Aaron_man"
  response_format: "wav"
```

## 使用

### CLI 命令

```bash
# 启动服务 (后台模式)
python run.py start

# 前台模式运行
python run.py start --no-daemon

# 指定端口
python run.py start --port 9000

# 查看状态
python run.py status

# 重启
python run.py restart

# 停止
python run.py stop
```

### API 调用

**单 Speaker 生成:**
```bash
curl -X POST http://localhost:8123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello, this is a test.",
    "voice": "zh-Aaron_man"
  }' \
  --output test.wav
```

**多 Speaker 播客:**
```bash
curl -X POST http://localhost:8123/v1/audio/podcast \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Speaker 1: Hello everyone!\nSpeaker 2: Nice to meet you!",
    "voice_mapping": {
      "Speaker 1": "zh-Aaron_man",
      "Speaker 2": "en-Alice_woman"
    }
  }' \
  --output podcast.wav
```

**获取可用声音列表:**
```bash
curl http://localhost:8123/v1/voices
# 或详细列表
curl http://localhost:8123/v1/voices/details
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/v1/voices` | GET | 声音列表 |
| `/v1/voices/details` | GET | 声音详情 |
| `/v1/audio/speech` | POST | 单 speaker TTS |
| `/v1/audio/podcast` | POST | 多 speaker 播客 |

## 项目结构

```
tts_service/
├── __init__.py
├── cli.py              # CLI 管理
├── config.py           # 配置管理
├── models.py           # Pydantic 模型
├── sample_manager.py   # 声音样本管理
├── server.py           # FastAPI 服务
├── server_runner.py    # 服务运行器
└── tts_engine.py       # TTS 生成核心

config.yaml             # 配置文件
run.py                  # 启动入口
```
