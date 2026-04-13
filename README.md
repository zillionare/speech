# TTS Web Service

基于 mlx-audio 和 VibeVoice 的文本转语音 Web Service，支持单 speaker 和多 speaker 播客生成。

## 特性

- **单 Speaker TTS**: 兼容 OpenAI TTS API 格式
- **多 Speaker 播客生成**: 支持 "Speaker X: 文本" 格式
- **声音样本管理**: 自动扫描 {speaker}.wav 文件
- **CLI 管理**: start/stop/restart/status 命令
- **配置化**: YAML 配置文件，CLI 可覆盖
- **首次运行向导**: 交互式配置模型、量化、缓存目录
- **模型量化支持**: 4bit/8bit 可选
- **支持长文本**: 最大 64KB 输入文本

## 模型支持

| 模型 | 大小 | 量化 | 推荐场景 |
|------|------|------|----------|
| `gafiatulin/vibevoice-1.5b-mlx` | 1.5B | 8bit | **生产推荐** - 平衡速度与质量 |
| `milkey/vibevoice-7b-mlx:4bit` | 7B | 4bit | 更高质量，需要更多内存 |
| `mlx-community/VibeVoice-Realtime-0.5B-4bit` | 0.5B | 4bit | 快速推理，资源受限 |

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 首次运行

直接启动服务会自动运行配置向导：

```bash
python run.py start --no-daemon
```

向导会引导你配置：
1. HuggingFace 镜像地址（默认 hf-mirror.com）
2. TTS 模型选择（默认 1.5B 8bit）
3. 量化方式（4bit/8bit）
4. 模型缓存目录（默认 ~/.speech/models）
5. 声音样本目录（默认 ~/.speech/voices）

## 配置

编辑 `config.yaml`:

```yaml
# 首次运行配置
first_run:
  hf_endpoint: "https://hf-mirror.com"
  skip_welcome: false

# 模型配置
model:
  name: "gafiatulin/vibevoice-1.5b-mlx"
  cache_dir: "~/.speech/models"
  quantization: "8bit"  # 可选: "4bit", "8bit", 或 null
  num_steps: 10
  cfg_scale: 1.3

samples:
  base_dir: "~/.speech/voices"
  allowed_speakers: []

server:
  host: "0.0.0.0"
  port: 8123
  workers: 1
  log_level: "info"

defaults:
  voice: "zh-Aaron_man"
  response_format: "wav"
  speed: 1.0
```

## 使用

### CLI 命令

```bash
# 启动服务 (后台模式)
python run.py start

# 前台模式运行 (推荐首次运行)
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
├── first_run.py        # 首次运行向导
├── models.py           # Pydantic 模型
├── sample_manager.py   # 声音样本管理
├── server.py           # FastAPI 服务
├── server_runner.py    # 服务运行器
└── tts_engine.py       # TTS 生成核心

config.yaml             # 配置文件
run.py                  # 启动入口
```
