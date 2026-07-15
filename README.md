# VibeVoice MLX Studio

本目录是一个独立 worktree，用来重建基于 `vibevoice-mlx` 的本地中文语音生成服务。

核心目标：

- 使用 `gafiatulin/vibevoice-1.5b-mlx`
- 默认 8-bit 量化
- 不再依赖 `mlx-audio`
- 使用 `voices/{speaker}.wav + {speaker}.txt` 管理本地声音库
- 在首次使用时把声音编码缓存到 `voices/.cache/{speaker}.safetensors`
- 支持 `Aaron: 这是我要说的话` 这种多角色对话输入
- 通过内置 FastAPI 静态页面完成声音管理与生成

## 安装

建议在当前仓库使用的 Python 环境中安装：

```bash
pip install -r requirements.txt
```

如果你要直接运行生成引擎，还需要确保 worktree 内的 `vibevoice-mlx/` 子目录可用。

## 配置

默认配置在 `config.yaml`：

```yaml
model:
  model_id: "gafiatulin/vibevoice-1.5b-mlx"
  quantize_bits: 8
  diffusion_steps: 10

voices:
  base_dir: "./voices"
  default_voice: "zh-Bowen_man"
  cache_subdir: ".cache"

outputs:
  base_dir: "./outputs"

server:
  host: "0.0.0.0"
  port: 8123
```

## 启动

前台运行：

```bash
python run.py start --no-daemon
```

后台运行：

```bash
python run.py start --daemon
```

常用命令：

```bash
python run.py status
python run.py restart
python run.py stop
```

## Web UI

启动后访问：

```text
http://127.0.0.1:8123/
```

页面支持：

- 上传或替换本地声音样本
- 编辑对应 transcript
- 手动预热 `.safetensors` 缓存
- 输入单人旁白或多角色对话文本
- 查看最近生成结果并直接试听/下载

上传的参考音频会在落盘前自动做轻量预处理：多声道下混为单声道、抑制底噪，并裁掉前后静音，避免把录音环境问题直接带进 voice clone。

## API

主要产品入口是内置 Web UI，但也保留兼容接口：

- `GET /health`
- `GET /api/config`
- `GET /api/voices`
- `POST /api/voices`
- `PUT /api/voices/{speaker}/transcript`
- `POST /api/voices/{speaker}/cache`
- `DELETE /api/voices/{speaker}`
- `POST /api/generate`
- `GET /api/generations`
- `GET /v1/voices`
- `GET /v1/voices/details`
- `POST /v1/audio/speech`
- `POST /v1/audio/podcast`

## 对话输入格式

普通旁白：

```text
这是一个单人旁白示例。
```

多角色对话：

```text
Aaron: 这是第一句。
Bowen: 这是第二句。
Xinran: 这是第三句。
```

解析逻辑：

- 先按角色名查找本地声音
- 如果提供了 `voice_mapping`，优先使用映射值
- 找不到时回退到默认声音 `zh-Bowen_man`

## 目录结构

```text
tts_service/
  cli.py
  config.py
  models.py
  sample_manager.py
  server.py
  server_runner.py
  static/
  tts_engine.py
voices/
outputs/
config.yaml
run.py
spec.md
```

## 当前限制

- `vibevoice-mlx --voice` 使用的是本地 `.safetensors`，不会按中文名字自动下载远端预置音色。
- 因此本项目采用“本地 WAV + 按需缓存”的方式来管理声音，而不是依赖远端符号名。
