"""将 config.yaml 改写为适合部署机 pro 的版本。

在 rsync 之后于部署机上运行：

    .venv/bin/python scripts/pro_config.py --config config.yaml

它会就地改写 config.yaml 中与开发机不同的字段：绝对路径（voices/outputs/证书）、
激活的 TTS provider（yovoice）以及远程 ASR 指向本机回环地址。
敏感信息（yovoice token）不写入 config.yaml，而是由 deploy_app.sh 写入
同目录的 .speech-secrets.yaml。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite config.yaml for the pro deployment host")
    parser.add_argument("--config", default="config.yaml", help="config.yaml 路径")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="部署目录的绝对路径，默认取 config.yaml 所在目录",
    )
    parser.add_argument("--asr-base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--yovoice-base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--yovoice-model-id", default="index-2.5-q8")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    base = Path(args.base_dir).resolve() if args.base_dir else config_path.parent
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}

    model = data.setdefault("model", {})
    model["tts_provider"] = "yovoice"
    model["yovoice_base_url"] = args.yovoice_base_url
    model["yovoice_model_id"] = args.yovoice_model_id
    data["active_tts_provider"] = "yovoice"

    data.setdefault("voices", {})["base_dir"] = str(base / "voices")
    data.setdefault("outputs", {})["base_dir"] = str(base / "outputs")

    server = data.setdefault("server", {})
    server["ssl_keyfile"] = str(base / ".certs" / "dev-key.pem")
    server["ssl_certfile"] = str(base / ".certs" / "dev-cert.pem")

    asr = data.setdefault("asr", {})
    asr["backend"] = "openai_remote"
    asr["remote_base_url"] = args.asr_base_url
    if not asr.get("remote_model"):
        asr["remote_model"] = asr.get("model", "Qwen3-ASR-0.6B-MLX-8bit")
    if not asr.get("remote_api_key"):
        asr["remote_api_key"] = "1234"

    config_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(
        f"已改写 {config_path}: provider=yovoice({args.yovoice_base_url}) "
        f"asr={asr['backend']}({args.asr_base_url}) base_dir={base}"
    )


if __name__ == "__main__":
    main()
