#!/usr/bin/env python3
"""
使用 0.5B 模型测试 (之前成功过的模型)
"""

import sys
sys.path.insert(0, '/Users/aaronyang/workspace/speech')

import os
import tempfile
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.expanduser('~/.speech/models')

from tts_service.config import load_config
from tts_service.tts_engine import TTSEngine
from tts_service.first_run import setup_environment
import yaml


def test_with_0_5b_model():
    """使用 0.5B 模型测试 (这是之前 VibeVoiceFusion 项目成功过的模型)"""
    print("=" * 60)
    print("使用 mlx-community/VibeVoice-Realtime-0.5B-4bit 测试")
    print("=" * 60)

    config_data = {
        "first_run": {
            "hf_endpoint": "https://hf-mirror.com",
            "skip_welcome": True
        },
        "model": {
            "name": "mlx-community/VibeVoice-Realtime-0.5B-4bit",
            "cache_dir": "~/.speech/models",
            # 0.5B-4bit 模型不需要 quantization 参数
            "num_steps": 10,
            "cfg_scale": 1.3
        },
        "samples": {
            "base_dir": "~/.speech/voices",
            "allowed_speakers": []
        },
        "server": {
            "host": "0.0.0.0",
            "port": 8123,
            "workers": 1,
            "log_level": "info"
        },
        "defaults": {
            "voice": "en-Emma_woman",
            "response_format": "wav",
            "speed": 1.0
        },
        "pid_file": "/tmp/tts_service.pid"
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    print(f"\n1. 配置: {config_path}")

    config = load_config(config_path)
    print(f"2. 配置加载: {config.model.name}")

    setup_environment(config)
    print(f"3. 环境设置完成")

    print(f"\n4. 创建 TTS 引擎...")
    engine = TTSEngine(config)
    print(f"   TTS 引擎创建成功")

    test_text = "Hello, this is a test of the TTS web service."
    print(f"\n5. 生成音频...")
    print(f"   文本: {test_text}")

    try:
        start_time = time.time()
        audio_data = engine.generate_single(
            text=test_text,
            voice=config.defaults.voice,
            output_format=config.defaults.response_format,
            speed=config.defaults.speed
        )
        duration = time.time() - start_time

        output_path = os.path.expanduser("~/test_output_0.5b.wav")
        with open(output_path, 'wb') as f:
            f.write(audio_data)

        print(f"\n   ✅ 音频生成成功!")
        print(f"   用时: {duration:.2f}s")
        print(f"   文件大小: {len(audio_data)} bytes")
        print(f"   保存路径: {output_path}")
        return True

    except Exception as e:
        print(f"\n   ❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        try:
            os.unlink(config_path)
        except:
            pass


if __name__ == "__main__":
    success = test_with_0_5b_model()
    print("\n" + "=" * 60)
    print("✅ 测试通过!" if success else "❌ 测试失败!")
    print("=" * 60)
