#!/usr/bin/env python3
"""
完整 TTS 生成测试脚本
测试配置加载、模型加载和音频生成
"""

import sys
sys.path.insert(0, '/Users/aaronyang/workspace/speech')

import os
import tempfile
import time

# 设置 HF_ENDPOINT
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = os.path.expanduser('~/.speech/models')

from tts_service.config import load_config
from tts_service.tts_engine import TTSEngine
from tts_service.first_run import setup_environment


def test_tts_generation():
    """测试完整 TTS 生成流程"""
    print("=" * 60)
    print("TTS 完整生成测试")
    print("=" * 60)

    # 1. 创建测试配置
    config_data = {
        "first_run": {
            "hf_endpoint": "https://hf-mirror.com",
            "skip_welcome": True
        },
        "model": {
            "name": "gafiatulin/vibevoice-1.5b-mlx",
            "cache_dir": "~/.speech/models",
            "quantization": "8bit",
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

    # 写入临时配置文件
    import yaml
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name

    print(f"\n1. 配置文件: {config_path}")

    # 2. 加载配置
    try:
        config = load_config(config_path)
        print(f"\n2. 配置加载成功:")
        print(f"   Model: {config.model.name}")
        print(f"   Quantization: {config.model.quantization}")
        print(f"   Cache dir: {config.model.expanded_cache_dir}")
    except Exception as e:
        print(f"   配置加载失败: {e}")
        return False

    # 3. 设置环境
    setup_environment(config)
    print(f"\n3. 环境设置完成")
    print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")

    # 4. 创建 TTS 引擎
    print(f"\n4. 创建 TTS 引擎...")
    try:
        engine = TTSEngine(config)
        print(f"   TTS 引擎创建成功")
        print(f"   Model: {engine.model_name}")
        print(f"   Quantization: {engine.quantization}")
    except Exception as e:
        print(f"   TTS 引擎创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 5. 测试音频生成
    test_text = "Hello, this is a test of the TTS service with 8bit quantization."
    print(f"\n5. 生成音频...")
    print(f"   文本: {test_text}")
    print(f"   声音: {config.defaults.voice}")
    print(f"   格式: {config.defaults.response_format}")

    try:
        start_time = time.time()
        audio_data = engine.generate_single(
            text=test_text,
            voice=config.defaults.voice,
            output_format=config.defaults.response_format,
            speed=config.defaults.speed
        )
        duration = time.time() - start_time

        # 保存音频文件
        output_path = os.path.expanduser("~/test_output.wav")
        with open(output_path, 'wb') as f:
            f.write(audio_data)

        print(f"\n   ✅ 音频生成成功!")
        print(f"   生成时间: {duration:.2f}s")
        print(f"   文件大小: {len(audio_data)} bytes")
        print(f"   保存路径: {output_path}")

        # 6. 验证模型状态
        status = engine.get_model_status()
        print(f"\n6. 模型状态:")
        print(f"   已加载: {status['is_loaded']}")
        print(f"   模型名称: {status['model_name']}")
        print(f"   量化方式: {status.get('quantization', 'N/A')}")

        return True

    except Exception as e:
        print(f"\n   ❌ 音频生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理临时配置文件
        try:
            os.unlink(config_path)
        except:
            pass


if __name__ == "__main__":
    success = test_tts_generation()
    print("\n" + "=" * 60)
    if success:
        print("✅ 测试通过!")
    else:
        print("❌ 测试失败!")
    print("=" * 60)
    sys.exit(0 if success else 1)
