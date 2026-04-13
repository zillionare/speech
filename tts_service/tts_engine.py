"""TTS 引擎模块 - 使用 mlx-audio API 直接调用"""

import os
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

import numpy as np

from mlx_audio.tts import load as load_tts_model
from mlx_audio.audio_io import write as audio_write

from .sample_manager import SampleManager, MultiSpeakerParser


class ModelManager:
    """模型管理器，支持自动加载和卸载"""

    # 1小时无访问自动卸载 (秒)
    OFFLOAD_TIMEOUT = 3600

    def __init__(self, model_name: str, quantization: Optional[str] = None):
        self.model_name = model_name
        self.quantization = quantization
        self._model = None
        self._last_access_time = 0
        self._lock = threading.RLock()
        self._offload_timer = None

    def get_model(self):
        """获取模型（自动加载）"""
        with self._lock:
            if self._model is None:
                print(f"Loading TTS model: {self.model_name}...")
                if self.quantization:
                    print(f"  Quantization: {self.quantization}")
                start_time = time.time()

                # 根据量化配置加载模型
                # 注意: 有些模型可能缺少非关键参数，使用 strict=False 忽略缺失的参数
                load_kwargs = {'strict': False}
                if self.quantization:
                    load_kwargs['quantization'] = self.quantization

                self._model = load_tts_model(self.model_name, **load_kwargs)
                load_time = time.time() - start_time
                print(f"Model loaded in {load_time:.2f}s")

            self._last_access_time = time.time()
            self._schedule_offload()
            return self._model

    def _schedule_offload(self):
        """安排自动卸载"""
        # 取消之前的定时器
        if self._offload_timer is not None:
            self._offload_timer.cancel()

        # 创建新的定时器
        self._offload_timer = threading.Timer(
            self.OFFLOAD_TIMEOUT,
            self._offload_if_inactive
        )
        self._offload_timer.daemon = True
        self._offload_timer.start()

    def _offload_if_inactive(self):
        """如果超过时间无访问，卸载模型"""
        with self._lock:
            elapsed = time.time() - self._last_access_time
            if elapsed >= self.OFFLOAD_TIMEOUT and self._model is not None:
                print(f"Model inactive for {elapsed:.0f}s, offloading...")
                self._model = None
                import gc
                gc.collect()
                print("Model offloaded")
            elif self._model is not None:
                # 还有活跃使用，重新安排
                self._schedule_offload()

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None


class TTSEngine:
    """TTS 引擎，支持单 speaker 和多 speaker 生成"""

    # VibeVoice 内置 voices
    BUILTIN_VOICES = [
        "de-Spk0_man", "de-Spk1_woman",
        "en-Carter_man", "en-Davis_man", "en-Emma_woman", "en-Frank_man",
        "en-Grace_woman", "en-Mike_man",
        "fr-Spk0_man", "fr-Spk1_woman",
        "in-Samuel_man",
        "it-Spk0_woman", "it-Spk1_man",
        "jp-Spk0_man", "jp-Spk1_woman",
        "kr-Spk0_woman", "kr-Spk1_man",
        "nl-Spk0_man", "nl-Spk1_woman",
        "pl-Spk0_man", "pl-Spk1_woman",
        "pt-Spk0_woman", "pt-Spk1_man",
        "sp-Spk0_woman", "sp-Spk1_man",
    ]

    # 本地 voice 到内置 voice 的映射
    VOICE_MAPPING = {
        "en-Alice_woman": "en-Emma_woman",
        "en-Carter_man": "en-Carter_man",
        "en-Frank_man": "en-Frank_man",
        "en-Maya_woman": "en-Grace_woman",
        "en-Mary_woman_bgm": "en-Emma_woman",
        "in-Samuel_man": "in-Samuel_man",
        "zh-Aaron_man": "en-Frank_man",
        "zh-Bowen_man": "en-Davis_man",
        "zh-007_man": "en-Mike_man",
        "zh-007_woman": "en-Emma_woman",
        "zh-Xinran_woman": "en-Grace_woman",
        "zh-Anchen_man_bgm": "en-Carter_man",
    }

    def __init__(self, config):
        """初始化 TTS 引擎"""
        self.config = config
        self.model_name = config.model.name
        self.quantization = config.model.quantization
        self.sample_manager = SampleManager(
            config.samples.expanded_base_dir,
            config.samples.allowed_speakers if config.samples.allowed_speakers else None
        )
        # 使用模型管理器管理加载/卸载
        self._model_manager = ModelManager(self.model_name, self.quantization)

    def _get_model_voice(self, voice: str) -> Optional[str]:
        """获取模型可用的 voice 名称

        Returns:
            voice 名称，如果是本地文件则返回 None（表示使用 ref_audio）
        """
        # 优先检查是否为本地 voice 文件
        local_path = self.sample_manager.get_voice_path(voice)
        if local_path:
            return None  # 本地文件需要使用 ref_audio 参数

        if voice in self.BUILTIN_VOICES:
            return voice

        mapped = self.VOICE_MAPPING.get(voice)
        if mapped:
            return mapped

        # 对于不支持内置 voices 的模型（如 1.5B），返回 None
        if "woman" in voice.lower() or "_woman" in voice:
            return None
        elif "man" in voice.lower() or "_man" in voice:
            return None

        return None

    def generate_single(
        self,
        text: str,
        voice: str,
        output_format: str = "wav",
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        单 speaker TTS 生成（使用 mlx-audio API）
        """
        # 获取模型可用的 voice
        model_voice = self._get_model_voice(voice)

        print(f"Generating audio: voice={voice} -> {model_voice}, text={text[:50]}...")

        # 获取模型（自动加载/卸载管理）
        model = self._model_manager.get_model()

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                print(f"  Using model.generate() with device auto-detection")
                start_time = time.time()

                # 调用模型生成
                results = model.generate(
                    text=text,
                    voice=model_voice,
                    verbose=False,
                )

                # 处理 generator 结果
                audio_samples = None
                sample_rate = 24000

                for result in results:
                    if hasattr(result, 'audio'):
                        audio = result.audio
                        sample_rate = getattr(result, 'sample_rate', 24000)

                        if audio_samples is None:
                            audio_samples = audio
                        else:
                            # 拼接音频
                            if hasattr(audio_samples, 'tolist') and hasattr(audio, 'tolist'):
                                audio_samples = np.concatenate([audio_samples, audio])
                            else:
                                audio_samples = audio

                if audio_samples is None:
                    raise RuntimeError("No audio generated from model")

                # 保存音频文件
                output_file = os.path.join(tmpdir, f"output.{output_format}")
                audio_write(output_file, audio_samples, sample_rate)

                elapsed = time.time() - start_time

                # 读取生成的音频
                with open(output_file, 'rb') as f:
                    data = f.read()

                print(f"  Generated: {len(data)} bytes in {elapsed:.2f}s")
                return data

            except Exception as e:
                print(f"  Generation error: {e}")
                raise

    def generate_podcast(
        self,
        text: str,
        voice_mapping: Dict[str, str],
        output_format: str = "wav",
        speed: float = 1.0
    ) -> bytes:
        """
        多 speaker 播客生成
        """
        # 解析多 speaker 文本
        parser = MultiSpeakerParser(voice_mapping)
        segments = parser.parse(text)

        if not segments:
            raise ValueError("No valid speaker segments found in text")

        print(f"Generating podcast with {len(segments)} segments")

        audio_segments = []

        for i, segment in enumerate(segments):
            speaker_id = segment['speaker_id']
            speaker_name = segment['speaker_name']
            segment_text = segment['text']

            # 获取 voice
            voice = voice_mapping.get(speaker_id) or \
                    voice_mapping.get(speaker_name) or \
                    self.config.defaults.voice

            print(f"  Segment {i+1}/{len(segments)}: {speaker_name} -> {voice}")

            # 生成
            audio_data = self.generate_single(
                text=segment_text,
                voice=voice,
                output_format=output_format,
                speed=speed
            )
            audio_segments.append(audio_data)

        # 合并音频
        return self._merge_audio_segments(audio_segments, output_format)

    def _merge_audio_segments(self, segments: List[bytes], output_format: str) -> bytes:
        """合并多个音频片段"""
        if len(segments) == 1:
            return segments[0]

        print(f"Merging {len(segments)} audio segments...")

        with tempfile.TemporaryDirectory() as tmpdir:
            segment_files = []

            for i, data in enumerate(segments):
                seg_path = os.path.join(tmpdir, f"seg_{i:04d}.{output_format}")
                with open(seg_path, 'wb') as f:
                    f.write(data)
                segment_files.append(seg_path)

            output_path = os.path.join(tmpdir, f"merged.{output_format}")

            # 尝试 ffmpeg
            try:
                list_file = os.path.join(tmpdir, "list.txt")
                with open(list_file, 'w') as f:
                    for seg_path in segment_files:
                        f.write(f"file '{os.path.abspath(seg_path)}'\n")

                result = subprocess.run([
                    'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
                    '-i', list_file, '-c', 'copy', output_path
                ], capture_output=True, text=True, timeout=60)

                if result.returncode == 0 and os.path.exists(output_path):
                    with open(output_path, 'rb') as f:
                        return f.read()
            except (subprocess.SubprocessError, FileNotFoundError):
                pass

            # 回退：手动拼接 WAV
            return self._concat_wav_files(segment_files)

    def _concat_wav_files(self, file_paths: List[str]) -> bytes:
        """直接拼接 WAV 文件"""
        if not file_paths:
            return b''

        with open(file_paths[0], 'rb') as f:
            base_data = f.read()

        if len(base_data) < 44 or base_data[:4] != b'RIFF' or base_data[8:12] != b'WAVE':
            combined = base_data
            for fp in file_paths[1:]:
                with open(fp, 'rb') as f:
                    combined += f.read()
            return combined

        audio_data = base_data[44:]
        for fp in file_paths[1:]:
            with open(fp, 'rb') as f:
                data = f.read()
                if len(data) > 44:
                    audio_data += data[44:]

        total_size = 36 + len(audio_data)
        header = bytearray(44)
        header[0:4] = b'RIFF'
        header[4:8] = struct.pack('<I', total_size)
        header[8:12] = b'WAVE'
        header[12:16] = b'fmt '
        header[16:20] = struct.pack('<I', 16)
        header[20:22] = struct.pack('<H', 1)
        header[22:44] = base_data[22:44]

        return bytes(header) + audio_data

    def list_voices(self) -> List[str]:
        """返回可用声音列表（本地 + 内置）"""
        local_voices = self.sample_manager.list_voices()
        return local_voices + self.BUILTIN_VOICES

    def get_voice_info(self, voice_name: str) -> Dict:
        """获取声音详情"""
        local_path = self.sample_manager.get_voice_path(voice_name)
        local_text = self.sample_manager.get_voice_text(voice_name)
        is_builtin = voice_name in self.BUILTIN_VOICES
        model_voice = self._get_model_voice(voice_name) if not is_builtin and not local_path else None

        return {
            "name": voice_name,
            "path": local_path,
            "has_text": local_text is not None,
            "text_preview": local_text[:100] if local_text else None,
            "is_builtin": is_builtin,
            "maps_to": model_voice
        }

    def get_model_status(self) -> Dict:
        """获取模型状态"""
        return {
            "is_loaded": self._model_manager.is_loaded(),
            "model_name": self.model_name,
            "quantization": self.quantization,
            "last_access_time": self._model_manager._last_access_time,
            "offload_timeout": self._model_manager.OFFLOAD_TIMEOUT
        }
