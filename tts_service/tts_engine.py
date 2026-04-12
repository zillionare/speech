"""TTS 引擎模块 v2 - 使用 CLI 方式调用"""

import os
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Tuple, Union

from .sample_manager import SampleManager, MultiSpeakerParser


class TTSEngine:
    """TTS 引擎，支持单 speaker 和多 speaker 生成（使用 CLI 方式）"""

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
        # 英文 voices
        "en-Alice_woman": "en-Emma_woman",
        "en-Carter_man": "en-Carter_man",
        "en-Frank_man": "en-Frank_man",
        "en-Maya_woman": "en-Grace_woman",
        "en-Mary_woman_bgm": "en-Emma_woman",
        "in-Samuel_man": "in-Samuel_man",
        # 中文 voices - 映射到相近的英文 voice
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
        self.sample_manager = SampleManager(
            config.samples.expanded_base_dir,
            config.samples.allowed_speakers if config.samples.allowed_speakers else None
        )

    def _get_model_voice(self, voice: str) -> Optional[str]:
        """
        获取模型可用的 voice 名称

        1. 如果是内置 voice，直接使用
        2. 如果有映射，使用映射后的 voice
        3. 如果没有映射，检查是否包含内置 voice 的相似名
        """
        # 直接匹配内置 voice
        if voice in self.BUILTIN_VOICES:
            return voice

        # 查找映射
        mapped = self.VOICE_MAPPING.get(voice)
        if mapped:
            return mapped

        # 尝试从 voice 名推断（如从 "zh-Aaron_man" 推断出 "_man" 结尾）
        if "woman" in voice.lower() or "_woman" in voice:
            # 女性声音默认使用 en-Emma_woman
            return "en-Emma_woman"
        elif "man" in voice.lower() or "_man" in voice:
            # 男性声音默认使用 en-Frank_man
            return "en-Frank_man"

        # 默认
        return "en-Emma_woman"

    def generate_single(
        self,
        text: str,
        voice: str,
        output_format: str = "wav",
        speed: float = 1.0,
        **kwargs
    ) -> bytes:
        """
        单 speaker TTS 生成（使用 mlx-audio CLI）
        """
        # 获取模型可用的 voice
        model_voice = self._get_model_voice(voice)

        print(f"Generating audio: voice={voice} -> {model_voice}, text_length={len(text)}")

        # 使用临时目录
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, f"output.{output_format}")

            try:
                # 构建 CLI 命令
                cmd = [
                    "python", "-m", "mlx_audio.tts.generate",
                    "--model", self.model_name,
                    "--text", text,
                    "--voice", model_voice,
                ]

                if speed != 1.0:
                    cmd.extend(["--speed", str(speed)])

                # 将文件重命名到目标路径
                # CLI 默认输出为 audio_000.wav
                default_output = "audio_000.wav"

                print(f"  Running: {' '.join(cmd[:8])}...")
                start_time = time.time()

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=180,  # 3分钟超时
                    cwd=tmpdir  # 在临时目录中运行
                )

                elapsed = time.time() - start_time
                print(f"  CLI completed in {elapsed:.2f}s")

                if result.returncode != 0:
                    error_msg = result.stderr if result.stderr else "Unknown error"
                    raise RuntimeError(f"CLI failed: {error_msg}")

                # CLI 输出到当前目录的 audio_000.wav
                cli_output = os.path.join(tmpdir, default_output)

                # 读取生成的音频
                if os.path.exists(cli_output):
                    with open(cli_output, 'rb') as f:
                        data = f.read()
                    print(f"  Generated: {len(data)} bytes")
                    return data
                else:
                    raise RuntimeError(f"Output file not found: {cli_output}")

            except subprocess.TimeoutExpired:
                raise RuntimeError("TTS generation timeout (180s)")
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
            print(f"    Text: {segment_text[:50]}...")

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

            # 保存所有片段
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
                else:
                    print(f"  FFmpeg failed: {result.stderr[:200]}")
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                print(f"  FFmpeg not available: {e}")

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
