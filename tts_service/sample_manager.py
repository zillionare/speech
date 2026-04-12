"""声音样本管理模块"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional


class SampleManager:
    """管理声音样本文件，支持 {speaker}.wav + {speaker}.txt 配对"""

    # 默认搜索路径，按优先级排序
    DEFAULT_SEARCH_PATHS = [
        "./voices",
        "~/.config/tts-service/voices",
        "~/.tts-service/voices",
        "~/voices",
    ]

    def __init__(self, base_dir: str = "", allowed_speakers: Optional[List[str]] = None):
        """
        初始化样本管理器

        Args:
            base_dir: 声音样本目录 (为空则自动搜索)
            allowed_speakers: 允许的说话人列表 (为空则扫描所有)
        """
        self.allowed_speakers = set(allowed_speakers) if allowed_speakers else None
        self._voice_presets: Dict[str, Path] = {}
        self._voice_texts: Dict[str, str] = {}  # 可选的文本描述

        # 确定 voices 目录
        self.base_dir = self._find_voices_dir(base_dir)
        self._scan_voices()

    def _find_voices_dir(self, base_dir: str) -> Path:
        """查找 voices 目录"""
        # 如果指定了存在的目录，直接使用
        if base_dir:
            expanded = Path(base_dir).expanduser()
            if expanded.exists():
                return expanded

        # 搜索默认路径
        for path in self.DEFAULT_SEARCH_PATHS:
            expanded = Path(path).expanduser()
            if expanded.exists():
                print(f"Found voices directory: {expanded}")
                return expanded

        # 如果没有找到，使用第一个默认路径（创建）
        default_path = Path(self.DEFAULT_SEARCH_PATHS[0])
        print(f"Creating voices directory: {default_path}")
        default_path.mkdir(parents=True, exist_ok=True)
        return default_path

    def _scan_voices(self):
        """扫描声音样本目录"""
        if not self.base_dir.exists():
            print(f"Warning: Voices directory not found at {self.base_dir}")
            return

        # 扫描所有 .wav 文件
        wav_files = list(self.base_dir.glob("*.wav"))
        wav_files.extend(self.base_dir.glob("*.WAV"))  # 支持大写

        for wav_file in wav_files:
            # 获取 speaker 名称 (不含扩展名)
            speaker_name = wav_file.stem

            # 如果指定了允许列表，则过滤
            if self.allowed_speakers and speaker_name not in self.allowed_speakers:
                continue

            self._voice_presets[speaker_name] = wav_file

            # 检查是否有对应的 .txt 文件
            txt_file = wav_file.with_suffix('.txt')
            if txt_file.exists():
                try:
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        self._voice_texts[speaker_name] = f.read().strip()
                except Exception as e:
                    print(f"Warning: Failed to read {txt_file}: {e}")

        # 按名称排序
        self._voice_presets = dict(sorted(self._voice_presets.items()))
        print(f"Found {len(self._voice_presets)} voice files in {self.base_dir}")

    def get_voice_path(self, speaker_name: str) -> Optional[str]:
        """
        获取声音文件路径

        Args:
            speaker_name: 说话人名称

        Returns:
            wav 文件路径，不存在则返回 None
        """
        # 精确匹配
        if speaker_name in self._voice_presets:
            return str(self._voice_presets[speaker_name])

        # 尝试部分匹配 (不区分大小写)
        speaker_lower = speaker_name.lower()
        for preset_name, path in self._voice_presets.items():
            if preset_name.lower() == speaker_lower:
                return str(path)

        return None

    def get_voice_text(self, speaker_name: str) -> Optional[str]:
        """获取声音对应的描述文本（可选）"""
        return self._voice_texts.get(speaker_name)

    def get_default_voice(self) -> Optional[str]:
        """获取默认声音（第一个可用的）"""
        if self._voice_presets:
            return str(list(self._voice_presets.values())[0])
        return None

    def list_voices(self) -> List[str]:
        """返回所有可用的声音名称列表"""
        return list(self._voice_presets.keys())

    def voice_exists(self, speaker_name: str) -> bool:
        """检查声音是否存在"""
        return self.get_voice_path(speaker_name) is not None


class MultiSpeakerParser:
    """解析多 speaker 文本，格式 "Speaker X: text" """

    # 匹配 "Speaker X:" 或 "{X}:" 格式
    SPEAKER_PATTERN = re.compile(r'^(?:Speaker\s*(\d+)|([^:]+)):\s*(.*)$', re.IGNORECASE)

    def __init__(self, voice_mapping: Optional[Dict[str, str]] = None):
        """
        初始化解析器

        Args:
            voice_mapping: Speaker ID 到 voice 文件的映射
                例如: {"Speaker 1": "zh-Aaron_man", "1": "en-Alice_woman"}
        """
        self.voice_mapping = voice_mapping or {}

    def parse(self, text: str) -> List[Dict]:
        """
        解析多 speaker 文本

        Args:
            text: 输入文本，如 "Speaker 1: Hello!\nSpeaker 2: Hi!"

        Returns:
            解析结果列表，每个元素包含:
            - speaker_id: Speaker ID (如 "1")
            - speaker_name: Speaker 名称 (如 "Speaker 1")
            - text: 说话内容
            - voice: 对应的声音文件路径 (如果在 voice_mapping 中)
        """
        lines = text.strip().split('\n')
        segments = []
        current_segment = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = self.SPEAKER_PATTERN.match(line)
            if match:
                # 保存上一个 segment
                if current_segment:
                    segments.append(current_segment)

                # 提取 speaker
                speaker_num = match.group(1)
                speaker_name = match.group(2)
                content = match.group(3)

                if speaker_num:
                    speaker_id = speaker_num
                    speaker_name = f"Speaker {speaker_num}"
                else:
                    speaker_id = speaker_name.strip()

                current_segment = {
                    'speaker_id': speaker_id,
                    'speaker_name': str(speaker_name).strip(),
                    'text': content.strip(),
                    'voice': self.voice_mapping.get(speaker_id) or self.voice_mapping.get(speaker_name)
                }
            else:
                # 继续上一行的内容
                if current_segment:
                    current_segment['text'] += ' ' + line
                else:
                    # 没有 speaker 的行，视为 Speaker 1
                    current_segment = {
                        'speaker_id': '1',
                        'speaker_name': 'Speaker 1',
                        'text': line,
                        'voice': self.voice_mapping.get('1') or self.voice_mapping.get('Speaker 1')
                    }

        # 添加最后一个 segment
        if current_segment:
            segments.append(current_segment)

        return segments

    @staticmethod
    def format_segment(speaker_id: str, text: str, speaker_name: Optional[str] = None) -> str:
        """格式化单个 segment 为字符串"""
        if speaker_name:
            return f"{speaker_name}: {text}"
        return f"Speaker {speaker_id}: {text}"

    def reconstruct(self, segments: List[Dict]) -> str:
        """将解析后的 segments 重新组合成字符串"""
        lines = []
        for seg in segments:
            name = seg.get('speaker_name') or f"Speaker {seg['speaker_id']}"
            lines.append(f"{name}: {seg['text']}")
        return '\n'.join(lines)
