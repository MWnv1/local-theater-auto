# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
# Generated: 2026-11-07 09:31:28 +08:00 (Asia/Shanghai) [UTC 2026-11-07T01:31:28Z]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""离线系统配音（pyttsx3 / Windows SAPI5）。

修复的重点：
1. 原始稿在同一个 engine 上连续 `save_to_file` 再一次性 `runAndWait()`，
   Windows SAPI5 下经常只有第一句写出、其余是 0 字节（表现为后面镜头静音，
   且时长解析回落到默认 3.0s）。这里改为**每句独立新建 engine**。
2. 写出后校验文件大小，0 字节自动重试一次，仍失败则抛异常而不是静默返回。
3. 文件名带文本哈希，台词改了不会命中旧音频。
4. 支持按角色指定发音人（环境变量 TTS_VOICE_MAP），便于区分人物音色。
"""

import os
import threading
from typing import Dict, Optional

from . import subdir, safe_name, text_key

_AUDIO_DIR = subdir("audio")
_lock = threading.Lock()

TTS_RATE = int(os.environ.get("TTS_RATE", "180"))
TTS_VOICE = os.environ.get("TTS_VOICE", "").strip()
# 形如 "小林=Microsoft Huihui,阿杰=Microsoft Kangkang"
_RAW_MAP = os.environ.get("TTS_VOICE_MAP", "").strip()
VOICE_MAP: Dict[str, str] = {}
if _RAW_MAP:
    for pair in _RAW_MAP.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            VOICE_MAP[k.strip()] = v.strip()


def _pick_voice(engine, hint: Optional[str]) -> Optional[str]:
    """在系统已安装发音人里按名字片段模糊匹配。"""
    if not hint:
        return None
    try:
        voices = engine.getProperty("voices") or []
    except Exception:
        return None
    hint_lower = hint.lower()
    for v in voices:
        name = (getattr(v, "name", "") or "").lower()
        vid = (getattr(v, "id", "") or "").lower()
        if hint_lower in name or hint_lower in vid:
            return v.id
    return None


def _synthesize_once(text: str, out_path: str, voice_hint: Optional[str], rate: int) -> None:
    import pyttsx3  # 延迟导入

    engine = pyttsx3.init()
    try:
        engine.setProperty("rate", rate)
        vid = _pick_voice(engine, voice_hint)
        if vid:
            engine.setProperty("voice", vid)
        if os.path.exists(out_path):
            os.remove(out_path)
        engine.save_to_file(text, out_path)
        engine.runAndWait()
    finally:
        try:
            engine.stop()
        except Exception:
            pass


def _text_to_audio(text: str, filename: str, character: str = "", force: bool = False) -> str:
    """单句合成，返回绝对路径（带 0 字节校验与一次重试）。"""
    out_path = os.path.join(_AUDIO_DIR, filename)
    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return os.path.abspath(out_path)

    voice_hint = VOICE_MAP.get(character) or TTS_VOICE or None
    with _lock:
        _synthesize_once(text, out_path, voice_hint, TTS_RATE)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            # 重试：换默认发音人、稍慢语速
            _synthesize_once(text, out_path, None, TTS_RATE)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            raise RuntimeError(
                f"TTS 未生成有效音频：{out_path}（文本：{text[:40]}）。"
                "请检查系统语音包是否安装（设置 → 时间和语言 → 语音）。"
            )
    return os.path.abspath(out_path)


def gen_all_tts(script_struct: dict, force: bool = False) -> Dict:
    """批量生成台词音频。

    返回：{scene_id: [{"character","text","audio_path","index"}, ...]}
    """
    audio_map: Dict[int, list] = {}
    for scene in script_struct.get("scenes", []):
        scene_id = int(scene["scene_id"])
        scene_audios = []
        for idx, line in enumerate(scene.get("lines", [])):
            text = (line.get("text") or "").strip()
            if not text:
                continue
            char = line.get("character", "旁白")
            fname = f"scene{scene_id}_{idx}_{text_key(char + text)}.wav"
            path = _text_to_audio(text, fname, character=char, force=force)
            scene_audios.append(
                {
                    "character": char,
                    "text": text,
                    "audio_path": path,
                    "index": idx,
                }
            )
        audio_map[scene_id] = scene_audios
    return audio_map
