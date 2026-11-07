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

"""本地图文小剧场自动化（OpenVINO 版）。

统一在此定义路径与公用小工具，避免各模块各自用相对路径导致缓存散落
（MCP Server 由 LobeChat 拉起时，工作目录并不确定）。
"""

import hashlib
import os
import re

# ---- 路径配置 --------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.environ.get("THEATER_CACHE", os.path.join(PROJECT_ROOT, "cache"))
MODEL_ROOT = os.environ.get("MODEL_ROOT", r"D:\ai_models")

SUBDIRS = ("assets", "audio", "frames", "video_clips", "logs")


def ensure_dirs() -> None:
    """创建缓存子目录。"""
    for d in SUBDIRS:
        os.makedirs(os.path.join(CACHE_ROOT, d), exist_ok=True)


def subdir(name: str) -> str:
    p = os.path.join(CACHE_ROOT, name)
    os.makedirs(p, exist_ok=True)
    return p


def ffmpeg_path() -> str:
    """ffmpeg 可执行文件。默认取 PATH，可用 FFMPEG_PATH 覆盖。"""
    return os.environ.get("FFMPEG_PATH", "ffmpeg")


def ffprobe_path() -> str:
    """ffprobe 可执行文件。默认从 ffmpeg 同目录推导，可用 FFPROBE_PATH 覆盖。"""
    explicit = os.environ.get("FFPROBE_PATH")
    if explicit:
        return explicit
    exe = ffmpeg_path()
    d = os.path.dirname(exe)
    if d:
        name = "ffprobe.exe" if exe.lower().endswith(".exe") else "ffprobe"
        return os.path.join(d, name)
    return "ffprobe"


# ---- 名称处理 --------------------------------------------------------------
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_name(raw: str, fallback: str = "item") -> str:
    """把角色名等任意文本转成可安全用作文件名的字符串。

    保留中文（Windows 支持），去掉非法字符与首尾点/空格，
    过长时用内容哈希做后缀防止碰撞。
    """
    text = _BAD_CHARS.sub("_", str(raw)).strip().strip(".")
    text = re.sub(r"_+", "_", text)
    if not text:
        text = fallback
    if len(text) > 40:
        digest = hashlib.md5(str(raw).encode("utf-8")).hexdigest()[:8]
        text = text[:32] + "_" + digest
    return text


def text_key(text: str) -> str:
    """短文本的稳定哈希键，用于缓存命中判断。"""
    return hashlib.md5(str(text).encode("utf-8")).hexdigest()[:10]
