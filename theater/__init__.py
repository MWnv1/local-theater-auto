# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
# Generated: 2026-11-07 09:31:28 +08:00 (Asia/Shanghai) [UTC 2026-11-07T01:31:28Z]
#
# Dual-licensed - choose exactly one:
#   (1) GNU AGPL-3.0-only  -> see LICENSE
#   (2) Commercial license -> see COMMERCIAL-LICENSE.md
# Decide with LICENSING.md. Under the AGPL-3.0 path, redistributing this
# software or exposing it as a network service obliges you to offer all users
# the complete corresponding source of your modified version (AGPL section 13).

"""本地图文小剧场自动化（OpenVINO 版）。

统一在此定义路径与公用小工具，避免各模块各自用相对路径导致缓存散落
（MCP Server 由 LobeChat 拉起时，工作目录并不确定）。
"""

import hashlib
import os
import re
import shutil

# ---- 路径配置 --------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_ROOT = os.environ.get("THEATER_CACHE", os.path.join(PROJECT_ROOT, "cache"))
MODEL_ROOT = os.environ.get("MODEL_ROOT", r"D:\ai_models")

SUBDIRS = ("assets", "audio", "frames", "video_clips", "logs")

# ffmpeg 的自动查找位置（按优先级）。装到这些位置任何一个都能直接用，
# 不必再设 FFMPEG_PATH —— 省掉"装好了却说找不到 ffmpeg"的坑。
_FFMPEG_NAME = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
_FFPROBE_NAME = "ffprobe.exe" if os.name == "nt" else "ffprobe"

_FFMPEG_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "tools", "ffmpeg", "bin", _FFMPEG_NAME),
    os.path.join(PROJECT_ROOT, "tools", "ffmpeg", _FFMPEG_NAME),
    os.path.join(PROJECT_ROOT, "_tools", "ffmpeg", "bin", _FFMPEG_NAME),
    os.path.join(PROJECT_ROOT, "_tools", "ffmpeg", _FFMPEG_NAME),
    r"D:\ffmpeg\bin\ffmpeg.exe",
    r"D:\ffmpeg\ffmpeg.exe",
)


def ensure_dirs() -> None:
    """创建缓存子目录。"""
    for d in SUBDIRS:
        os.makedirs(os.path.join(CACHE_ROOT, d), exist_ok=True)


def subdir(name: str) -> str:
    p = os.path.join(CACHE_ROOT, name)
    os.makedirs(p, exist_ok=True)
    return p


def ffmpeg_path() -> str:
    """ffmpeg 可执行文件。

    查找顺序：
      1. 环境变量 FFMPEG_PATH（显式指定优先）
      2. 项目内的常见安装位置（`tools/ffmpeg/bin`、`_tools/ffmpeg/bin`、`D:\\ffmpeg\\bin`）
      3. 系统 PATH

    找不到时返回 "ffmpeg"，由调用方给出可读错误。
    """
    explicit = os.environ.get("FFMPEG_PATH")
    if explicit:
        return explicit
    for cand in _FFMPEG_CANDIDATES:
        if os.path.exists(cand):
            return cand
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def ffprobe_path() -> str:
    """ffprobe 可执行文件。默认从 ffmpeg 同目录推导，可用 FFPROBE_PATH 覆盖。"""
    explicit = os.environ.get("FFPROBE_PATH")
    if explicit:
        return explicit
    exe = ffmpeg_path()
    d = os.path.dirname(exe)
    if d:
        name = _FFPROBE_NAME
        p = os.path.join(d, name)
        return p if os.path.exists(p) else name
    found = shutil.which("ffprobe")
    return found or "ffprobe"


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
