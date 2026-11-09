# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
# Generated: 2026-11-09 (Asia/Shanghai)
#
# Dual-licensed - choose exactly one:
#   (1) GNU AGPL-3.0-only  -> see LICENSE
#   (2) Commercial license -> see COMMERCIAL-LICENSE.md
# Decide with LICENSING.md. Under the AGPL-3.0 path, redistributing this
# software or exposing it as a network service obliges you to offer all users
# the complete corresponding source of your modified version (AGPL section 13).

"""立绘自动骨骼绑定（2D 伪骨骼）。

为什么不用 Live2D / Spine：两者的运行库都有授权限制，且需要手工在编辑器里
打点。小剧场要的是"丢一张 FLUX 生成的角色立绘进去，它就能点头、挥手、走路"。

做法是**解剖先验的自动切分**：
  1. 用 alpha 通道求出身体真实包围盒（去掉四周透明留白）；
  2. 按人体比例把包围盒横向/纵向切成 头 / 躯干 / 左臂 / 右臂 / 左腿 / 右腿；
  3. 每个部件记录 bbox + 旋转轴（pivot，取关节所在位置）+ z 叠放序；
  4. 关节（颈、腰、双肩、双髋）按包围盒比例定位，供动画器驱动。

输出 `rig.json`，`animator.py` 读它做逐帧变换。切分结果是**可复现**的：
同一张图两次 build 得到同一 signature。
"""

import hashlib
import json
import os
from typing import Dict, Optional

from PIL import Image

from . import safe_name, subdir

RIG_VERSION = 1

# 人体比例（相对身体包围盒高度/宽度），来自真人立绘的经验值，够用即可
_HEAD_BOTTOM = 0.255
_SHOULDER_TOP = 0.265
_WAIST_TOP = 0.55
_TORSO_BOTTOM = 0.62
_ARM_HALF_WIDTH = 0.235   # 手臂占身体宽度的比例（两侧各一份）
_LEG_SPLIT = 0.06         # 两条腿之间的中缝宽度（占身体宽度）

DEFAULT_PARTS = ("head", "torso", "arm_l", "arm_r", "leg_l", "leg_r")


def _alpha_bbox(img: Image.Image) -> tuple:
    """alpha>0 的真实包围盒；全透明时退回整图。"""
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bbox = img.getchannel("A").getbbox()
    return bbox or (0, 0, img.width, img.height)


def build_rig(char_png: str, out_dir: Optional[str] = None) -> dict:
    """给一张透明立绘建立骨骼，返回 rig 字典并落盘 rig.json。"""
    img = Image.open(char_png).convert("RGBA")
    x0, y0, x1, y1 = _alpha_bbox(img)
    bw, bh = x1 - x0, y1 - y0
    if bw < 8 or bh < 8:
        raise ValueError(f"立绘有效像素太小，无法绑定骨骼：{char_png}")

    def px(rx: float) -> int:
        return int(x0 + bw * rx)

    def py(ry: float) -> int:
        return int(y0 + bh * ry)

    parts: Dict[str, dict] = {}

    # 头：包围盒上部，横向留一点余量（发型可能比脸宽）
    parts["head"] = {
        "bbox": [px(0.08), py(0.0), px(0.92), py(_HEAD_BOTTOM)],
        "pivot": [px(0.50), py(_HEAD_BOTTOM)],
        "z": 5,
    }
    # 躯干
    parts["torso"] = {
        "bbox": [px(_ARM_HALF_WIDTH), py(_SHOULDER_TOP), px(1 - _ARM_HALF_WIDTH), py(_TORSO_BOTTOM)],
        "pivot": [px(0.50), py(_WAIST_TOP)],
        "z": 3,
    }
    # 双臂（肩点作旋转轴）
    parts["arm_l"] = {
        "bbox": [px(0.02), py(_SHOULDER_TOP + 0.01), px(_ARM_HALF_WIDTH + 0.02), py(_WAIST_TOP)],
        "pivot": [px(0.10), py(_SHOULDER_TOP + 0.02)],
        "z": 2,
    }
    parts["arm_r"] = {
        "bbox": [px(1 - _ARM_HALF_WIDTH - 0.02), py(_SHOULDER_TOP + 0.01), px(0.98), py(_WAIST_TOP)],
        "pivot": [px(0.90), py(_SHOULDER_TOP + 0.02)],
        "z": 2,
    }
    # 双腿（髋点作旋转轴）
    parts["leg_l"] = {
        "bbox": [px(0.10), py(_TORSO_BOTTOM - 0.02), px(0.50 - _LEG_SPLIT), py(1.0)],
        "pivot": [px(0.28), py(_TORSO_BOTTOM)],
        "z": 1,
    }
    parts["leg_r"] = {
        "bbox": [px(0.50 + _LEG_SPLIT), py(_TORSO_BOTTOM - 0.02), px(0.90), py(1.0)],
        "pivot": [px(0.72), py(_TORSO_BOTTOM)],
        "z": 1,
    }

    for p in parts.values():
        bx = p["bbox"]
        bx[0] = max(0, bx[0])
        bx[1] = max(0, bx[1])
        bx[2] = min(img.width, bx[2])
        bx[3] = min(img.height, bx[3])
        if bx[2] <= bx[0] or bx[3] <= bx[1]:
            raise ValueError(f"部件切分失败（bbox 退化）：{bx}")

    joints = {
        "neck": [px(0.50), py(_HEAD_BOTTOM)],
        "waist": [px(0.50), py(_WAIST_TOP)],
        "shoulder_l": [px(0.18), py(_SHOULDER_TOP + 0.02)],
        "shoulder_r": [px(0.82), py(_SHOULDER_TOP + 0.02)],
        "hip_l": [px(0.28), py(_TORSO_BOTTOM)],
        "hip_r": [px(0.72), py(_TORSO_BOTTOM)],
        "center": [px(0.50), py(0.50)],
    }

    sig_src = f"{os.path.basename(char_png)}|{img.width}x{img.height}|{x0},{y0},{x1},{y1}|v{RIG_VERSION}"
    signature = hashlib.md5(sig_src.encode("utf-8")).hexdigest()[:12]

    out_dir = out_dir or subdir("rigs")
    os.makedirs(out_dir, exist_ok=True)
    rig_path = os.path.join(out_dir, f"{safe_name(os.path.splitext(os.path.basename(char_png))[0], 'char')}_{signature}.json")

    rig = {
        "version": RIG_VERSION,
        "image": os.path.abspath(char_png),
        "size": [img.width, img.height],
        "body_bbox": [x0, y0, x1, y1],
        "parts": parts,
        "joints": joints,
        "signature": signature,
        "rig_path": os.path.abspath(rig_path),
    }
    with open(rig_path, "w", encoding="utf-8") as f:
        json.dump(rig, f, ensure_ascii=False, indent=2)
    return rig


def load_rig(path: str) -> dict:
    """读回 rig.json（校验版本，避免旧结构静默出错）。"""
    with open(path, "r", encoding="utf-8") as f:
        rig = json.load(f)
    if int(rig.get("version", 0)) != RIG_VERSION:
        raise ValueError(f"rig 版本不匹配：{rig.get('version')} != {RIG_VERSION}")
    if not os.path.exists(rig.get("image", "")):
        raise FileNotFoundError(f"rig 指向的立绘不存在：{rig.get('image')}")
    return rig


def rig_for(char_png: str, out_dir: Optional[str] = None) -> dict:
    """有缓存就复用，没有就现建（按立绘路径+尺寸签名）。"""
    out_dir = out_dir or subdir("rigs")
    os.makedirs(out_dir, exist_ok=True)
    stem = safe_name(os.path.splitext(os.path.basename(char_png))[0], "char")
    for fn in os.listdir(out_dir):
        if fn.startswith(stem + "_") and fn.endswith(".json"):
            try:
                rig = load_rig(os.path.join(out_dir, fn))
                if os.path.abspath(rig["image"]) == os.path.abspath(char_png):
                    return rig
            except Exception:
                continue
    return build_rig(char_png, out_dir=out_dir)


def describe(rig: dict) -> str:
    """人读摘要（MCP 工具里直接给用户看）。"""
    parts = ", ".join(f"{k}{tuple(v['bbox'])}" for k, v in sorted(rig["parts"].items()))
    return (f"骨骼 {rig['signature']}：{len(rig['parts'])} 个部件，"
            f"{len(rig['joints'])} 个关节，立绘 {rig['size'][0]}x{rig['size'][1]}；{parts}")
