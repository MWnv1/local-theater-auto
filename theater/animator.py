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

"""骨骼动画驱动：把 rigging 出来的部件按动作参数逐帧变换。

动作是**纯函数式**的：`pose_at(action, t)` 只依赖时间，同一 t 永远同一姿态，
所以帧序列可以任意重算、并行、断点续跑，也不需要存关键帧文件。

五个内置动作：
  idle   呼吸（躯干轻微缩放 + 头部上下漂移），默认待机
  nod    点头（头部绕颈关节旋转）
  wave   挥手（右臂绕肩关节来回摆）
  walk   走路（两腿交替前后摆 + 躯干上下起伏 + 手臂反向摆）
  talk   说话（头部小幅点动 + 双臂轻微开合，配音时用）

渲染方式：每个部件裁出 bbox → 绕 pivot 旋转 → 加平移 → 按 z 叠到透明画布。
"""

import json
import math
import os
from typing import Dict, List, Optional

from PIL import Image

from . import subdir

ACTIONS: Dict[str, dict] = {
    "idle": {"period": 3.2, "amp": 0.6},
    "nod": {"period": 1.4, "amp": 1.0},
    "wave": {"period": 1.6, "amp": 1.0},
    "walk": {"period": 1.1, "amp": 1.0},
    "talk": {"period": 0.9, "amp": 0.8},
}

DEFAULT_FPS = int(os.environ.get("ANIM_FPS", "24"))

_IMAGE_CACHE: Dict[str, Image.Image] = {}


def _load_image(rig: dict) -> Image.Image:
    path = rig["image"]
    key = f"{path}|{os.path.getsize(path) if os.path.exists(path) else 0}"
    if key not in _IMAGE_CACHE:
        _IMAGE_CACHE[key] = Image.open(path).convert("RGBA")
    return _IMAGE_CACHE[key]


def pose_at(action: str, t: float) -> dict:
    """给定动作与时间，返回每个部件的旋转/位移（度、像素）。确定性纯函数。"""
    if action not in ACTIONS:
        action = "idle"
    p = ACTIONS[action]["period"]
    w = 2 * math.pi * t / p
    pose: Dict[str, dict] = {}

    if action == "idle":
        breathe = math.sin(w)
        pose["head"] = {"rot": 1.2 * math.sin(w * 0.5), "dx": 0.0, "dy": 1.5 * breathe}
        pose["torso"] = {"rot": 0.0, "dx": 0.0, "dy": 1.2 * breathe}
        pose["arm_l"] = {"rot": -1.5 + 1.0 * breathe, "dx": 0.0, "dy": 0.0}
        pose["arm_r"] = {"rot": 1.5 - 1.0 * breathe, "dx": 0.0, "dy": 0.0}
    elif action == "nod":
        n = math.sin(w)
        pose["head"] = {"rot": 7.0 * n, "dx": 0.0, "dy": 2.0 * abs(n)}
        pose["torso"] = {"rot": 0.0, "dx": 0.0, "dy": 0.8 * abs(n)}
    elif action == "wave":
        s = math.sin(w)
        pose["arm_r"] = {"rot": -38.0 - 26.0 * s, "dx": 3.0 * s, "dy": -2.0 * s}
        pose["arm_l"] = {"rot": 3.0, "dx": 0.0, "dy": 0.0}
        pose["head"] = {"rot": 3.5 * math.sin(w * 0.5), "dx": 0.0, "dy": 0.0}
        pose["torso"] = {"rot": -2.0, "dx": 0.0, "dy": 0.0}
    elif action == "walk":
        s = math.sin(w)
        pose["leg_l"] = {"rot": 17.0 * s, "dx": 0.0, "dy": 0.0}
        pose["leg_r"] = {"rot": -17.0 * s, "dx": 0.0, "dy": 0.0}
        pose["arm_l"] = {"rot": -12.0 * s, "dx": 0.0, "dy": 0.0}
        pose["arm_r"] = {"rot": 12.0 * s, "dx": 0.0, "dy": 0.0}
        pose["torso"] = {"rot": 0.0, "dx": 0.0, "dy": -4.0 * abs(math.sin(w))}
        pose["head"] = {"rot": 1.5 * s, "dx": 0.0, "dy": -2.0 * abs(math.sin(w))}
    elif action == "talk":
        s = math.sin(w)
        s2 = math.sin(w * 2.3)
        pose["head"] = {"rot": 2.5 * s, "dx": 0.0, "dy": 1.2 * s2}
        pose["arm_l"] = {"rot": -6.0 - 4.0 * s, "dx": 0.0, "dy": 0.0}
        pose["arm_r"] = {"rot": 6.0 + 4.0 * s, "dx": 0.0, "dy": 0.0}
        pose["torso"] = {"rot": 0.6 * s, "dx": 0.0, "dy": 0.0}
    return pose


def _rot_part(part_img: Image.Image, angle: float) -> Image.Image:
    if abs(angle) < 0.05:
        return part_img
    return part_img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)


def render_frame(rig: dict, action: str, t: float,
                 size: Optional[tuple] = None, background: Optional[Image.Image] = None) -> Image.Image:
    """渲染 t 时刻的一帧（透明 RGBA，或叠到给定背景上）。"""
    src = _load_image(rig)
    canvas = Image.new("RGBA", (src.width, src.height), (0, 0, 0, 0))
    pose = pose_at(action, t)

    for name in sorted(rig["parts"], key=lambda k: rig["parts"][k]["z"]):
        part = rig["parts"][name]
        bx = [int(v) for v in part["bbox"]]
        pv = [int(v) for v in part["pivot"]]
        crop = src.crop(tuple(bx))
        # pivot 换算成 crop 内的相对坐标
        rx = max(0, min(crop.width - 1, pv[0] - bx[0]))
        ry = max(0, min(crop.height - 1, pv[1] - bx[1]))
        moved = pose.get(name, {})

        # 以关节为轴旋转：先把部件扩成"以 pivot 为中心"的方画布，
        # rotate(expand=False) 绕中心转就不会把关节位置带偏。
        half_w = max(rx, crop.width - rx)
        half_h = max(ry, crop.height - ry)
        pad = Image.new("RGBA", (half_w * 2, half_h * 2), (0, 0, 0, 0))
        pad.paste(crop, (half_w - rx, half_h - ry))
        turned = _rot_part(pad, moved.get("rot", 0.0))

        dst = (int(bx[0] + rx - half_w) + int(moved.get("dx", 0)),
               int(bx[1] + ry - half_h) + int(moved.get("dy", 0)))
        canvas.alpha_composite(turned, (dst[0], dst[1]))

    if size and (int(size[0]), int(size[1])) != (canvas.width, canvas.height):
        canvas = canvas.resize((int(size[0]), int(size[1])), Image.Resampling.LANCZOS)
    if background is not None:
        bg = background.convert("RGBA")
        if bg.size != canvas.size:
            bg = bg.resize(canvas.size, Image.Resampling.LANCZOS)
        bg.alpha_composite(canvas)
        return bg
    return canvas


def frame_diff(a: Image.Image, b: Image.Image) -> float:
    """两帧差异比例（0~1）：非零差异像素占两帧 alpha 并集的比重。"""
    import numpy as np

    if a.size != b.size:
        b = b.resize(a.size, Image.Resampling.LANCZOS)
    aa = np.asarray(a.convert("RGBA"), dtype=np.int16)
    bb = np.asarray(b.convert("RGBA"), dtype=np.int16)
    diff = np.abs(aa - bb).sum(axis=2) > 6
    union = (aa[:, :, 3] > 0) | (bb[:, :, 3] > 0)
    denom = max(1, int(union.sum()))
    return float(diff.sum()) / float(denom)


def render_clip(rig: dict, action: str, seconds: float, fps: int = DEFAULT_FPS,
                out_dir: Optional[str] = None, size: Optional[tuple] = None) -> dict:
    """渲染 PNG 序列 + manifest（manifest 里留关节轨，便于后期换帧率/加特效）。"""
    out_dir = out_dir or subdir("anim")
    os.makedirs(out_dir, exist_ok=True)
    frames = max(1, int(round(seconds * fps)))
    paths: List[str] = []
    tracks: List[dict] = []
    for i in range(frames):
        t = i / float(fps)
        img = render_frame(rig, action, t, size=size)
        p = os.path.join(out_dir, f"{action}_{i:04d}.png")
        img.save(p)
        paths.append(os.path.abspath(p))
        tracks.append({"t": round(t, 3), "pose": pose_at(action, t)})

    manifest = os.path.join(out_dir, "clip_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"action": action, "fps": int(fps), "frames": frames,
                   "seconds": round(seconds, 3), "size": list(size) if size else None,
                   "rig": rig.get("signature"), "files": paths,
                   "joint_tracks": tracks}, f, ensure_ascii=False, indent=2)
    return {"dir": os.path.abspath(out_dir), "frames": frames, "fps": int(fps),
            "manifest": os.path.abspath(manifest), "files": paths}


def render_still_grid(rig: dict, action: str, times: List[float]) -> List[Image.Image]:
    """调试用：一次渲染多个时刻（不落盘）。"""
    return [render_frame(rig, action, t) for t in times]
