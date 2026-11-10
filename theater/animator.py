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

"""人物动作驱动：整张立绘做平移 / 旋转。

角色只需要三种自由度——左右平移、上下平移、旋转。整图仿射比"按 alpha 连通域
切成头/躯干/四肢再绕关节转"稳得多：切分依赖抠图质量，立绘边缘毛一点就会把
胳膊切成两半，旋转时露出裂缝；而整图平移/旋转在任何立绘上都成立，也不挑素材。

动作仍是**纯函数式**的：`pose_at(action, t)` 只依赖时间，返回归一化位移
（占图片宽/高的比例）与旋转角度，同一 t 永远同一结果，所以帧序列可以任意重算、
并行渲染、断点续跑，不需要关键帧文件。

内置动作：
  pan_left / pan_right   左右往返平移
  pan_up   / pan_down    上下往返平移
  rotate                 小幅往返旋转（±10°）
  spin                   持续同向旋转（360°/周期）
  idle / nod / wave / walk / talk
                         旧名字保留成小幅度平移/旋转的别名，老脚本与管线不用改
"""

import json
import math
import os
from typing import Dict, List, Optional

from PIL import Image

from . import subdir

ACTIONS: Dict[str, dict] = {
    "pan_left": {"period": 4.0, "amp_x": -0.18},
    "pan_right": {"period": 4.0, "amp_x": 0.18},
    "pan_up": {"period": 4.0, "amp_y": -0.16},
    "pan_down": {"period": 4.0, "amp_y": 0.16},
    "rotate": {"period": 3.0, "rot": 10.0},
    "spin": {"period": 6.0, "spin": 360.0},
    # 以下旧动作名向后兼容，全部退化成小幅度平移/旋转
    "idle": {"period": 3.2, "amp_y": 0.012},
    "nod": {"period": 1.4, "amp_y": 0.03},
    "wave": {"period": 1.6, "amp_x": 0.05},
    "walk": {"period": 1.1, "amp_x": 0.09, "amp_y": 0.02},
    "talk": {"period": 0.9, "rot": 1.6},
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
    """给定动作与时刻，返回归一化变换。

    dx / dy 是**相对图片宽、高**的比例（-0.18 表示"向左移 18% 宽度"），
    rot 是角度。纯函数：同样的 (action, t) 永远同一个 dict。
    """
    if action not in ACTIONS:
        action = "idle"
    cfg = ACTIONS[action]
    period = float(cfg.get("period") or 1.0)
    w = 2.0 * math.pi * t / period
    s = math.sin(w)
    dx = float(cfg.get("amp_x", 0.0)) * s
    dy = float(cfg.get("amp_y", 0.0)) * s
    rot = float(cfg.get("rot", 0.0)) * s
    spin = float(cfg.get("spin", 0.0))
    if spin:
        rot = spin * ((t / period) % 1.0)
    return {"dx": round(dx, 5), "dy": round(dy, 5), "rot": round(rot, 3)}


def warp(img: Image.Image, dx: float, dy: float, rot: float = 0.0) -> Image.Image:
    """绕图片中心旋转 rot 度，再整体平移 (dx, dy) 像素，返回同尺寸 RGBA。

    用 AFFINE 一次做完（而不是先 rotate 再 paste），亚像素位移不会因为取整
    而抖动；出界区域填全透明，不会出现边缘拉丝的 smeared 边。
    """
    if abs(rot) < 1e-6 and abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return img
    w, h = img.size
    cx, cy = w / 2.0, h / 2.0
    th = math.radians(rot)
    cos_t, sin_t = math.cos(th), math.sin(th)
    a, b = cos_t, sin_t
    d, e = -sin_t, cos_t
    c = cx - cos_t * (cx + dx) - sin_t * (cy + dy)
    f = cy + sin_t * (cx + dx) - cos_t * (cy + dy)
    return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f),
                         resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0, 0))


def transform_px(pose: dict, size) -> Dict[str, float]:
    """把归一化 pose 换算成该尺寸下的像素位移。"""
    w, h = int(size[0]), int(size[1])
    return {"dx": round(float(pose.get("dx", 0.0)) * w, 3),
            "dy": round(float(pose.get("dy", 0.0)) * h, 3),
            "rot": float(pose.get("rot", 0.0))}


def render_frame(rig: dict, action: str, t: float,
                 size: Optional[tuple] = None, background: Optional[Image.Image] = None) -> Image.Image:
    """渲染 t 时刻的一帧（透明 RGBA，或叠到给定背景上）。"""
    src = _load_image(rig)
    pose = pose_at(action, t)
    canvas = warp(src, float(pose["dx"]) * src.width, float(pose["dy"]) * src.height,
                  float(pose["rot"]))
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
    """渲染 PNG 序列 + manifest（manifest 留每帧变换轨，便于后期换帧率/加特效）。"""
    out_dir = out_dir or subdir("anim")
    os.makedirs(out_dir, exist_ok=True)
    src = _load_image(rig)
    frames = max(1, int(round(seconds * fps)))
    paths: List[str] = []
    tracks: List[dict] = []
    for i in range(frames):
        t = i / float(fps)
        img = render_frame(rig, action, t, size=size)
        p = os.path.join(out_dir, f"{action}_{i:04d}.png")
        img.save(p)
        paths.append(os.path.abspath(p))
        pose = pose_at(action, t)
        tracks.append({"t": round(t, 3), "pose": pose,
                       "px": transform_px(pose, src.size)})

    manifest = os.path.join(out_dir, "clip_manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({"action": action, "fps": int(fps), "frames": frames,
                   "seconds": round(seconds, 3), "size": list(size) if size else None,
                   "rig": rig.get("signature"), "files": paths,
                   "motion_tracks": tracks, "joint_tracks": tracks},
                  f, ensure_ascii=False, indent=2)
    return {"dir": os.path.abspath(out_dir), "frames": frames, "fps": int(fps),
            "manifest": os.path.abspath(manifest), "files": paths}


def render_still_grid(rig: dict, action: str, times: List[float]) -> List[Image.Image]:
    """调试用：一次渲染多个时刻（不落盘）。"""
    return [render_frame(rig, action, t) for t in times]