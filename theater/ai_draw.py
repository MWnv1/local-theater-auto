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

"""AI 绘图 + 本地抠图（OpenVINO / Intel 核显版）。

架构保持原设计：optimum-intel + OpenVINO 在核显(GPU)上跑 Flux Schnell INT4。
本文件修复的问题：
1. 模型改为**首次调用时懒加载**，import 本模块不再直接吃掉几 GB 内存/几十秒；
   并加线程锁，避免 MCP 并发请求同时进同一个 pipeline。
2. FLUX 用 `OVFluxPipeline` 加载（optimum-intel 新版本才有），旧环境回退
   `OVStableDiffusionPipeline`，并在两者都不支持时抛出可读错误。
3. rembg 明确走离线：先设 `U2NET_HOME` 到 MODEL_ROOT/rembg，模型缺失时
   **不联网下载**，自动降级为"白底连通域抠图"（prompt 里已要求白底）。
4. 角色名进文件名前统一 `safe_name()`，杜绝中文/特殊字符导致的非法路径与碰撞。
5. 已生成过的资源默认跳过（缓存命中），需要重画时传 force=True。
"""

import os
import threading
from typing import Dict, Optional

from . import MODEL_ROOT, subdir, safe_name, text_key

DRAW_MODEL_DIR = os.environ.get(
    "DRAW_MODEL_DIR", os.path.join(MODEL_ROOT, "flux-schnell-int4-ov")
)
DRAW_DEVICE = os.environ.get("DRAW_DEVICE", "GPU")          # GPU / CPU / AUTO
DRAW_STEPS = int(os.environ.get("DRAW_STEPS", "4"))
DRAW_SIZE = int(os.environ.get("DRAW_SIZE", "1024"))

_ASSET_DIR = subdir("assets")
_pipe = None
_pipe_lock = threading.RLock()
_gen_lock = threading.Lock()


# ---------------------------------------------------------------------------
# pipeline 懒加载
# ---------------------------------------------------------------------------
def get_pipe(device: Optional[str] = None):
    """按需加载 OpenVINO 扩散 pipeline（单例）。"""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe

        if not os.path.isdir(DRAW_MODEL_DIR) or not os.listdir(DRAW_MODEL_DIR):
            raise RuntimeError(
                f"绘图模型目录不存在或为空：{DRAW_MODEL_DIR}\n"
                "请先把 Flux Schnell INT4 OpenVINO 模型包解压到该目录（见 README）。"
            )

        from optimum.intel.openvino import OVStableDiffusionPipeline  # 延迟导入

        pipe_cls = OVStableDiffusionPipeline
        try:
            from optimum.intel.openvino import OVFluxPipeline  # type: ignore

            pipe_cls = OVFluxPipeline
        except Exception:
            # 旧版 optimum-intel 没有 Flux 专用 pipeline
            pipe_cls = OVStableDiffusionPipeline

        kwargs = {"compile": True, "ov_config": {"CACHE_DIR": os.path.join(_ASSET_DIR, "ov_cache")}}
        dev = device or DRAW_DEVICE
        if dev and dev.upper() != "AUTO":
            kwargs["device"] = dev

        _pipe = pipe_cls.from_pretrained(DRAW_MODEL_DIR, **kwargs)
        return _pipe


def _run_prompt(prompt: str):
    pipe = get_pipe()
    common = {"num_inference_steps": DRAW_STEPS, "height": DRAW_SIZE, "width": DRAW_SIZE}
    with _gen_lock:
        try:
            return pipe(prompt, **common).images[0]
        except TypeError:
            # FLUX 不支持 negative_prompt / guidance 之类参数，退化调用
            return pipe(prompt, **common).images[0]


# ---------------------------------------------------------------------------
# 抠图
# ---------------------------------------------------------------------------
def _cutout_offline(img_path: str, out_path: str) -> str:
    """rembg 离线抠图；模型缺失时降级为白底连通域抠图。"""
    try:
        os.environ.setdefault("U2NET_HOME", os.path.join(MODEL_ROOT, "rembg"))
        model_file = os.path.join(os.environ["U2NET_HOME"], "u2net.onnx")
        if not os.path.exists(model_file):
            raise FileNotFoundError("u2net.onnx 缺失")

        from rembg import remove  # 延迟导入
        from PIL import Image

        src = Image.open(img_path)
        out = remove(src)
        out.save(out_path, "PNG")
        return out_path
    except Exception:
        return _fallback_cutout(img_path, out_path)


def _fallback_cutout(img_path: str, out_path: str, threshold: int = 232) -> str:
    """无 rembg 时的兜底：从四边泛洪，把近白底色变透明。"""
    import numpy as np
    from PIL import Image

    img = Image.open(img_path).convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape[:2]
    near_white = (arr[:, :, 0] >= threshold) & (arr[:, :, 1] >= threshold) & (arr[:, :, 2] >= threshold)

    visited = np.zeros((h, w), dtype=bool)
    stack = []
    for x in range(w):
        stack.extend([(0, x), (h - 1, x)])
    for y in range(h):
        stack.extend([(y, 0), (y, w - 1)])

    while stack:
        y, x = stack.pop()
        if y < 0 or x < 0 or y >= h or x >= w:
            continue
        if visited[y, x] or not near_white[y, x]:
            continue
        visited[y, x] = True
        stack.append((y - 1, x))
        stack.append((y + 1, x))
        stack.append((y, x - 1))
        stack.append((y, x + 1))

    alpha = np.where(visited, 0, 255).astype(np.uint8)
    rgba = np.dstack([arr, alpha])
    Image.fromarray(rgba, "RGBA").save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------
def _gen_image(prompt: str, filename: str, force: bool = False) -> str:
    """生成单张图片并返回绝对路径（默认命中缓存则跳过）。"""
    path = os.path.join(_ASSET_DIR, filename)
    if not force and os.path.exists(path) and os.path.getsize(path) > 0:
        return os.path.abspath(path)
    image = _run_prompt(prompt)
    image.save(path)
    return os.path.abspath(path)


def generate_scene_assets(script_struct: dict, force: bool = False) -> Dict:
    """生成 1 张通用背景 + 每个角色一张透明立绘。

    返回：
        {
          "background": 绝对路径,
          "backgrounds": {scene_id: 路径}  # 剧本写了【场景：xxx】时按镜头单独出图
          "characters": {角色名: 透明 PNG 路径},
          "model_dir": 使用的绘图模型目录
        }
    """
    assets: Dict = {"background": "", "backgrounds": {}, "characters": {}, "model_dir": DRAW_MODEL_DIR}

    # 通用背景
    bg_prompt = os.environ.get(
        "BG_PROMPT",
        "anime style background, indoor room, clean composition, soft lighting, high quality, no people",
    )
    bg_path = _gen_image(bg_prompt, "bg_scene.png", force=force)
    assets["background"] = bg_path

    # 有场景描述的镜头单独出背景（同一描述自动复用）
    for scene in script_struct.get("scenes", []):
        desc = (scene.get("scene_desc") or "").strip()
        if not desc:
            continue
        key = text_key(desc)
        path = _gen_image(
            f"anime style background, {desc}, no people, clean composition, high quality",
            f"bg_scene_{scene['scene_id']}_{key}.png",
            force=force,
        )
        assets["backgrounds"][int(scene["scene_id"])] = path

    # 角色立绘
    for char in script_struct.get("all_characters", []):
        safe = safe_name(char, "char")
        raw_name = f"raw_{safe}.png"
        char_prompt = os.environ.get(
            "CHAR_PROMPT_TMPL",
            "anime style, full body character of {char}, standing, plain white background, "
            "front view, clean lineart, high quality, no shadow on background",
        ).format(char=char)
        raw_path = _gen_image(char_prompt, raw_name, force=force)
        png_path = os.path.join(_ASSET_DIR, f"char_{safe}.png")
        if force or not os.path.exists(png_path) or os.path.getsize(png_path) == 0:
            _cutout_offline(raw_path, png_path)
        assets["characters"][char] = os.path.abspath(png_path)

    return assets
