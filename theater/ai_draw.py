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

"""AI 绘图 + 本地抠图（OpenVINO / PyTorch 双后端，NVIDIA / AMD / Intel 通用）。

后端选择（`DRAW_BACKEND=auto` 时自动判定，见 `theater/hardware.py`）：
    NVIDIA + torch          -> torch / cuda      （最快）
    AMD + torch-directml    -> torch / dml       （Windows 上的 AMD 加速路径）
    Intel（核显 / Arc）     -> openvino / GPU
    其他                    -> openvino / CPU    （任何平台都能跑，只是慢）

原有修复仍然保留：
1. 模型**首次调用时懒加载**，import 本模块不吃内存；线程锁防并发重入。
2. FLUX 优先 `OVFluxPipeline`，旧版 optimum-intel 回退 `OVStableDiffusionPipeline`。
3. rembg 走离线：`U2NET_HOME` 指向 MODEL_ROOT/rembg，缺模型时不联网，
   降级为"白底连通域抠图"（prompt 里已要求白底）。
4. 角色名进文件名前统一 `safe_name()`。
5. 已生成的资源默认跳过（缓存命中），force=True 可重画。
"""

import os
import threading
from typing import Dict, Optional

from . import MODEL_ROOT, subdir, safe_name, text_key, hardware

# OpenVINO 后端用 INT4 IR 目录；torch 后端用 HF/diffusers 格式目录（两者不能混用）
DRAW_MODEL_DIR = os.environ.get(
    "DRAW_MODEL_DIR", os.path.join(MODEL_ROOT, "flux-schnell-int4-ov")
)
TORCH_MODEL_DIR = os.environ.get(
    "TORCH_MODEL_DIR", os.path.join(MODEL_ROOT, "flux-schnell")
)
# auto / openvino / torch；auto 时按显卡厂商自动选（见 hardware.pick_draw_backend）
DRAW_BACKEND = os.environ.get("DRAW_BACKEND", "auto")
# auto / GPU / CPU / NPU（仅 OpenVINO 有意义）；auto 时 Intel 走 GPU、其余走 CPU
DRAW_DEVICE = os.environ.get("DRAW_DEVICE", "auto")
DRAW_STEPS = int(os.environ.get("DRAW_STEPS", "4"))
DRAW_SIZE = int(os.environ.get("DRAW_SIZE", "1024"))

_ASSET_DIR = subdir("assets")
_pipe = None
_pipe_lock = threading.RLock()
_gen_lock = threading.Lock()


# ---------------------------------------------------------------------------
# 后端判定
# ---------------------------------------------------------------------------
def _has_module(name: str) -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def resolve_backend() -> Dict[str, str]:
    """返回 {"backend": "openvino"/"torch", "device": ..., "reason": ...}。"""
    forced = (DRAW_BACKEND or "auto").strip().lower()
    vendor = hardware.primary_vendor()

    if forced in ("openvino", "torch"):
        if forced == "openvino":
            dev = hardware.suggest_openvino_device(vendor, DRAW_DEVICE)
            return {"backend": "openvino", "device": dev, "reason": "DRAW_BACKEND 显式指定"}
        return {"backend": "torch", "device": DRAW_DEVICE if DRAW_DEVICE != "auto" else "cuda",
                "reason": "DRAW_BACKEND 显式指定"}

    avail = {
        "openvino": _has_module("optimum.intel.openvino") or _has_module("openvino"),
        "torch": _has_module("torch") and _has_module("diffusers"),
        "directml": _has_module("torch_directml"),
    }
    picked = hardware.pick_draw_backend(vendor, avail, os.name == "nt")
    if not picked["backend"]:
        raise RuntimeError(
            "没有可用的绘图后端。请在项目 venv 里安装其中之一：\n"
            "  Intel/纯 CPU : pip install openvino optimum[openvino]\n"
            "  NVIDIA       : pip install torch diffusers --index-url https://download.pytorch.org/whl/cu124\n"
            "  AMD (Windows): pip install torch torch-directml diffusers\n"
            f"（检测到的显卡：{hardware.list_adapters() or '未识别'} / 厂商 {vendor}）"
        )
    if picked["backend"] == "openvino":
        picked["device"] = hardware.suggest_openvino_device(vendor, DRAW_DEVICE)
    elif DRAW_DEVICE not in ("auto", "", None):
        picked["device"] = DRAW_DEVICE
    return picked


# ---------------------------------------------------------------------------
# pipeline 懒加载
# ---------------------------------------------------------------------------
def get_pipe(device: Optional[str] = None):
    """按需加载绘图 pipeline（单例，线程安全）。"""
    global _pipe
    with _pipe_lock:
        if _pipe is not None:
            return _pipe

        info = resolve_backend()
        model_dir = TORCH_MODEL_DIR if info["backend"] == "torch" else DRAW_MODEL_DIR
        if not os.path.isdir(model_dir) or not os.listdir(model_dir):
            raise RuntimeError(
                f"绘图模型目录不存在或为空：{model_dir}\n"
                "请先把模型放到该目录（都放在 D 盘）：\n"
                "  Intel/CPU : Flux Schnell INT4 OpenVINO 模型 -> DRAW_MODEL_DIR\n"
                "  NVIDIA/AMD: FLUX.1-schnell 的 diffusers 权重   -> TORCH_MODEL_DIR\n"
                "详见 README 第 11 节「显卡适配」与第 12 节「全 D 盘安装」。"
            )

        if info["backend"] == "torch":
            _pipe = _load_torch_pipe(device or info["device"], model_dir)
        else:
            _pipe = _load_openvino_pipe(device or info["device"], model_dir)
        return _pipe


def _load_openvino_pipe(device: str, model_dir: Optional[str] = None):
    """optimum-intel + OpenVINO。Intel GPU / 任何平台 CPU。"""
    from optimum.intel.openvino import OVStableDiffusionPipeline  # 延迟导入

    pipe_cls = OVStableDiffusionPipeline
    try:
        from optimum.intel.openvino import OVFluxPipeline  # type: ignore

        pipe_cls = OVFluxPipeline
    except Exception:
        # 旧版 optimum-intel 没有 Flux 专用 pipeline
        pipe_cls = OVStableDiffusionPipeline

    kwargs = {"compile": True, "ov_config": {"CACHE_DIR": os.path.join(_ASSET_DIR, "ov_cache")}}
    dev = (device or "").strip()
    if dev and dev.upper() != "AUTO":
        kwargs["device"] = dev
    return pipe_cls.from_pretrained(model_dir or DRAW_MODEL_DIR, **kwargs)


def _load_torch_pipe(device: str, model_dir: Optional[str] = None):
    """diffusers + PyTorch。NVIDIA 走 CUDA，AMD 走 DirectML，兜底 CPU。

    注意：INT4 的 OpenVINO 模型（含 .xml/.bin）不能给 torch 用，
    这里要求目录里是 HuggingFace 格式权重（model_index.json）。

    显存策略：FLUX.1-schnell 是 12B 参数，bf16 约 24GB，16G 显存装不下。

    实测（RTX 5060 Ti 16G + torch 2.9.1+cu128 + accelerate 1.15 + Windows）：
      enable_model_cpu_offload()       -> 推理第一步直接 0xC0000005 访问冲突（进程崩）
      enable_sequential_cpu_offload()  -> 正常出图，1024×1024 约 25 分钟，512×512 约 6 分钟
    所以显存不足时默认走 **sequential**（慢但确定能跑）；想让整层换入换出（更快但本机崩）
    就设 `FLUX_OFFLOAD=model`，显存够大想全量驻留就设 `FLUX_OFFLOAD=none`。
    """
    import torch
    from diffusers import DiffusionPipeline

    target = model_dir or TORCH_MODEL_DIR
    dev = (device or "").strip().lower()
    offload_mode = os.environ.get("FLUX_OFFLOAD", "auto").strip().lower()

    if dev == "dml":
        import torch_directml as dml  # type: ignore

        torch_dev = dml.device()
        dtype = torch.float16
    elif dev.startswith("cuda") and torch.cuda.is_available():
        torch_dev = dev
        dtype = torch.bfloat16
    else:
        torch_dev = "cpu"
        dtype = torch.float32

    if not os.path.exists(os.path.join(target, "model_index.json")):
        raise RuntimeError(
            f"{target} 里没有 model_index.json，不是 diffusers 目录。\n"
            "NVIDIA/AMD 的 torch 后端不能用 OpenVINO 的 INT4（.xml/.bin）模型，"
            "请下载 black-forest-labs/FLUX.1-schnell 的 diffusers 权重。"
        )

    pipe = DiffusionPipeline.from_pretrained(
        target, torch_dtype=dtype, local_files_only=True
    )

    if torch_dev == "cpu" or offload_mode == "none":
        print("[warn] 全量驻留（或 CPU 推理），12B 模型需要 ≥24GB 显存/内存")
        pipe.to(torch_dev)
        return pipe

    if offload_mode in ("model", "sequential"):
        mode = offload_mode
    else:
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            total = 0
        # 显存 ≥24GB 才有资格谈"全量驻留"，否则只能 offload
        mode = "none" if total and total >= 24 else "sequential"

    if mode == "none":
        pipe.to(torch_dev)
        return pipe
    if mode == "model":
        print("[info] FLUX_OFFLOAD=model：整层换入换出（本机实测会崩，仅在你确认能跑时用）")
        pipe.enable_model_cpu_offload()
        return pipe

    print("[info] 启用 sequential offload（逐层换入，慢但稳）"
          "；想更快可设 DRAW_SIZE=512 或换 fp8/量化权重")
    pipe.enable_sequential_cpu_offload()
    return pipe



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
_rembg_session = None


def _get_rembg_session():
    """取一个**指定 u2net** 的 rembg session（全局只建一次）。

    rembg 2.x 的 `remove()` 默认模型不是 u2net（新版默认 birefnet-general，约 1 GB），
    不显式指定就会触发联网下载，既破坏"全本地离线"，也会把模型塞到默认目录。
    这里固定用 u2net，且在 import 前把 U2NET_HOME 钉在 MODEL_ROOT/rembg。
    """
    global _rembg_session
    if _rembg_session is not None:
        return _rembg_session

    from rembg import new_session

    model_dir = os.path.join(MODEL_ROOT, "rembg")
    os.environ["U2NET_HOME"] = model_dir  # 必须在 new_session 之前生效
    if not os.path.exists(os.path.join(model_dir, "u2net.onnx")):
        raise FileNotFoundError(f"u2net.onnx 缺失：{model_dir}")

    _rembg_session = new_session("u2net")
    return _rembg_session


def _cutout_offline(img_path: str, out_path: str) -> str:
    """rembg 离线抠图；模型缺失时降级为白底连通域抠图。"""
    try:
        from rembg import remove  # 延迟导入
        from PIL import Image

        src = Image.open(img_path)
        out = remove(src, session=_get_rembg_session())
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
    # 报回"实际用到的"模型目录（torch 与 OpenVINO 是两套目录，别混）
    try:
        _backend = resolve_backend()
        used_dir = TORCH_MODEL_DIR if _backend["backend"] == "torch" else DRAW_MODEL_DIR
    except Exception:
        used_dir = DRAW_MODEL_DIR
    assets: Dict = {"background": "", "backgrounds": {}, "characters": {}, "model_dir": used_dir}

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
