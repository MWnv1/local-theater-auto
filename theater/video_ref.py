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

"""参考视频 → 相似视频。

"相似"的定义写死在这里，免得含糊：
  * 时长、帧率、分辨率与原片一致（时长偏差 ≤5%）；
  * 构图/运动与原片一致 —— 用逐帧 img2img 把原帧当 init image，
    低 strength 保住构图和运动轨迹，只换画风/材质；
  * 音轨默认保留原片（可用 keep_audio=False 关掉）。

为什么不直接用 SVD / Wan-VACE：16 GB 显存跑 14B 视频模型要么爆显存，要么
一条 5 秒视频要几十分钟；而"参考视频 → 相似视频"真正要保的是构图和运动，
逐帧 img2img（SD1.5，8 步左右）在 512 长边上每帧 0.3~0.6 秒，5 秒视频约 1 分钟，
质量与速度的平衡明显更好。需要更强的动作一致性时把 strength 调低到 0.3 左右。

后端：
  SIM_VIDEO_BACKEND=auto|img2img|none
  img2img  → 需要 SD 模型目录（SIM_SD_MODEL，默认 <MODEL_ROOT>/sd15-base）
  none     → 只做"分析 + 复制音轨"（不重绘，用于纯信息提取场景）
"""

import json
import math
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from PIL import Image

from . import MODEL_ROOT, ffmpeg_path, ffprobe_path, subdir

SIM_SD_MODEL = os.environ.get("SIM_SD_MODEL", os.path.join(MODEL_ROOT, "sd15-base"))
SIM_BACKEND = os.environ.get("SIM_VIDEO_BACKEND", "auto")
SIM_SIZE = int(os.environ.get("SIM_SIZE", "512"))
SIM_STEPS = int(os.environ.get("SIM_STEPS", "8"))
SIM_STRENGTH = float(os.environ.get("SIM_STRENGTH", "0.45"))
SIM_MAX_FRAMES = int(os.environ.get("SIM_MAX_FRAMES", "240"))

_pipe = None


# ---------------------------------------------------------------------------
# 纯函数部分（可单测，不碰 ffmpeg / 模型）
# ---------------------------------------------------------------------------
def _parse_fps(text: str, default: float = 25.0) -> float:
    """'30000/1001' → 29.97；非法输入回落默认值。"""
    try:
        if "/" in str(text):
            num, den = str(text).split("/", 1)
            num, den = float(num), float(den)
            if num <= 0 or den <= 0:
                return default
            return round(num / den, 3)
        val = float(text)
        return val if val > 0 else default
    except Exception:
        return default


def plan_frames(duration: float, fps: float, max_frames: int = SIM_MAX_FRAMES) -> dict:
    """决定要重绘多少帧、每隔多少帧取一帧（超过上限就抽帧，再复制回原帧率）。"""
    total = max(1, int(round(duration * fps)))
    count = min(total, max(1, int(max_frames)))
    step = max(1.0, round(total / count, 4))
    return {"total": total, "count": count, "step_frames": step,
            "fps": fps, "duration": duration}


# ---------------------------------------------------------------------------
# 分析
# ---------------------------------------------------------------------------
def _ffprobe_json(path: str) -> dict:
    r = subprocess.run(
        [ffprobe_path(), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe 读取失败：{r.stderr[-300:]}")
    return json.loads(r.stdout or "{}")


def analyze_reference(path: str) -> dict:
    """读参考视频的客观参数 + 抽 4 帧算主色/边缘密度（给后续"像不像"用）。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    info = _ffprobe_json(path)
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if not v:
        raise RuntimeError(f"参考文件里没有视频流：{path}")
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)

    fps = _parse_fps(v.get("r_frame_rate") or v.get("avg_frame_rate"), 25.0)
    duration = float(v.get("duration") or info.get("format", {}).get("duration") or 0.0)
    width, height = int(v.get("width") or 0), int(v.get("height") or 0)
    n_frames = int(v.get("nb_frames") or 0) or int(round(duration * fps))

    palette, edge_density = _sample_stats(path, duration)

    return {
        "path": os.path.abspath(path),
        "duration": round(duration, 3),
        "fps": fps,
        "width": width,
        "height": height,
        "frames": n_frames,
        "has_audio": bool(a),
        "audio_codec": (a or {}).get("codec_name", ""),
        "palette": palette,
        "edge_density": edge_density,
    }


def _sample_stats(path: str, duration: float) -> Tuple[List[List[int]], float]:
    """抽 4 个时间点算平均色与边缘密度（近似，够用来做相似度比较）。"""
    import numpy as np

    if duration <= 0:
        return [], 0.0
    frames = []
    for i in range(4):
        t = duration * (i + 1) / 5.0
        tmp = os.path.join(tempfile.gettempdir(), f"_ref_{os.getpid()}_{i}.png")
        cmd = [ffmpeg_path(), "-y", "-ss", f"{t:.3f}", "-i", path,
               "-frames:v", "1", "-vf", f"scale={SIM_SIZE}:{SIM_SIZE}:force_original_aspect_ratio=decrease",
               tmp]
        subprocess.run(cmd, capture_output=True, text=True)
        if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            frames.append(np.asarray(Image.open(tmp).convert("RGB"), dtype=np.uint8))
            os.remove(tmp)

    if not frames:
        return [], 0.0
    palette = [[int(v) for v in arr.reshape(-1, 3).mean(axis=0)] for arr in frames]
    grads = []
    for arr in frames:
        g = np.abs(np.diff(arr.mean(axis=2), axis=0)).mean()
        grads.append(float(g) / 255.0)
    return palette, round(sum(grads) / len(grads), 4)


def edge_iou(a_path: str, b_path: str, size: int = 256, thresh: int = 28) -> float:
    """两张图边缘掩码的 IoU —— 衡量"构图是否还像原片"。"""
    import numpy as np

    def mask(p):
        im = Image.open(p).convert("L").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.int16)
        gx = np.abs(np.diff(arr, axis=1, prepend=arr[:, :1]))
        gy = np.abs(np.diff(arr, axis=0, prepend=arr[:1, :]))
        return (gx + gy) > thresh

    ma, mb = mask(a_path), mask(b_path)
    inter = (ma & mb).sum()
    union = (ma | mb).sum()
    return float(inter) / float(union) if union else 0.0


# ---------------------------------------------------------------------------
# 重绘后端
# ---------------------------------------------------------------------------
def _sd_available() -> bool:
    if not os.path.isdir(SIM_SD_MODEL):
        return False
    if not os.path.exists(os.path.join(SIM_SD_MODEL, "model_index.json")):
        return False
    try:
        import diffusers  # noqa: F401
        return True
    except Exception:
        return False


def _get_pipe():
    """SD1.5 img2img 管道（fp16 常驻，16 GB 显存绰绰有余）。"""
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SIM_SD_MODEL, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _pipe = _pipe.to(device)
    _pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        _pipe.enable_attention_slicing()
    return _pipe


def _restyle(frame_path: str, out_path: str, prompt: str, negative: str,
             strength: float, steps: int) -> str:
    """单帧 img2img：原帧作 init image，低 strength 保构图。

    重绘在 `SIM_SIZE` 长边上做，出图后再放回原分辨率。直接在 1080p 上跑 img2img
    在本机是 ~75 秒/帧（24 帧要半小时），缩到 512 后是 ~1 秒/帧，而"相似"要的
    构图和运动跟分辨率无关——像素级细节本来也会被 strength 改掉。
    """
    pipe = _get_pipe()
    init = Image.open(frame_path).convert("RGB")
    w, h = init.size
    work = init
    if max(w, h) > SIM_SIZE:
        scale = SIM_SIZE / float(max(w, h))
        # SD 要求边长是 8 的倍数
        nw = max(64, int(round(w * scale / 8.0)) * 8)
        nh = max(64, int(round(h * scale / 8.0)) * 8)
        work = init.resize((nw, nh), Image.Resampling.LANCZOS)
    gen = pipe(
        prompt=prompt, negative_prompt=negative, image=work, strength=strength,
        num_inference_steps=max(2, steps), guidance_scale=7.0,
    ).images[0]
    if gen.size != (w, h):
        gen = gen.resize((w, h), Image.Resampling.LANCZOS)
    gen.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def _extract_frames(path: str, fps: float, workdir: str, stride: float) -> List[str]:
    """按 stride 抽帧（保持原分辨率），返回帧文件列表。"""
    os.makedirs(workdir, exist_ok=True)
    pattern = os.path.join(workdir, "src_%05d.png")
    cmd = [ffmpeg_path(), "-y", "-i", path, "-vf", f"fps={fps / stride:.6f}", pattern]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"抽帧失败：{r.stderr[-300:]}")
    files = sorted(os.path.join(workdir, f) for f in os.listdir(workdir) if f.startswith("src_") and f.endswith(".png"))
    if not files:
        raise RuntimeError("没有抽到任何帧")
    return files


def generate_similar_video(ref_path: str, prompt: str, out_path: str,
                           strength: Optional[float] = None, steps: Optional[int] = None,
                           keep_audio: bool = True, max_frames: Optional[int] = None,
                           backend: Optional[str] = None, progress=None) -> Tuple[str, dict]:
    """按参考视频生成相似视频，返回 (输出路径, 分析/执行信息)。"""
    info = analyze_reference(ref_path)
    duration, fps = info["duration"], info["fps"]
    if duration <= 0:
        raise RuntimeError("参考视频时长为 0，无法复刻")
    strength = SIM_STRENGTH if strength is None else float(strength)
    steps = SIM_STEPS if steps is None else int(steps)
    backend = (backend or SIM_BACKEND or "auto").lower()
    if backend == "auto":
        backend = "img2img" if _sd_available() else "none"

    plan = plan_frames(duration, fps, max_frames or SIM_MAX_FRAMES)
    workdir = subdir(os.path.join("sim", os.path.splitext(os.path.basename(out_path))[0]))
    for fn in os.listdir(workdir):
        if fn.endswith(".png"):
            os.remove(os.path.join(workdir, fn))

    # 1) 按"每隔 step 帧取一帧"抽帧（超过帧预算时就是抽帧重绘，再复制回原时长）
    stride = max(1.0, plan["step_frames"])
    src_frames = _extract_frames(ref_path, fps, workdir, stride)

    negative = os.environ.get(
        "SIM_NEGATIVE",
        "blurry, low quality, watermark, text, deformed, extra limbs, jpeg artifacts",
    )

    out_frames: List[str] = []
    if backend == "img2img":
        for i, fpath in enumerate(src_frames[: plan["count"]]):
            dst = os.path.join(workdir, f"gen_{i:05d}.png")
            _restyle(fpath, dst, prompt, negative, strength, steps)
            out_frames.append(dst)
            if progress:
                progress(i + 1, min(len(src_frames), plan["count"]))
    else:
        out_frames = src_frames

    # 2) 组装：生成帧 → 按原 fps 输出；帧数不足时用 tpad 拖住最后一帧，保证时长一致
    frame_fps = fps / stride
    silent = os.path.join(workdir, "silent.mp4")
    cmd = [ffmpeg_path(), "-y", "-r", f"{frame_fps:.6f}"]
    lst = os.path.join(workdir, "gen_list.txt")
    with open(lst, "w", encoding="utf-8") as f:
        for p in out_frames:
            f.write("file '%s'\n" % p.replace("\\", "/"))
    cmd += ["-f", "concat", "-safe", "0", "-i", lst,
            "-vf", f"fps={fps:.6f},scale={info['width']}:{info['height']}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
            "-t", f"{duration:.3f}", silent]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"组装视频失败：{r.stderr[-400:]}")

    # 3) 合并音轨
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if keep_audio and info["has_audio"]:
        cmd2 = [ffmpeg_path(), "-y", "-i", silent, "-i", ref_path,
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                "-t", f"{duration:.3f}", out_path]
    else:
        cmd2 = [ffmpeg_path(), "-y", "-i", silent, "-c", "copy", out_path]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode != 0:
        raise RuntimeError(f"合并音轨失败：{r2.stderr[-400:]}")

    out_info = analyze_reference(out_path)
    out_info["backend"] = backend
    out_info["strength"] = strength
    out_info["steps"] = steps
    out_info["prompt"] = prompt
    out_info["redrawn"] = len(out_frames) if backend == "img2img" else 0
    out_info["duration_delta"] = round(abs(out_info["duration"] - duration), 3)
    out_info["workdir"] = os.path.abspath(workdir)
    return os.path.abspath(out_path), out_info


def similarity_report(ref_path: str, out_path: str, samples: int = 4) -> dict:
    """给"像不像"一个可核对的数字：时长偏差、分辨率、边缘 IoU。"""
    a = analyze_reference(ref_path)
    b = analyze_reference(out_path)
    durations = (a["duration"], b["duration"])
    # 抽同样时间点的帧做边缘 IoU
    ious = []
    wd = tempfile.mkdtemp(prefix="simcmp_")
    for i in range(samples):
        t = a["duration"] * (i + 1) / (samples + 1)
        pa = os.path.join(wd, f"a{i}.png")
        pb = os.path.join(wd, f"b{i}.png")
        for src, dst in ((ref_path, pa), (out_path, pb)):
            subprocess.run([ffmpeg_path(), "-y", "-ss", f"{t:.3f}", "-i", src, "-frames:v", "1", dst],
                           capture_output=True, text=True)
        if os.path.exists(pa) and os.path.exists(pb):
            ious.append(edge_iou(pa, pb))
    return {
        "ref": {"duration": durations[0], "fps": a["fps"], "size": [a["width"], a["height"]]},
        "out": {"duration": durations[1], "fps": b["fps"], "size": [b["width"], b["height"]]},
        "duration_delta_pct": round(abs(durations[1] - durations[0]) / max(0.001, durations[0]) * 100, 2),
        "fps_match": abs(a["fps"] - b["fps"]) < 0.05,
        "size_match": (a["width"], a["height"]) == (b["width"], b["height"]),
        "edge_iou_mean": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }
