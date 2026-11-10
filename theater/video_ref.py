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
  * 构图/运动与原片一致 —— 先把原片逐帧比出"这一帧相对上一帧平移多少、转多少"
    （相位相关 + 旋转搜索，见 frame_offset / analyze_motion），把统计结果交给
    AI 归纳成一句画面描述，再用逐帧 img2img 重绘；
  * 音轨默认保留原片（可用 keep_audio=False 关掉）。

为什么不直接用 SVD / Wan-VACE：16 GB 显存跑 14B 视频模型要么爆显存，要么
一条 5 秒视频要几十分钟；而"参考视频 → 相似视频"真正要保的是构图和运动，
逐帧 img2img（SD1.5）在 768 长边上每帧 2 秒左右，5 秒视频约 1 分钟。

**2026-11-10 修过的"果冻"事故（重点，别再踩）**
  症状：生成片整幅变成大理石纹/融化块，人物五官糊成一团。
  原因不是"模型不行"，是**去噪步数不够**：`num_inference_steps=8` 配
  `strength=0.45` 实际只跑 4 步（SD 按 `int(steps*strength)` 取整），
  `strength=0.28` 只剩 2 步，等于拿着噪声直接出图。三个修复：
    1. `_restyle()` 强制 `steps*strength >= SIM_MIN_DENOISE`（默认 12 步），
       `SIM_STEPS=32`，于是 0.35 强度实际跑 11 步、0.45 跑 14 步；
    2. 重绘尺度 512 → **768**：512 长边上 16:9 只有 512×288，细节全丢，
       放大回 1080p 就是一片糊；768 下边缘 IoU 从 0.44 提到 0.57；
    3. `SIM_MOTION_WARP` 默认 **0**：把上一帧的"生成图"当下一帧 init 会
       把瑕疵一路累积（第一帧糊一点，第八帧全是纹），每帧直接用原片最稳。
  另外加了 `_guard_blend()`：出图后跟原帧算边缘 IoU，低于 `SIM_MIN_IOU`
  就按 `SIM_GUARD_MIX` 跟原帧混合，宁可保守也不交一张"果冻"。

后端：
  SIM_VIDEO_BACKEND=auto|img2img|none
  img2img  → 需要 SD 模型目录（SIM_SD_MODEL，默认 <MODEL_ROOT>/sd15-base）
  none     → 只做"分析 + 复制音轨"（不重绘，用于纯信息提取场景）

运动分析：
  SIM_MOTION_SIZE=256   分析尺度（长边像素），越小越快越鲁棒
  SIM_MOTION_LLM=0      0=用规则模板出英文提示；1=加载本地 Qwen3-4B 改写
  SIM_MOTION_WARP=0     1=把上一帧生成图按偏移推成下一帧 init；0=每帧都用原片
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
SIM_CONTROLNET_DIR = os.environ.get(
    "SIM_CONTROLNET_DIR", r"D:\AI_T_MCP\models\controlnet-sd15-canny")
SIM_BACKEND = os.environ.get("SIM_VIDEO_BACKEND", "auto")
SIM_SIZE = int(os.environ.get("SIM_SIZE", "768"))
SIM_STEPS = int(os.environ.get("SIM_STEPS", "32"))
_SIM_STRENGTH_ENV = os.environ.get("SIM_STRENGTH", "").strip()
SIM_STRENGTH = float(_SIM_STRENGTH_ENV) if _SIM_STRENGTH_ENV else 0.35
SIM_CONTROLNET = os.environ.get("SIM_CONTROLNET", "auto")   # auto|on|off
SIM_CN_SCALE = float(os.environ.get("SIM_CONTROLNET_SCALE", "0.9"))
SIM_MIN_DENOISE = int(os.environ.get("SIM_MIN_DENOISE", "12"))
SIM_SEED = int(os.environ.get("SIM_SEED", "20261110"))
SIM_GUARD = os.environ.get("SIM_GUARD", "1")
SIM_MIN_IOU = float(os.environ.get("SIM_MIN_IOU", "0.22"))
SIM_GUARD_MIX = float(os.environ.get("SIM_GUARD_MIX", "0.55"))
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
# 逐帧运动分析：这一帧相对上一帧，平移了多少、转了多少
# ---------------------------------------------------------------------------
MOTION_SIZE = int(os.environ.get("SIM_MOTION_SIZE", "256"))
SIM_MOTION_LLM = os.environ.get("SIM_MOTION_LLM", "0")
# 默认 0：用上一帧"生成图"当 init 会把瑕疵逐帧放大（1.5.0 前的"果冻"根因之一）
SIM_MOTION_WARP = os.environ.get("SIM_MOTION_WARP", "0")


def _truthy(val) -> bool:
    return str(val).strip().lower() not in ("", "0", "false", "off", "no")


MOTION_LLM_ON = _truthy(SIM_MOTION_LLM)     # 是否让本地 Qwen3-4B 改写运动描述
MOTION_WARP_ON = _truthy(SIM_MOTION_WARP)   # 是否把上一帧生成图按偏移推成下一帧 init


def _gray(path: str, size: Tuple[int, int]) -> "np.ndarray":
    """读成灰度 float（0~1），并缩到分析尺寸。"""
    import numpy as np

    im = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _gray_rotated(path: str, size: Tuple[int, int], deg: float) -> "np.ndarray":
    """把整帧绕中心转 deg 度后再取灰度（旋转在分析尺度上做，便宜且够用）。

    注意符号：PIL 的 `rotate(角度)` 是**逆时针**，而 `animator.warp()` 的角度按
    图像坐标（y 向下）正方向是顺时针。这里取 `-deg`，让本函数报出来的 `rot`
    与 `warp` 同号——不然测到 +6° 拿去 warp 会转反。
    """
    if abs(deg) < 1e-6:
        return _gray(path, size)
    im = Image.open(path).convert("L").resize(size, Image.Resampling.BILINEAR)
    im = im.rotate(-deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)
    import numpy as np

    return np.asarray(im, dtype=np.float32) / 255.0


def _hann(a):
    """去均值 + 汉宁窗：相位相关最怕边缘跳变把峰值带偏。"""
    import numpy as np

    wy = np.hanning(a.shape[0]).astype(np.float32)[:, None]
    wx = np.hanning(a.shape[1]).astype(np.float32)[None, :]
    return (a - float(a.mean())) * wy * wx


def _phase_peak(a, b) -> Tuple[int, int]:
    """相位相关：返回把 b 对齐到 a 的整数位移 (dy, dx)。"""
    import numpy as np

    f = np.fft.rfft2(a) * np.conj(np.fft.rfft2(b))
    c = np.abs(np.fft.irfft2(f, s=a.shape))
    h, w = a.shape
    dy, dx = np.unravel_index(int(np.argmax(c)), c.shape)
    dy = int(dy) - h if dy > h // 2 else int(dy)
    dx = int(dx) - w if dx > w // 2 else int(dx)
    return dy, dx


def _corr_at(cur, prev, dx: int, dy: int) -> float:
    """cur[y, x] ≈ prev[y-dy, x-dx] 成立时，重叠区的归一化相关系数（0~1）。"""
    import numpy as np

    h, w = cur.shape
    ya0, ya1 = max(0, dy), h + min(0, dy)
    xa0, xa1 = max(0, dx), w + min(0, dx)
    if ya1 - ya0 < 8 or xa1 - xa0 < 8:
        return 0.0
    A = cur[ya0:ya1, xa0:xa1]
    B = prev[ya0 - dy:ya1 - dy, xa0 - dx:xa1 - dx]
    A = A - A.mean()
    B = B - B.mean()
    denom = float(np.sqrt((A * A).sum()) * np.sqrt((B * B).sum()))
    return float((A * B).sum() / denom) if denom > 1e-6 else 0.0


def frame_offset(prev_path: str, cur_path: str, size: int = MOTION_SIZE,
                 rot_search: int = 8, rot_step: int = 2) -> dict:
    """算 cur 相对 prev 的整体运动（平移 + 旋转），结果换算回原片像素。

    做法：先在分析尺度上相位相关拿平移候选，再在 ±rot_search 度里搜旋转
    （每换一个角度就把 prev 转过去、重新相关、算重叠区相关系数，取最高）。
    相关系数同时当置信度——切镜、闪白这类"画面根本不像"的帧对会被打低分，
    后面就不会拿它去推 init image。
    """
    w0, h0 = Image.open(cur_path).convert("L").size
    scale = min(1.0, float(size) / float(max(w0, h0)))
    aw = max(64, int(round(w0 * scale)))
    ah = max(64, int(round(h0 * scale)))
    raw_cur = _gray(cur_path, (aw, ah))
    raw_prev = _gray(prev_path, (aw, ah))
    # 画面变化比例：整帧都变 = 镜头/整体平移；只有一小块变 = 主体自己在动
    changed = float((abs(raw_cur - raw_prev) > 0.12).mean())
    cur = _hann(raw_cur)

    def _probe(deg: float) -> dict:
        prev = _hann(_gray_rotated(prev_path, (aw, ah), deg))
        dy, dx = _phase_peak(prev, cur)
        # 相位相关的符号约定在不同 FFT 库里会翻，直接拿相关系数当裁判取更像的那组
        cand = [(dx, dy, _corr_at(cur, prev, dx, dy)),
                (-dx, -dy, _corr_at(cur, prev, -dx, -dy))]
        cand.sort(key=lambda t: t[2], reverse=True)
        return {"dx": cand[0][0], "dy": cand[0][1], "rot": float(deg), "corr": cand[0][2]}

    angles = list(range(-rot_search, rot_search + 1, max(1, rot_step)) or [0])
    best = max((_probe(a) for a in angles), key=lambda d: d["corr"])
    for deg in (best["rot"] - 1, best["rot"] + 1):
        if abs(deg) > rot_search + 1:
            continue
        p = _probe(deg)
        if p["corr"] > best["corr"]:
            best = p

    sx = w0 / float(aw)
    sy = h0 / float(ah)
    return {"dx": round(best["dx"] * sx, 2), "dy": round(best["dy"] * sy, 2),
            "rot": round(best["rot"], 2),
            "conf": round(max(0.0, min(1.0, best["corr"])), 4),
            "changed": round(changed, 3),
            "analysis": [aw, ah]}


def describe_motion(offsets: List[dict], fps: Optional[float] = None) -> dict:
    """把逐帧位移归纳成一句人话 + 一句英文生成提示。"""
    if not offsets:
        return {"label": "静止（只有一帧，无可比位移）", "prompt_hint": "static shot",
                "mean_dx": 0.0, "mean_dy": 0.0, "speed_px": 0.0, "speed_per_sec": 0.0,
                "total_rot": 0.0, "static_ratio": 1.0, "frames": 1}

    n = len(offsets)
    mean_dx = sum(o["dx"] for o in offsets) / n
    mean_dy = sum(o["dy"] for o in offsets) / n
    speed = sum(math.hypot(o["dx"], o["dy"]) for o in offsets) / n
    total_rot = sum(o["rot"] for o in offsets)
    still = sum(1 for o in offsets
                if math.hypot(o["dx"], o["dy"]) < 1.0 and abs(o["rot"]) < 0.5) / n
    abs_x = sum(abs(o["dx"]) for o in offsets)
    abs_y = sum(abs(o["dy"]) for o in offsets)
    local = sum(float(o.get("changed") or 0.0) for o in offsets) / n

    if still >= 0.6:
        if local >= 0.02:
            # 整帧没动但明显有像素在变 → 是主体自己在动，不是镜头在动
            label = (f"镜头基本不动，画面内有局部动作（平均 {local:.0%} 的像素在变）")
            hint = ("static camera, only the subject moves inside the frame; "
                    "keep the subject fully visible and unoccluded")
        else:
            label = f"基本静止（{still:.0%} 的相邻帧几乎不动）"
            hint = "static shot, no camera motion"
    elif abs_x >= abs_y and abs(mean_dx) < 0.4 * (abs_x / n) and abs_x / n >= 2.0:
        label = f"左右往返平移，平均每帧 {abs_x / n:.1f} px"
        hint = ("gentle back-and-forth pan, the motion is already baked into the frame")
    elif abs_y > abs_x and abs(mean_dy) < 0.4 * (abs_y / n) and abs_y / n >= 2.0:
        label = f"上下往返平移，平均每帧 {abs_y / n:.1f} px"
        hint = ("gentle up-and-down pan, the motion is already baked into the frame")
    elif abs_x >= abs_y:
        label = (f"{'向右' if mean_dx > 0 else '向左'}平移，平均每帧 "
                 f"{abs(mean_dx):.1f} px")
        hint = (f"smooth pan to the {'right' if mean_dx > 0 else 'left'}, "
                f"the motion is already baked into the frame")
    else:
        label = (f"{'向下' if mean_dy > 0 else '向上'}平移，平均每帧 "
                 f"{abs(mean_dy):.1f} px")
        hint = (f"smooth pan {'downward' if mean_dy > 0 else 'upward'}, "
                f"the motion is already baked into the frame")

    per_frame_rot = total_rot / n
    if abs(per_frame_rot) >= 0.4:
        label += f"，并伴随{'顺' if per_frame_rot > 0 else '逆'}时针旋转（累计 {abs(total_rot):.1f}°）"
        hint += f", slight {'clockwise' if per_frame_rot > 0 else 'counter-clockwise'} rotation"

    return {"label": label, "prompt_hint": hint,
            "mean_dx": round(mean_dx, 2), "mean_dy": round(mean_dy, 2),
            "speed_px": round(speed, 2),
            "speed_per_sec": round(speed * float(fps or 0), 2),
            "total_rot": round(total_rot, 2),
            "static_ratio": round(still, 3), "change_ratio": round(local, 3),
            "kind": "local" if (still >= 0.6 and local >= 0.02) else
                    ("static" if still >= 0.6 else "global"),
            "frames": n + 1}


def analyze_motion(frame_paths: List[str], fps: Optional[float] = None,
                   size: int = MOTION_SIZE) -> dict:
    """逐帧比对：第 i 帧相对第 i-1 帧的偏移，最后给出整体结论。"""
    offsets: List[dict] = []
    for i in range(1, len(frame_paths)):
        off = frame_offset(frame_paths[i - 1], frame_paths[i], size=size)
        off["i"] = i
        offsets.append(off)
    return {"offsets": offsets, "summary": describe_motion(offsets, fps),
            "frames": len(frame_paths)}


def motion_hint_llm(summary: dict) -> str:
    """让本地剧本模型（Qwen3-4B）把运动统计改写成一句英文画面描述。

    默认关（SIM_MOTION_LLM=0）：加载 7.5 GB 权重在本机约 147 秒，只在真的要
    "AI 分析"、且打算连着跑多段视频时才开。任何失败都返回空串，调用方回落
    到规则描述，不会因为模型没装好就把整条视频卡住。
    """
    if not MOTION_LLM_ON:
        return ""
    try:
        from pipeline_full import _load_llm

        backend, model, tok = _load_llm()
        sys_p = ("你是视频运镜分析器。只输出一句英文描述画面运动，用作图像生成提示词，"
                 "不要解释、不要编号。")
        stats = json.dumps({k: v for k, v in summary.items() if k != "label"},
                           ensure_ascii=False)
        user = f"逐帧运动统计：{stats}"
        if backend == "torch":
            import torch

            msgs = [{"role": "system", "content": sys_p}, {"role": "user", "content": user}]
            try:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
            gen = out[0][inputs["input_ids"].shape[1]:]
            return tok.decode(gen, skip_special_tokens=True).strip().splitlines()[0][:200]
        inputs = tok(f"{sys_p}\n{user}\n", return_tensors="pt")
        out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        txt = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return txt.strip().splitlines()[0][:200] if txt.strip() else ""
    except Exception as e:  # 模型没装好只是降级，不是失败
        print(f"[motion] LLM 分析不可用，改用规则描述：{type(e).__name__}: {e}")
        return ""


def _warp_prev(prev_png: str, dx: float, dy: float, rot: float, out_png: str) -> str:
    """把上一帧生成图按本帧测到的运动推过去，当作本帧 img2img 的 init image。"""
    from .animator import warp

    img = Image.open(prev_png).convert("RGBA")
    warp(img, dx, dy, rot).convert("RGB").save(out_png)
    return out_png


def motion_signature(path: str, samples: int = 8) -> dict:
    """对一段视频抽若干帧做同样的逐帧分析，用来比较两段视频的"动法"像不像。"""
    info = analyze_reference(path)
    wd = tempfile.mkdtemp(prefix="motsig_")
    plan = plan_frames(info["duration"], info["fps"], samples)
    stride = max(1.0, plan["step_frames"])
    frames = _extract_frames(path, info["fps"], wd, stride)
    motion = analyze_motion(frames, fps=info["fps"] / stride)
    return {"summary": motion["summary"], "offsets": motion["offsets"],
            "source_frames": len(frames)}


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
    """SD1.5 img2img 管道（精度与 attention 内核按本机配置挑，见 accelerator）。"""
    global _pipe
    if _pipe is not None:
        return _pipe
    import torch
    from diffusers import StableDiffusionImg2ImgPipeline

    from . import accelerator

    accel = accelerator.apply_global()
    dtype = accelerator.sd_kwargs()["torch_dtype"]
    _pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        SIM_SD_MODEL, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False,
        local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _pipe = _pipe.to(device)
    _pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        _pipe.enable_attention_slicing()
        applied = accelerator.optimize_pipe(_pipe)
        print(f"[accel] SD1.5 {str(dtype).replace('torch.', '')} / {device}"
              f" / attention={accel['attention']} / channels_last={applied['channels_last']}")
    return _pipe


def _controlnet_available() -> bool:
    """ControlNet 目录里同时有 config.json 与 fp16 权重才算可用。"""
    if not os.path.isdir(SIM_CONTROLNET_DIR):
        return False
    if not os.path.exists(os.path.join(SIM_CONTROLNET_DIR, "config.json")):
        return False
    for fn in ("diffusion_pytorch_model.fp16.safetensors",
               "diffusion_pytorch_model.safetensors"):
        if os.path.exists(os.path.join(SIM_CONTROLNET_DIR, fn)):
            return True
    return False


_cn_pipe = None


def _get_controlnet_pipe():
    """SD1.5 + Canny ControlNet 的 img2img 管道（可选，目录不在就返回 None）。

    为什么不默认用它：多一个 720 MB 模型、每帧多 0.4 秒；但只要它在，
    strength 就能开到 0.5 还不跑偏——轮廓由 ControlNet 锁住，模型只改材质和光。
    """
    global _cn_pipe
    if _cn_pipe is not None:
        return _cn_pipe
    if not _controlnet_available():
        return None
    import torch
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    from . import accelerator

    accel = accelerator.apply_global()
    dtype = accelerator.sd_kwargs()["torch_dtype"]
    try:
        # 官方仓库只放 fp16 权重，必须显式指 variant，否则 diffusers 找 .bin 找不到
        cn = ControlNetModel.from_pretrained(SIM_CONTROLNET_DIR, torch_dtype=dtype,
                                             variant="fp16", local_files_only=True)
    except Exception:
        cn = ControlNetModel.from_pretrained(SIM_CONTROLNET_DIR, torch_dtype=dtype,
                                             local_files_only=True)
    _cn_pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        SIM_SD_MODEL, controlnet=cn, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False, local_files_only=True,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _cn_pipe = _cn_pipe.to(device)
    _cn_pipe.set_progress_bar_config(disable=True)
    if device == "cuda":
        accelerator.optimize_pipe(_cn_pipe)
    print(f"[accel] ControlNet canny 就绪（dtype={str(dtype).replace('torch.', '')}，"
          f"conditioning_scale={SIM_CN_SCALE}，attention={accel['attention']}）")
    return _cn_pipe


def _canny_edges(im, low: int = 60, high: int = 160):
    """给 ControlNet 用的 canny 近似图（PIL 实现，不装 opencv）。

    ControlNet 期望"黑底白线"。PIL 的 FIND_EDGES 只给梯度幅值，所以按本帧自身的
    2%/98% 分位归一化，再按阈值二值化，最后用 MaxFilter 把线加粗一点。
    不用 `autocontrast`：纯色画面（照墙、过场）经 autocontrast 会被"拉"成全白，
    ControlNet 拿到全白条件就会到处乱生东西；这里先数直方图众数占比，画面对
    97% 以上是同一个梯度值就直接返全黑（= 无条件，退回纯 img2img）。
    """
    import numpy as np
    from PIL import ImageFilter

    g = im.convert("L").filter(ImageFilter.GaussianBlur(2.5))
    e = g.filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(e, dtype=np.uint8)
    hist = np.bincount(arr.ravel(), minlength=256)
    if hist.max() >= 0.97 * arr.size:
        return Image.new("RGB", im.size, (0, 0, 0))
    lo, hi = np.percentile(arr, [5, 99])
    if hi - lo < 12:
        return Image.new("RGB", im.size, (0, 0, 0))
    arr = np.clip((arr.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    arr = np.where(arr > high, 255, 0).astype(np.uint8)
    e = Image.fromarray(arr, mode="L").filter(ImageFilter.MaxFilter(3))
    return e.convert("RGB")


def _edge_mask(im, size: int = 768):
    """粗略边缘掩码（PIL 实现，不引入 opencv）。

    先缩到 768 长边再取边缘：同一对 1080p 图在原尺寸上边缘像素只有 1 px 宽，
    模糊一下就没剩几个点，IoU 会莫名掉到 0.08；缩放后同一对图稳定在 0.4 左右，
    指标才有可比性。
    """
    import numpy as np
    from PIL import ImageFilter

    if max(im.size) > size:
        scale = size / float(max(im.size))
        im = im.resize((max(1, int(im.size[0] * scale)), max(1, int(im.size[1] * scale)),
                        ), Image.Resampling.LANCZOS)
    g = im.convert("L").filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(3))
    return np.asarray(g, dtype=np.uint8) > 40


def _edge_iou(a, b) -> float:
    import numpy as np

    ea, eb = _edge_mask(a), _edge_mask(b)
    inter = np.logical_and(ea, eb).sum()
    union = np.logical_or(ea, eb).sum()
    return float(inter / max(union, 1))


def _guard_blend(gen, src, iou: float):
    """出图跟原帧的边缘差太多就按比例混回原帧 —— 宁可保守，也不交"果冻"。"""
    if not _truthy(SIM_GUARD) or iou >= SIM_MIN_IOU:
        return gen, False
    mix = min(0.85, max(0.2, SIM_GUARD_MIX))
    return Image.blend(gen, src, mix), True


def _controlnet_on() -> bool:
    """是否真的启用 ControlNet：明确 on 但没有权重 → 打印一次回落提示。"""
    mode = str(SIM_CONTROLNET).strip().lower()
    if mode in ("off", "0", "false", "no"):
        return False
    if _controlnet_available():
        return True
    if mode in ("on", "1", "true", "yes"):
        print(f"[warn] SIM_CONTROLNET={SIM_CONTROLNET} 但 {SIM_CONTROLNET_DIR} 没有权重，"
              "回落到普通 img2img")
    return False


def _default_strength() -> float:
    """默认 0.35：有 ControlNet 时同一强度下轮廓更贴原帧（实测 0.35/scale 0.9）。

    别指望"有 CN 就可以开到 0.45"——实测 0.45 会在墙面/天花板上刷出颗粒噪点，
    反而更像事故现场；结构的收益在 0.35 已经吃满。
    """
    if _SIM_STRENGTH_ENV:
        return SIM_STRENGTH
    return 0.35


def _restyle(frame_path: str, out_path: str, prompt: str, negative: str,
             strength: float, steps: int) -> str:
    """单帧 img2img：原帧作 init image，低 strength 保构图。

    重绘在 `SIM_SIZE`（默认 768）长边上做，出图后再放回原分辨率。直接在 1080p 上
    跑 img2img 在本机是 ~75 秒/帧；缩到 768 是 ~2 秒/帧，而"相似"要的构图和运动
    跟分辨率无关。512 太小（16:9 只剩 512×288），细节被抹平后再放大就是糊。

    **步数下限**：diffusers 实际只跑 `int(num_inference_steps * strength)` 步。
    8 步配 0.35 强度只有 2 步，出来就是噪声糊块（2026-11-10 的"果冻"）。
    所以这里强制 `steps*strength >= SIM_MIN_DENOISE`（默认 12）。

    有 ControlNet 权重时（`SIM_CONTROLNET_DIR`）走 canny 条件：轮廓由 ControlNet
    锁住，模型只改材质/光线，人物五官不会被 strength 抹掉。
    """
    cn_on = _controlnet_on()
    pipe = _get_controlnet_pipe() if cn_on else _get_pipe()
    init = Image.open(frame_path).convert("RGB")
    w, h = init.size
    work = init
    if max(w, h) > SIM_SIZE:
        scale = SIM_SIZE / float(max(w, h))
        # SD 要求边长是 8 的倍数
        nw = max(64, int(round(w * scale / 8.0)) * 8)
        nh = max(64, int(round(h * scale / 8.0)) * 8)
        work = init.resize((nw, nh), Image.Resampling.LANCZOS)

    strength = min(0.95, max(0.05, float(strength)))
    total = max(2, int(steps), int(math.ceil(SIM_MIN_DENOISE / strength)))
    kwargs = {}
    if SIM_SEED >= 0:
        import torch

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs["generator"] = torch.Generator(device=dev).manual_seed(SIM_SEED)

    gen = pipe(
        prompt=prompt, negative_prompt=negative, image=work, strength=strength,
        num_inference_steps=total, guidance_scale=7.0, **kwargs,
        **({"control_image": _canny_edges(work),
            "controlnet_conditioning_scale": SIM_CN_SCALE} if cn_on else {}),
    ).images[0]

    iou = _edge_iou(work, gen)
    gen, blended = _guard_blend(gen, work, iou)
    if blended:
        print(f"[guard] 边缘 IoU {iou:.3f} < {SIM_MIN_IOU}，按 {SIM_GUARD_MIX} 混回原帧")

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
    strength = _default_strength() if strength is None else float(strength)
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
    usable = src_frames[: plan["count"]]

    # 1.5) 逐帧运动分析：这一帧相对上一帧平移了多少、转了多少 → 交给 AI 生成
    motion = analyze_motion(usable, fps=fps / stride)
    motion_sum = motion["summary"]
    print(f"[motion] {motion_sum['label']}；"
          f"{motion_sum['speed_px']} px/帧（{motion_sum['speed_per_sec']} px/s），"
          f"累计旋转 {motion_sum['total_rot']}°，静止帧占比 {motion_sum['static_ratio']}")
    llm_hint = motion_hint_llm(motion_sum)
    hint = llm_hint or motion_sum["prompt_hint"]
    prompt_all = f"{prompt}, {hint}" if hint else prompt
    with open(os.path.join(workdir, "motion.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": motion_sum, "llm_hint": llm_hint,
                   "prompt_used": prompt_all, "offsets": motion["offsets"]},
                  f, ensure_ascii=False, indent=2)

    negative = os.environ.get(
        "SIM_NEGATIVE",
        "blurry, out of focus, low contrast, low quality, jpeg artifacts, distorted, "
        "melting, warped, smeared, extra limbs, bad anatomy, deformed face, "
        "grainy, noisy, speckles, dithering, dust, watermark, text, oversaturated",
    )

    out_frames: List[str] = []
    warped = 0
    if backend == "img2img":
        prev_gen: Optional[str] = None
        for i, fpath in enumerate(usable):
            dst = os.path.join(workdir, f"gen_{i:05d}.png")
            init = fpath
            # 只有"整帧在动"（镜头/整体平移）才把上一帧生成图推过去当 init；
            # 局部动作（主体自己走）推整帧会把背景一起带走，直接用原帧更准。
            if (i > 0 and prev_gen and MOTION_WARP_ON and motion["offsets"]
                    and motion_sum.get("kind") == "global"):
                off = motion["offsets"][min(i - 1, len(motion["offsets"]) - 1)]
                if (math.hypot(off["dx"], off["dy"]) >= 2.0 or abs(off["rot"]) >= 0.8) \
                        and off.get("conf", 0) >= 0.35:
                    init = _warp_prev(prev_gen, off["dx"], off["dy"], off["rot"],
                                      os.path.join(workdir, f"pred_{i:05d}.png"))
                    warped += 1
            _restyle(init, dst, prompt_all, negative, strength, steps)
            prev_gen = dst
            out_frames.append(dst)
            if progress:
                progress(i + 1, len(usable))
    else:
        out_frames = usable

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
    out_info["prompt"] = prompt_all
    out_info["redrawn"] = len(out_frames) if backend == "img2img" else 0
    out_info["duration_delta"] = round(abs(out_info["duration"] - duration), 3)
    out_info["motion"] = {"summary": motion_sum, "llm_hint": llm_hint,
                          "offsets_total": len(motion["offsets"]),
                          "offsets": motion["offsets"][:16],
                          "kind": motion_sum.get("kind"),
                          "warped_init": bool(MOTION_WARP_ON),
                          "warped_frames": warped}
    out_info["workdir"] = os.path.abspath(workdir)
    return os.path.abspath(out_path), out_info


def similarity_report(ref_path: str, out_path: str, samples: int = 4,
                      motion_samples: int = 6) -> dict:
    """给"像不像"一个可核对的数字：时长、帧率、分辨率、边缘 IoU，外加运动是否同向。"""
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

    report = {
        "ref": {"duration": durations[0], "fps": a["fps"], "size": [a["width"], a["height"]]},
        "out": {"duration": durations[1], "fps": b["fps"], "size": [b["width"], b["height"]]},
        "duration_delta_pct": round(abs(durations[1] - durations[0]) / max(0.001, durations[0]) * 100, 2),
        "fps_match": abs(a["fps"] - b["fps"]) < 0.05,
        "size_match": (a["width"], a["height"]) == (b["width"], b["height"]),
        "edge_iou_mean": round(sum(ious) / len(ious), 4) if ious else 0.0,
    }

    # 运动一致性：两段视频各自抽帧算逐帧位移，方向/速度对得上才算"动法相似"
    try:
        mr = motion_signature(ref_path, samples=motion_samples)["summary"]
        mo = motion_signature(out_path, samples=motion_samples)["summary"]
        ref_vec = (mr["mean_dx"], mr["mean_dy"])
        out_vec = (mo["mean_dx"], mo["mean_dy"])
        dot = ref_vec[0] * out_vec[0] + ref_vec[1] * out_vec[1]
        if mr["speed_px"] < 1.0 and mo["speed_px"] < 1.0:
            same_dir = True          # 两边都几乎不动，"动法"也算一致
        elif mr["speed_px"] < 1.0 or mo["speed_px"] < 1.0:
            same_dir = False
        else:
            same_dir = dot > 0
        speed_ref, speed_out = mr["speed_px"], mo["speed_px"]
        speed_ratio = round(speed_out / speed_ref, 3) if speed_ref > 0.2 else None
        report["motion"] = {
            "ref_label": mr["label"], "out_label": mo["label"],
            "ref_kind": mr.get("kind"), "out_kind": mo.get("kind"),
            "ref_vector": [mr["mean_dx"], mr["mean_dy"]],
            "out_vector": [mo["mean_dx"], mo["mean_dy"]],
            "direction_match": bool(same_dir),
            "speed_ratio": speed_ratio,
            "ref_rot_total": mr["total_rot"], "out_rot_total": mo["total_rot"],
        }
    except Exception as e:  # 运动比对失败不该把整份报告拖垮
        report["motion"] = {"error": f"{type(e).__name__}: {e}"}
    return report
