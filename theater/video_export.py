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

"""视频导出（QSV 硬编码 / 可回退）。

修复的点：
1. **`h264_qsv` 不认识 `-crf`**（QSV 用 `-global_quality`），原稿的 `-crf 23`
   会被忽略甚至报错；这里按编码器分别拼参。
2. 所有 ffmpeg 调用都检查 returncode，失败抛异常并把 stderr 尾部带出来，
   不再静默产出空文件。
3. 画面统一 `scale + pad` 到 1920×1080（原先 1024×1024 直接拉伸会变形），
   并强制 `-pix_fmt yuv420p`、`-r 30`（否则微信/部分播放器打不开）。
4. 音频拼接改为"解码 → 统一 48kHz/双声道 → 重编码 PCM"，替代对 WAV 做
   `-c copy`（后者常出现爆音与时长漂移）。
5. 成片拼接先用 `-c copy`，随后用 ffprobe 校验总时长；对不上自动走重编码兜底。
6. QSV 不可用时自动回退 `libx264`（并打印告警），不会直接崩。
"""

import os
import re
import subprocess
from typing import Dict, List

from . import ffmpeg_path, ffprobe_path, subdir

_CLIP_DIR = subdir("video_clips")

VIDEO_ENCODER = os.environ.get("VIDEO_ENCODER", "h264_qsv")
QSV_QUALITY = os.environ.get("QSV_QUALITY", "23")        # -global_quality，0~51
X264_CRF = os.environ.get("X264_CRF", "20")
FPS = os.environ.get("VIDEO_FPS", "30")
W = int(os.environ.get("VIDEO_W", "1920"))
H = int(os.environ.get("VIDEO_H", "1080"))

_encoders_cache = None


def _ffmpeg_encoders() -> str:
    global _encoders_cache
    if _encoders_cache is None:
        try:
            r = subprocess.run(
                [ffmpeg_path(), "-hide_banner", "-encoders"],
                capture_output=True, text=True, encoding="utf-8", errors="ignore",
            )
            _encoders_cache = (r.stdout or "") + (r.stderr or "")
        except Exception as e:
            _encoders_cache = ""
            print(f"[warn] 无法执行 ffmpeg -encoders：{e}")
    return _encoders_cache


def resolve_encoder(preferred: str = None) -> str:
    """确认编码器可用，否则按 qsv → libx264 顺序回退。"""
    pref = preferred or VIDEO_ENCODER
    available = _ffmpeg_encoders()
    if pref and pref in available:
        return pref
    for cand in ("h264_qsv", "hevc_qsv", "libx264"):
        if cand in available:
            if cand != pref:
                print(f"[warn] 编码器 {pref} 不可用，已回退到 {cand}")
            return cand
    return "libx264"


def _run(cmd: List[str], what: str) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        tail = (r.stderr or "").strip().splitlines()[-12:]
        raise RuntimeError(f"{what} 失败（exit={r.returncode}）：\n" + "\n".join(tail))


def _get_audio_duration(audio_path: str) -> float:
    """优先 ffprobe；失败时用标准库 wave（wav 无需外部工具）；再退回解析 ffmpeg 输出。"""
    try:
        r = subprocess.run(
            [ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", audio_path],
            capture_output=True, text=True, timeout=30,
        )
        val = (r.stdout or "").strip()
        if r.returncode == 0 and val:
            return max(0.05, float(val))
    except Exception:
        pass

    if audio_path.lower().endswith(".wav"):
        try:
            import wave

            with wave.open(audio_path, "rb") as w:
                frames = w.getnframes()
                rate = w.getframerate()
                if rate:
                    return max(0.05, frames / float(rate))
        except Exception:
            pass

    try:
        r = subprocess.run(
            [ffmpeg_path(), "-hide_banner", "-i", audio_path],
            capture_output=True, text=True, encoding="utf-8", errors="ignore",
        )
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", r.stderr or "")
        if m:
            h, mi, s = m.groups()
            return int(h) * 3600 + int(mi) * 60 + float(s)
    except Exception:
        pass
    return 3.0


def _concat_audio(files: List[str], out_wav: str) -> str:
    """把多段音频拼成一个 wav（统一 48k/双声道 PCM）。"""
    if len(files) == 1:
        return files[0]
    lst = out_wav + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '" + p.replace("'", "'\\''") + "'\n")
    _run([
        ffmpeg_path(), "-y", "-f", "concat", "-safe", "0", "-i", lst,
        "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le", out_wav,
    ], "音频拼接")
    return out_wav


def _build_clip(frame_path: str, audio_path: str, duration: float, out_path: str) -> None:
    enc = resolve_encoder()
    cmd = [ffmpeg_path(), "-y",
           "-loop", "1", "-framerate", FPS, "-i", frame_path,
           "-i", audio_path,
           "-vf", f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
                  f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
           "-c:v", enc]
    if enc.endswith("_qsv"):
        cmd += ["-global_quality", QSV_QUALITY, "-preset", "fast"]
    else:
        cmd += ["-crf", X264_CRF, "-preset", "veryfast"]
    cmd += ["-pix_fmt", "yuv420p", "-r", FPS,
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
            "-t", f"{duration:.3f}", "-shortest", out_path]
    _run(cmd, f"镜头片段编码（{enc}）")


def export_final_video(script_struct: dict, frames: Dict[int, str], tts_audio: Dict[int, list],
                       out_path: str) -> str:
    """按镜头拼接画面 + 配音，输出成片，返回绝对路径。"""
    if not script_struct.get("scenes"):
        raise ValueError("剧本为空，无法导出")

    clip_list: List[str] = []
    total_expect = 0.0

    for scene in script_struct["scenes"]:
        scene_id = int(scene["scene_id"])
        if scene_id not in frames:
            raise RuntimeError(f"镜头 {scene_id} 缺少画面，请先调用合成步骤")
        frame_path = frames[scene_id]
        audios = tts_audio.get(scene_id, []) or []

        dur = 0.5  # 镜头留白
        audio_files: List[str] = []
        for a in audios:
            d = _get_audio_duration(a["audio_path"])
            dur += d + 0.15
            audio_files.append(a["audio_path"])

        if not audio_files:
            # 无台词镜头：静音片段
            silent = os.path.join(_CLIP_DIR, f"silence_{scene_id}.wav")
            if not os.path.exists(silent):
                _run([ffmpeg_path(), "-y", "-f", "lavfi", "-i",
                      "anullsrc=channel_layout=stereo:sample_rate=48000",
                      "-t", f"{dur:.2f}", "-c:a", "pcm_s16le", silent], "生成静音")
            audio_path = silent
        else:
            audio_path = _concat_audio(audio_files, os.path.join(_CLIP_DIR, f"audio_{scene_id}.wav"))

        clip_path = os.path.join(_CLIP_DIR, f"clip_{scene_id}.mp4")
        _build_clip(frame_path, audio_path, dur, clip_path)
        clip_list.append(clip_path)
        total_expect += dur

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    concat_file = os.path.join(_CLIP_DIR, "concat_list.txt")
    with open(concat_file, "w", encoding="utf-8") as f:
        for cp in clip_list:
            f.write("file '" + cp.replace("'", "'\\''") + "'\n")

    _run([ffmpeg_path(), "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
          "-c", "copy", "-movflags", "+faststart", out_path], "成片拼接")

    # 时长校验，不一致则重编码兜底
    got = _get_audio_duration(out_path)
    if got < max(0.5, total_expect * 0.8):
        print(f"[warn] 拼接后时长 {got:.1f}s 与预期 {total_expect:.1f}s 偏差过大，改用重编码拼接")
        _run([ffmpeg_path(), "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
              "-c:v", resolve_encoder(), "-c:a", "aac", "-pix_fmt", "yuv420p",
              "-movflags", "+faststart", out_path], "成片重编码拼接")
        got = _get_audio_duration(out_path)

    if got <= 0.5:
        raise RuntimeError("导出失败：成片时长异常（可能 ffmpeg 未能读取输入）")

    return out_path, round(got, 2)
