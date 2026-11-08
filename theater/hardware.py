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

"""硬件探测与后端/编码器选择：NVIDIA / AMD / Intel 通用。

本模块只做"决定用哪个后端、哪个编码器"，不负责真正加载模型或编码。
三条硬规则：

1. **全标准库**。没装任何厂商 CLI、没有任何 GPU 也能跑：探测失败只落到
   `unknown`，再由上层回退 CPU / libx264，不会抛异常。
2. **纯函数可单测**。`normalize_vendor` / `primary_vendor` / `select_encoder` /
   `encoder_quality_args` / `pick_draw_backend` 都不碰系统，测试见
   `tests/test_hardware.py`。
3. **探测结果缓存**。起一次 PowerShell / nvidia-smi 要几百毫秒，导出时每帧
   都探测会把时间全花在这儿，所以进程内只探测一次。

环境变量：
    GPU_ADAPTERS   形如 "NVIDIA GeForce RTX 5060 Ti|AMD Radeon RX 7800 XT"，
                   直接覆盖探测结果（无驱动 / 容器内 / 想强制指定时用）
    GPU_VENDOR     nvidia / amd / intel / unknown，只覆盖厂商判定
"""

import os
import platform
import shutil
import subprocess
from typing import Dict, List, Optional, Sequence

# ---- 厂商 -------------------------------------------------------------------
VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_UNKNOWN = "unknown"

# 关键字一律小写，匹配前统一 lowercase
_VENDOR_PATTERNS = (
    (VENDOR_NVIDIA, ("nvidia", "geforce", "rtx", "gtx", "quadro", "tesla", "titan")),
    (VENDOR_AMD, ("amd", "radeon", "advanced micro devices", "ati ", "instinct")),
    (VENDOR_INTEL, ("intel", "iris", "arc graphics", "intel arc", "uhd graphics", "hd graphics")),
)

# 独显优先于核显：核显+独显的笔记本/台式机，都应按独显选后端
_VENDOR_PRIORITY = (VENDOR_NVIDIA, VENDOR_AMD, VENDOR_INTEL)

# 虚拟/远程显示适配器，不算 GPU
_IGNORED_ADAPTER_HINTS = ("microsoft basic", "remote desktop", "virtual display", "parsec", "sunshine")


def normalize_vendor(text: str) -> str:
    """把显卡型号字符串归一成 nvidia / amd / intel / unknown。"""
    low = (text or "").lower()
    if not low:
        return VENDOR_UNKNOWN
    for vendor, keys in _VENDOR_PATTERNS:
        if any(k in low for k in keys):
            return vendor
    return VENDOR_UNKNOWN


def primary_vendor(adapters: Optional[Sequence[str]] = None) -> str:
    """从适配器列表里选"主显卡"厂商（NVIDIA > AMD > Intel）。"""
    env = os.environ.get("GPU_VENDOR", "").strip().lower()
    if env in (VENDOR_NVIDIA, VENDOR_AMD, VENDOR_INTEL):
        return env
    if adapters is None:
        adapters = list_adapters()
    found = {normalize_vendor(a) for a in adapters}
    for vendor in _VENDOR_PRIORITY:
        if vendor in found:
            return vendor
    return VENDOR_UNKNOWN


# ---- 探测（带缓存） ----------------------------------------------------------
_adapters_cache: Optional[List[str]] = None


def _run(cmd: List[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") if r.returncode == 0 else ""
    except Exception:
        return ""


def _windows_adapters() -> List[str]:
    """Win32_VideoController 枚举。AMD/Intel 没有好用的 CLI，只能走系统接口。"""
    ps = shutil.which("powershell") or shutil.which("pwsh")
    if not ps:
        return []
    out = _run([ps, "-NoProfile", "-NonInteractive", "-Command",
                "(Get-CimInstance Win32_VideoController).Name"])
    return [l.strip() for l in out.splitlines() if l.strip()]


def _linux_adapters() -> List[str]:
    if not shutil.which("lspci"):
        return []
    out = _run(["lspci"])
    names = []
    for line in out.splitlines():
        low = line.lower()
        if "vga compatible controller" in low or "3d controller" in low or "display controller" in low:
            # "01:00.0 VGA compatible controller: NVIDIA Corporation ..."
            names.append(line.split(":", 2)[-1].strip())
    return names


def list_adapters(refresh: bool = False) -> List[str]:
    """列出本机显卡型号（去重、过滤虚拟显示适配器）。"""
    global _adapters_cache
    if _adapters_cache is not None and not refresh:
        return _adapters_cache

    env = os.environ.get("GPU_ADAPTERS", "").strip()
    if env:
        _adapters_cache = [s.strip() for s in env.split("|") if s.strip()]
        return _adapters_cache

    names: List[str] = []

    # nvidia-smi 给出的型号最准（"NVIDIA GeForce RTX 5060 Ti"）
    smi = shutil.which("nvidia-smi")
    if smi:
        out = _run([smi, "--query-gpu=name", "--format=csv,noheader"], timeout=15)
        names += [l.strip() for l in out.splitlines() if l.strip()]

    system = platform.system()
    if system == "Windows":
        names += _windows_adapters()
    elif system == "Linux":
        names += _linux_adapters()

    cleaned: List[str] = []
    for n in names:
        low = n.lower()
        if any(h in low for h in _IGNORED_ADAPTER_HINTS):
            continue
        if n not in cleaned:
            cleaned.append(n)

    _adapters_cache = cleaned
    return _adapters_cache


def reset_cache() -> None:
    """清空探测缓存（测试/热插拔后使用）。"""
    global _adapters_cache
    _adapters_cache = None


# ---- 编码器 -----------------------------------------------------------------
# 每家硬件的 H.264 硬编码器不同，参数也不同，不能混用：
#   NVIDIA -> h264_nvenc（-rc vbr -cq）
#   AMD    -> h264_amf  （-rc cqp -qp_i/-qp_p）
#   Intel  -> h264_qsv  （-global_quality）
# 列表里保留别家编码器是给"多显卡 / 驱动异常"留的次选，最后一定落到 libx264。
ENCODER_PRIORITY: Dict[str, tuple] = {
    VENDOR_NVIDIA: ("h264_nvenc", "hevc_nvenc", "av1_nvenc", "h264_qsv", "h264_amf", "libx265", "libx264"),
    VENDOR_AMD: ("h264_amf", "hevc_amf", "av1_amf", "h264_qsv", "h264_nvenc", "libx265", "libx264"),
    VENDOR_INTEL: ("h264_qsv", "hevc_qsv", "av1_qsv", "h264_nvenc", "h264_amf", "libx265", "libx264"),
    VENDOR_UNKNOWN: ("h264_nvenc", "h264_qsv", "h264_amf", "libx265", "libx264"),
}


def select_encoder(vendor: str, available_text: str, preferred: Optional[str] = None,
                   codec: str = "h264") -> str:
    """在 ffmpeg 支持的编码器里，按厂商优先级挑一个。

    :param vendor:          nvidia / amd / intel / unknown
    :param available_text:  `ffmpeg -encoders` 的完整输出（空字符串表示 ffmpeg 不可用）
    :param preferred:       用户显式指定（VIDEO_ENCODER），确实可用时优先
    :param codec:           h264 / hevc / av1，用于过滤候选
    """
    order = ENCODER_PRIORITY.get(vendor or VENDOR_UNKNOWN, ENCODER_PRIORITY[VENDOR_UNKNOWN])
    avail = available_text or ""

    pref = (preferred or "").strip()
    if pref and pref.lower() != "auto" and pref in avail:
        return pref

    for enc in order:
        if not enc.startswith(codec):
            continue
        if enc in avail:
            return enc

    # 一个硬件编码器都没有（ffmpeg 缺失 / 精简版）：软编码兜底
    return "libx265" if codec == "hevc" else "libx264"


_HW_ENCODER_SUFFIX = ("_qsv", "_nvenc", "_amf", "_vaapi")


def is_hardware_encoder(encoder: str) -> bool:
    return (encoder or "").endswith(_HW_ENCODER_SUFFIX)


def encoder_quality_args(encoder: str, quality: str = "23") -> List[str]:
    """不同硬编码器的画质参数完全不同，这里统一给出正确的那一组。

    - qsv   : `-global_quality`（不吃 -crf）
    - nvenc : `-rc vbr -cq N -b:v 0`（CQ 模式；老版本 NVENC 也认 -cq）
    - amf   : `-rc cqp -qp_i/-qp_p`（AMF 没有 -crf）
    - libx264/libx265: `-crf`
    """
    q = str(quality)
    enc = (encoder or "").lower()
    if enc.endswith("_qsv"):
        return ["-global_quality", q, "-preset", "fast"]
    if enc.endswith("_nvenc"):
        return ["-rc", "vbr", "-cq", q, "-b:v", "0", "-preset", "p5"]
    if enc.endswith("_amf"):
        return ["-rc", "cqp", "-qp_i", q, "-qp_p", q, "-quality", "quality"]
    return ["-crf", q, "-preset", "veryfast"]


# ---- 绘图后端 ---------------------------------------------------------------
def suggest_openvino_device(vendor: str, device_override: Optional[str] = None) -> str:
    """OpenVINO 设备：只有 Intel 有 GPU 插件；其他平台回 CPU（仍可跑，只是慢）。"""
    dev = (device_override or "").strip()
    if dev and dev.upper() != "AUTO":
        return dev
    return "GPU" if (vendor or "").lower() == VENDOR_INTEL else "CPU"


def pick_draw_backend(vendor: str, available: Optional[Dict[str, bool]] = None,
                      is_windows: Optional[bool] = None) -> Dict[str, str]:
    """决定绘图走哪条后端。

    :param available: {"openvino": bool, "torch": bool, "directml": bool}
    :return: {"backend": "openvino"/"torch"/None, "device": ..., "reason": ...}

    优先级（按实测速度）：
      NVIDIA + torch        -> torch / cuda      （最快）
      AMD + torch + directml-> torch / dml       （Windows 上唯一的 AMD 加速路径）
      Intel + openvino      -> openvino / GPU    （核显/Arc）
      其余有 openvino       -> openvino / CPU    （任何平台都能跑）
      其余有 torch          -> torch / cpu
      都没有                -> None（上层给安装提示）
    """
    avail = available or {}
    ov = bool(avail.get("openvino"))
    tt = bool(avail.get("torch"))
    dml = bool(avail.get("directml"))
    if is_windows is None:
        is_windows = platform.system() == "Windows"

    v = (vendor or VENDOR_UNKNOWN).lower()
    if v == VENDOR_NVIDIA and tt:
        return {"backend": "torch", "device": "cuda", "reason": "NVIDIA 显卡 + PyTorch CUDA"}
    if v == VENDOR_AMD and tt and dml and is_windows:
        return {"backend": "torch", "device": "dml", "reason": "AMD 显卡 + Torch-DirectML"}
    if v == VENDOR_INTEL and ov:
        return {"backend": "openvino", "device": "GPU", "reason": "Intel 显卡 + OpenVINO GPU"}
    if ov:
        return {"backend": "openvino", "device": "CPU", "reason": "OpenVINO CPU（无可用 GPU 插件）"}
    if tt:
        return {"backend": "torch", "device": "cpu", "reason": "PyTorch CPU"}
    return {"backend": None, "device": None, "reason": "未安装 openvino 或 torch，无法绘图"}


# ---- 汇总 -------------------------------------------------------------------
def hardware_summary() -> Dict:
    """给 check_env / MCP 工具用的一站式摘要（不含 ffmpeg 探测结果）。"""
    adapters = list_adapters()
    vendor = primary_vendor(adapters)
    return {
        "adapters": adapters,
        "vendor": vendor,
        "openvino_device": suggest_openvino_device(vendor),
        "platform": platform.system(),
    }
