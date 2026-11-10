# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
# Generated: 2026-11-10 (Asia/Shanghai)
#
# Dual-licensed - choose exactly one:
#   (1) GNU AGPL-3.0-only  -> see LICENSE
#   (2) Commercial license -> see COMMERCIAL-LICENSE.md
# Decide with LICENSING.md. Under the AGPL-3.0 path, redistributing this
# software or exposing it as a network service obliges you to offer all users
# the complete corresponding source of your modified version (AGPL section 13).

"""CUDA / Tensor Core 加速，以及"按本机配置"的量化档位。

这一层只做三件事，且都能关（`ACCEL=off` 就退回 torch 默认）：

1. **把 Tensor Core 打开**：`float32_matmul_precision("high")`（走 TF32）、
   `allow_tf32`、`cudnn.benchmark`、fp16/bf16 归约。实测 4096³ matmul：
   fp32 不开 TF32 ≈ 16.8 TFLOP/s，bf16 ≈ 49 TFLOP/s —— 差的就是 Tensor Core。
2. **挑一个真能跑的 attention 内核**：本机（RTX 5060 Ti / sm_120 / torch 2.9.1+cu128）
   flash attention 报 `No available kernel`（cuDNN 的 SDPA 没给 sm_120 编译），
   所以按 flash → mem_efficient → math 逐个探测，取第一个真跑通的，记在 profile 里，
   后面 diffusers / LLM 都吃这个设置。
3. **按显存选量化档位**：显存 ≥24GB 用 bf16 全量；≥12GB 用 bf16 + LLM NF4 4bit
   （Qwen3-4B 从 ~8GB 压到 ~3GB，省下的显存留给 FLUX/SD）；<8GB 再降 fp16。

档位（`plan()["tier"]`，显存为 0（纯 CPU）时是 `cpu`）：

  | 显存 | tier | dtype | attention | LLM 量化 |
  |---|---|---|---|---|
  | ≥24GB | xl   | bf16 | flash（可用时） | none（可 env 改 4bit） |
  | ≥12GB | high | bf16 | mem_efficient | 4bit NF4 |
  | ≥8GB  | mid  | fp16 | mem_efficient | 4bit NF4 |
  | <8GB   | low  | fp16 | math | 4bit NF4 |

环境变量：
  ACCEL=auto|on|off            总开关（off 完全不改 torch 全局状态）
  ACCEL_LLM_QUANT=auto|4bit|8bit|none
  ACCEL_SD_DTYPE=auto|fp16|bf16
  ACCEL_TF32=1|0               是否允许 TF32（默认 1）
"""

import os
from typing import Dict, Optional

_CFG: Optional[dict] = None
_PLAN: Optional[dict] = None
_APPLIED = False


def _env_flag(name: str, default: str = "1") -> bool:
    return str(os.environ.get(name, default)).strip().lower() not in ("0", "false", "off", "no", "")


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def detect() -> Dict:
    """本机硬件画像（只读，不加载任何模型）。"""
    global _CFG
    if _CFG is not None:
        return dict(_CFG)

    cfg: Dict = {"cuda": False, "gpu": "", "cc": None, "vram_gb": 0.0,
                 "torch": "", "cuda_version": "", "tier": "cpu",
                 "cpu": "", "cores": 0, "threads": 0, "ram_gb": 0.0}
    try:
        import torch

        cfg["torch"] = torch.__version__
        cfg["cuda_version"] = getattr(torch.version, "cuda", "") or ""
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            cfg.update({"cuda": True, "gpu": props.name,
                        "cc": list(torch.cuda.get_device_capability(0)),
                        "vram_gb": round(props.total_memory / 1024 ** 3, 1)})
    except Exception:
        pass

    try:
        import platform

        cfg["cpu"] = platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "")
        cfg["cores"] = os.cpu_count() or 0
    except Exception:
        pass
    try:
        import psutil

        cfg["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
        cfg["threads"] = psutil.cpu_count(logical=True) or cfg["cores"]
    except Exception:
        cfg["threads"] = cfg["cores"]

    vram = float(cfg["vram_gb"] or 0)
    cfg["tier"] = ("cpu" if not cfg["cuda"] else
                   "xl" if vram >= 24 else
                   "high" if vram >= 12 else
                   "mid" if vram >= 8 else "low")
    _CFG = cfg
    return dict(cfg)


def _pick_attention() -> str:
    """真跑一遍 SDPA，返回本机可用的最快内核：flash / mem_efficient / math。"""
    try:
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:
        return "math"
    if not torch.cuda.is_available():
        return "math"

    q = torch.randn(1, 4, 128, 32, device="cuda", dtype=torch.float16)
    import warnings

    with warnings.catch_warnings():
        # 探测时 torch 会把每个不可用内核的原因整段打印出来，刷屏且没用
        warnings.simplefilter("ignore")
        for name, backend in (("flash", SDPBackend.FLASH_ATTENTION),
                              ("mem_efficient", SDPBackend.EFFICIENT_ATTENTION)):
            try:
                with sdpa_kernel(backend):
                    F.scaled_dot_product_attention(q, q, q)
                torch.cuda.synchronize()
                return name
            except Exception:
                continue
    return "math"


def plan() -> Dict:
    """加速 + 量化的具体配置（按 detect() 的档位、可用 env 覆盖）。"""
    global _PLAN
    if _PLAN is not None:
        return dict(_PLAN)

    cfg = detect()
    tier = cfg["tier"]
    enabled = bool(cfg["cuda"]) and _env_flag("ACCEL", "1")

    default_dtype = {"xl": "bf16", "high": "bf16", "mid": "fp16", "low": "fp16"}.get(tier, "fp32")
    dtype_name = str(os.environ.get("ACCEL_SD_DTYPE", "auto")).strip().lower()
    if dtype_name in ("auto", "", "default"):
        dtype_name = default_dtype
    if dtype_name not in ("fp16", "bf16", "fp32"):
        dtype_name = default_dtype

    attention = _pick_attention() if enabled else "math"

    quant = str(os.environ.get("ACCEL_LLM_QUANT", "auto")).strip().lower()
    if quant in ("auto", "", "default"):
        quant = "none" if tier == "xl" else ("4bit" if cfg["cuda"] else "none")
    if quant not in ("4bit", "8bit", "none"):
        quant = "4bit"

    _PLAN = {
        "enabled": enabled,
        "tier": tier,
        "dtype": dtype_name,
        "attention": attention,
        "tf32": enabled and _env_flag("ACCEL_TF32", "1"),
        "cudnn_benchmark": enabled,
        "llm_quant": quant if enabled else "none",
        "channels_last": enabled and dtype_name in ("fp16", "bf16"),
        "compile": False,          # Windows 上 torch.compile 要 triton，本机没装
        "vram_gb": cfg["vram_gb"],
    }
    return dict(_PLAN)


def torch_dtype():
    import torch

    return {"fp16": torch.float16, "bf16": torch.bfloat16,
            "fp32": torch.float32}[plan()["dtype"]]


def apply_global() -> Dict:
    """把全局加速开关拨到位（幂等，可重复调用）。返回生效的配置。"""
    global _APPLIED
    p = plan()
    if not p["enabled"]:
        return p
    import torch

    if not _APPLIED:
        if p["tf32"]:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if p["cudnn_benchmark"]:
            torch.backends.cudnn.benchmark = True
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        except Exception:
            pass
        # SDPA：本机 flash 不可用就关掉，省得每个 attention 都先失败一次再回落
        try:
            att = p["attention"]
            torch.backends.cuda.enable_flash_sdp(att == "flash")
            torch.backends.cuda.enable_mem_efficient_sdp(att in ("flash", "mem_efficient"))
            torch.backends.cuda.enable_math_sdp(att == "math")
        except Exception:
            pass
        _APPLIED = True
        print(f"[accel] {describe()}")
    return p


def llm_kwargs(note: bool = False):
    """给 transformers 的 from_pretrained 参数（含按档位选好的量化配置）。

    返回 (kwargs, note_text)；note_text 会写进日志/返回值，方便核对"到底量化了没"。
    """
    import torch

    p = plan()
    tier = p["tier"]
    compute = torch_dtype() if p["enabled"] else torch.float32
    if not p["enabled"]:
        return {"torch_dtype": torch.float32}, "未启用加速（ACCEL=off 或没有 CUDA）"

    if p["llm_quant"] == "none":
        return ({"torch_dtype": compute, "device_map": "cuda", "low_cpu_mem_usage": True},
                f"{tier} 档：权重 {p['dtype']} 全量驻留（未量化）")

    try:
        from transformers import BitsAndBytesConfig
    except Exception as e:
        return ({"torch_dtype": compute, "device_map": "cuda", "low_cpu_mem_usage": True},
                f"bitsandbytes 不可用（{type(e).__name__}），退回 {p['dtype']} 全量")

    try:
        import bitsandbytes as bnb  # noqa: F401
    except Exception as e:
        return ({"torch_dtype": compute, "device_map": "cuda", "low_cpu_mem_usage": True},
                f"bitsandbytes 导入失败（{type(e).__name__}），退回 {p['dtype']} 全量")

    if p["llm_quant"] == "8bit":
        cfg = BitsAndBytesConfig(load_in_8bit=True)
        note = f"{tier} 档：LLM INT8（显存约省 40%）"
    else:
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        note = f"{tier} 档：LLM NF4 4bit（Qwen3-4B 权重约 8GB → ~3GB，算用 {p['dtype']}）"

    kwargs = {"quantization_config": cfg, "device_map": {"": 0}, "low_cpu_mem_usage": True}
    return (kwargs, note) if note else (kwargs, "")


def sd_kwargs() -> Dict:
    """diffusers from_pretrained 的量化/精度参数（fp16/bf16 权重即"半精度量化"）。"""
    return {"torch_dtype": torch_dtype() if plan()["enabled"] else _fp32(),
            "use_safetensors": True}


def _fp32():
    import torch

    return torch.float32


def optimize_pipe(pipe) -> Dict:
    """给 diffusers 管道加内存布局优化，返回实际生效的项（便于自检）。"""
    applied = {"channels_last": False, "vae_slicing": False}
    p = plan()
    if not p["enabled"]:
        return applied
    try:
        import torch

        if p["channels_last"]:
            pipe.to(memory_format=torch.channels_last)
            applied["channels_last"] = True
    except Exception:
        pass
    return applied


def tensor_tflops(measure: bool = False) -> Dict:
    """Tensor Core 实测吞吐（4096³ matmul，约 0.5 s）。默认不跑，避免拖慢每次导入。"""
    if not cuda_available():
        return {}
    if not measure:
        return {}
    import time

    import torch

    out = {}
    for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        try:
            a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            b = torch.randn(4096, 4096, device="cuda", dtype=dtype)
            for _ in range(2):
                a @ b
            torch.cuda.synchronize()
            t = time.time()
            n = 5
            for _ in range(n):
                a @ b
            torch.cuda.synchronize()
            out[name] = round(2 * 4096 ** 3 / ((time.time() - t) / n) / 1e9, 1)
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            out[name] = f"ERR {type(e).__name__}"
    return out


def describe() -> str:
    cfg, p = detect(), plan()
    if not cfg["cuda"]:
        return f"CPU 推理（{cfg['cores']} 线程 / {cfg['ram_gb']} GB），未启用 Tensor Core"
    return (f"{cfg['gpu']} {cfg['vram_gb']} GB（sm_{cfg['cc'][0]}{cfg['cc'][1]}）"
            f" / {p['tier']} 档：权重 {p['dtype']}、attention {p['attention']}、"
            f"TF32 {'开' if p['tf32'] else '关'}、LLM {p['llm_quant']}")


def status() -> Dict:
    """给 check_env / MCP 用的一览。"""
    cfg, p = detect(), plan()
    return {"hardware": cfg, "plan": p, "text": describe()}