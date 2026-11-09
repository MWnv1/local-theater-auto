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

"""环境自检：python / 依赖 / ffmpeg(QSV) / 模型 / 字体 / 语音 / 缓存。

用法：
    python check_env.py            # 人读报表
    python check_env.py --json     # 机读（MCP 用）

设计说明：全部用 importlib 探测，**不 import openvino / torch / optimum**，
避免自检本身就要加载几 GB 模型或卡住几十秒。
"""

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # 允许直接 python check_env.py

FAIL = "FAIL"
WARN = "WARN"
OK = "OK"

results = []


def add(level, item, detail):
    results.append({"level": level, "item": item, "detail": detail})


def check_python():
    bits = 64 if sys.maxsize > 2 ** 32 else 32
    add(OK if bits == 64 else FAIL, "Python 位数", f"{bits}-bit / {platform.python_version()}")
    if sys.version_info < (3, 9):
        add(FAIL, "Python 版本", f"{platform.python_version()}，需要 >= 3.9")


def check_packages():
    pkgs = [
        ("mcp", "mcp"),
        ("PIL", "pillow"),
        ("numpy", "numpy"),
        ("pyttsx3", "pyttsx3"),
        ("rembg", "rembg"),
        ("onnxruntime", "onnxruntime"),
        ("openvino", "openvino"),
        ("optimum", "optimum"),
        ("transformers", "transformers"),
        ("diffusers", "diffusers"),
        ("nncf", "nncf"),
        # 声音克隆（F5-TTS）这条链
        ("torchaudio", "torchaudio"),
        ("f5_tts", "f5-tts"),
        ("vocos", "vocos"),
        ("soundfile", "soundfile"),
        ("librosa", "librosa"),
        ("datasets", "datasets"),
    ]
    for mod, pipname in pkgs:
        spec = importlib.util.find_spec(mod)
        if spec is None:
            add(FAIL, f"依赖 {pipname}", "未安装：python -m pip install -r requirements.txt")
        else:
            add(OK, f"依赖 {pipname}", "已安装")


def check_ffmpeg():
    from theater import ffmpeg_path, ffprobe_path

    ff = shutil.which(ffmpeg_path()) or (ffmpeg_path() if os.path.exists(ffmpeg_path()) else None)
    if not ff:
        add(FAIL, "ffmpeg", "未找到，请把 ffmpeg.exe 所在 bin 目录加入 PATH，或设 FFMPEG_PATH")
        return
    add(OK, "ffmpeg", ff)
    try:
        r = subprocess.run([ff, "-hide_banner", "-version"], capture_output=True, text=True, timeout=20)
        first = (r.stdout or r.stderr or "").strip().splitlines()[0]
        add(OK, "ffmpeg 版本", first)
    except Exception as e:
        add(WARN, "ffmpeg 版本", f"执行异常：{e}")

    fp = shutil.which(ffprobe_path()) or (ffprobe_path() if os.path.exists(ffprobe_path()) else None)
    add(OK if fp else WARN, "ffprobe", fp or "未找到（时长解析会回退到解析 ffmpeg 输出）")

    try:
        r = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=30)
        text = (r.stdout or "") + (r.stderr or "")

        from theater import hardware

        vendor = hardware.primary_vendor()
        chosen = hardware.select_encoder(
            vendor, text, preferred=os.environ.get("VIDEO_ENCODER", "auto")
        )
        hw_ok = hardware.is_hardware_encoder(chosen)
        add(OK if hw_ok else WARN, "选定视频编码器",
            f"{chosen}（{vendor} 平台" + ("，硬编" if hw_ok else "，无可用硬件编码器 → CPU 软编") + "）")

        for enc in ("h264_nvenc", "h264_amf", "h264_qsv", "h264_videotoolbox", "libx264", "aac"):
            add(OK if enc in text else WARN, f"编码器 {enc}", "可用" if enc in text else "不可用")

        # 真编一次 3 帧黑场，确认这个编码器不是"只有名字没有实现"
        # 注意：NVENC 有最小帧尺寸限制，64x64 会报 "Frame Dimension less than
        # the minimum supported value"，必须用 320x240 这类正常尺寸来测。
        try:
            args = hardware.encoder_quality_args(chosen, "23")
            r2 = subprocess.run(
                [ff, "-hide_banner", "-v", "error", "-f", "lavfi",
                 "-i", "color=c=black:s=320x240:d=0.1", "-c:v", chosen] + args + ["-f", "null", "-"],
                capture_output=True, text=True, timeout=30,
            )
            if r2.returncode == 0:
                add(OK, f"{chosen} 实际可用性", "编码测试通过")
            else:
                tail = (r2.stderr or "").strip().splitlines()[-2:]
                add(WARN, f"{chosen} 实际可用性", "编码测试失败：" + " / ".join(tail))
        except Exception as e:
            add(WARN, f"{chosen} 实际可用性", f"测试异常：{e}")
    except Exception as e:
        add(WARN, "编码器检测", f"异常：{e}")


def check_hardware():
    """显卡厂商 + 绘图后端判定（NVIDIA / AMD / Intel 通用）。"""
    from theater import hardware

    adapters = hardware.list_adapters()
    if adapters:
        add(OK, "显卡", "、".join(adapters[:4]))
    else:
        add(WARN, "显卡", "未识别（不影响运行，默认 CPU + libx264）")

    vendor = hardware.primary_vendor(adapters)
    add(OK if vendor != hardware.VENDOR_UNKNOWN else WARN, "主显卡厂商", vendor)

    avail = {
        "openvino": importlib.util.find_spec("openvino") is not None,
        "torch": importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("diffusers") is not None,
        "directml": importlib.util.find_spec("torch_directml") is not None,
    }
    picked = hardware.pick_draw_backend(vendor, avail, os.name == "nt")
    if picked["backend"]:
        add(OK, "绘图后端", f"{picked['backend']} / {picked['device']}（{picked['reason']}）")
    else:
        add(FAIL, "绘图后端", picked["reason"] + "；见 README 第 11 节")


def check_models():
    from theater import MODEL_ROOT, hardware

    # 后端决定模型格式：torch 要 diffusers 目录（model_index.json），
    # OpenVINO 要 IR（.xml/.bin），两者不能混用，所以分开查。
    ov_dir = os.environ.get("DRAW_MODEL_DIR", os.path.join(MODEL_ROOT, "flux-schnell-int4-ov"))
    torch_dir = os.environ.get("TORCH_MODEL_DIR", os.path.join(MODEL_ROOT, "flux-schnell"))
    backend = ""
    try:
        import theater.ai_draw as ai_draw

        backend = ai_draw.resolve_backend()["backend"]
    except Exception:
        backend = "openvino"

    primary = torch_dir if backend == "torch" else ov_dir
    if os.path.isdir(primary) and os.listdir(primary):
        if backend == "torch":
            ok = os.path.exists(os.path.join(primary, "model_index.json"))
            add(OK if ok else WARN, "绘图模型", f"{primary}（diffusers 目录）" if ok
                else f"{primary} 缺 model_index.json，torch 后端无法加载")
        else:
            files = os.listdir(primary)
            has_ir = any(f.endswith((".xml", ".bin")) for f in files) or any(
                os.path.isdir(os.path.join(primary, f)) for f in files
            )
            add(OK if has_ir else WARN, "绘图模型", f"{primary}（{len(files)} 项）")
    else:
        add(FAIL, "绘图模型", f"不存在或为空：{primary}（当前后端 {backend}）")

    # 另一套格式不要求存在，只提示
    other = ov_dir if backend == "torch" else torch_dir
    if os.path.isdir(other) and os.listdir(other):
        add(OK, "备用绘图模型", f"{other}（当前后端用不到它）")

    llm_torch = os.environ.get("LLM_TORCH_DIR", os.path.join(MODEL_ROOT, "qwen3-4b"))
    llm_ov = os.environ.get("LLM_OV_DIR", os.path.join(MODEL_ROOT, "qwen3-4b-int4-ov"))
    ready = [d for d in (llm_torch, llm_ov) if os.path.isdir(d) and os.listdir(d)]
    add(OK if ready else WARN, "剧本生成模型", ready[0] if ready
        else f"未就绪（不影响 MCP 工具；仅影响自动写剧本）：{llm_torch} 或 {llm_ov}")

    # 声音克隆：F5-TTS 权重 + vocoder（两个目录都要齐）
    f5_dir = os.environ.get("F5_MODEL_DIR", os.path.join(MODEL_ROOT, "f5-tts", "F5TTS_v1_Base"))
    f5_ckpt = os.path.join(f5_dir, "model_1250000.safetensors")
    f5_vocab = os.path.join(f5_dir, "vocab.txt")
    add(OK if os.path.exists(f5_ckpt) and os.path.exists(f5_vocab) else WARN,
        "克隆模型 F5-TTS", f5_dir if os.path.exists(f5_ckpt) else f"缺 model_1250000.safetensors：{f5_dir}")

    vocos_dir = os.environ.get("VOCOS_DIR", os.path.join(MODEL_ROOT, "vocos-mel-24khz"))
    need = ("config.yaml", "pytorch_model.bin")
    ok = all(os.path.exists(os.path.join(vocos_dir, f)) for f in need)
    add(OK if ok else WARN, "声码器 vocos", vocos_dir if ok else f"缺 config.yaml/pytorch_model.bin：{vocos_dir}")

    # 参考视频 → 相似视频用的 SD1.5
    sd_dir = os.environ.get("SIM_SD_MODEL", os.path.join(MODEL_ROOT, "sd15-base"))
    ok = os.path.exists(os.path.join(sd_dir, "model_index.json"))
    add(OK if ok else WARN, "重绘模型 SD1.5", sd_dir if ok
        else f"缺 model_index.json（参考视频重绘不可用）：{sd_dir}")

    u2net = os.path.join(MODEL_ROOT, "rembg", "u2net.onnx")
    if os.path.exists(u2net):
        add(OK, "抠图模型 u2net.onnx", u2net)
    else:
        add(WARN, "抠图模型 u2net.onnx", f"缺失，将自动降级为白底抠图（建议放到 {u2net}）")


def check_new_capabilities():
    """声音克隆 / 骨骼动画 / 素材库 / 参考视频重绘：四项新能力的就绪状态。"""
    from theater import library, tts_clone, video_ref

    st = tts_clone.status()
    ready = st.get("clone_available")
    add(OK if ready else WARN, "配音引擎",
        f"{st.get('engine_resolved')}（clone 可用={ready}，setting={st.get('engine_setting')}）")

    from theater import animator, rigging  # noqa: F401

    add(OK, "骨骼绑定/动作", "动作：" + "、".join(sorted(animator.ACTIONS)))

    s = library.stats()
    add(OK if s.get("total") else WARN, "素材/音效库",
        f"{s.get('total', 0)} 条 / {s.get('size_mb', 0)} MB @ {s.get('root', '')}")

    ok = video_ref._sd_available()
    add(OK if ok else WARN, "参考视频重绘", "SD1.5 img2img 就绪" if ok else "SD1.5 不可用，只能做分析+复制音轨")


def check_fonts_and_tts():
    try:
        from theater.frame_compose import FONT_CANDIDATES
    except Exception as e:
        add(FAIL, "字体模块", f"无法导入 theater.frame_compose（缺 Pillow？）：{e}")
        FONT_CANDIDATES = [
            r"C:\Windows\Fonts\msyh.ttc",
            r"C:\Windows\Fonts\msyhbd.ttc",
            r"C:\Windows\Fonts\simhei.ttf",
        ]

    found = [p for p in FONT_CANDIDATES if os.path.exists(p)]
    add(OK if found else WARN, "中文字体", found[0] if found else "未找到 msyh/simhei，字幕可能显示方块")

    try:
        import pyttsx3

        eng = pyttsx3.init()
        voices = eng.getProperty("voices") or []
        add(OK if voices else WARN, "系统语音包", f"{len(voices)} 个发音人：" +
            "、".join((v.name or "")[:28] for v in voices[:3]))
        eng.stop()
    except Exception as e:
        add(FAIL, "pyttsx3 初始化", f"{e}")


def check_cache():
    from theater import CACHE_ROOT, ensure_dirs

    try:
        ensure_dirs()
        probe = os.path.join(CACHE_ROOT, ".write_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(probe)
        add(OK, "缓存目录可写", CACHE_ROOT)
    except Exception as e:
        add(FAIL, "缓存目录可写", f"{CACHE_ROOT}：{e}")


def _safe(fn, label):
    """单个检查项崩了也不能让整个自检挂掉。"""
    try:
        fn()
    except Exception as e:
        add(FAIL, f"检查项 {label}", f"{type(e).__name__}: {e}")


def main():
    # Windows 控制台默认 GBK，这里尽量切到 UTF-8，避免中文乱码
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if os.name == "nt":
            os.system("chcp 65001 > nul")
    except Exception:
        pass

    _safe(check_python, "python")
    _safe(check_packages, "packages")
    _safe(check_hardware, "hardware")
    _safe(check_cache, "cache")
    _safe(check_ffmpeg, "ffmpeg")
    _safe(check_models, "models")
    _safe(check_new_capabilities, "new-capabilities")
    _safe(check_fonts_and_tts, "fonts/tts")

    if "--json" in sys.argv:
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    else:
        print("=" * 72)
        print("本地小剧场 环境自检")
        print("=" * 72)
        for r in results:
            line = f"[{r['level']:4}] {r['item']:<22} {r['detail']}"
            print(line)
        fails = [r for r in results if r["level"] == FAIL]
        warns = [r for r in results if r["level"] == WARN]
        print("-" * 72)
        print(f"FAIL {len(fails)} 项, WARN {len(warns)} 项")
        if fails:
            print("→ 先把 FAIL 修掉再跑流程；WARN 可以带病运行（会自动降级）。")

    sys.exit(1 if any(r["level"] == FAIL for r in results) else 0)


if __name__ == "__main__":
    main()
