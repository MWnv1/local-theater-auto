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

r"""AI 声音克隆配音（F5-TTS），替换 Windows SAPI。

为什么换：SAPI 只有固定的系统发音人，同一角色换个镜头就"变声"，也没法
按作者自己的声音做旁白。F5-TTS 是 zero-shot 克隆模型：给 5~10 秒参考音频
（可选再给这段音频的文字），就能用那个音色念任意台词，全离线。

引擎选择（TTS_ENGINE）：
  auto   默认。克隆模型可用就用克隆，不可用自动退回 SAPI（并在返回值里注明）
  clone  强制克隆（不可用时报错，而不是偷偷降级）
  sapi   强制系统语音（旧行为，装不上模型时的保底）

参考音频约定（TTS_REF_DIR，默认 cache/voices）：
  <角色名>.wav        角色音色（用 WAV 16k/24k 单声道最佳，3~10 秒）
  <角色名>.txt        可选：这段音频念的文字，给了对齐更准
  或直接用 TTS_VOICE_MAP=小周=D:\voices\zhou.wav,旁白=narrator.wav

环境变量：
  F5_MODEL_DIR      F5-TTS 权重目录（默认 <MODEL_ROOT>/f5-tts）
  VOCOS_DIR         vocos 声码器目录（默认 <MODEL_ROOT>/vocos-mel-24khz）
  TTS_DEVICE        cuda | cpu（默认 cuda 可用则 cuda）
  TTS_SPEED         语速倍率（默认 1.0）
"""

import hashlib
import os
import threading
from typing import Dict, List, Optional

from . import MODEL_ROOT, safe_name, subdir, text_key

ENGINE = os.environ.get("TTS_ENGINE", "auto").strip().lower() or "auto"
REF_DIR = os.environ.get("TTS_REF_DIR", subdir("voices"))
F5_MODEL_DIR = os.environ.get("F5_MODEL_DIR", os.path.join(MODEL_ROOT, "f5-tts"))
VOCOS_DIR = os.environ.get("VOCOS_DIR", os.path.join(MODEL_ROOT, "vocos-mel-24khz"))
TTS_DEVICE = os.environ.get("TTS_DEVICE", "auto")
TTS_SPEED = float(os.environ.get("TTS_SPEED", "1.0"))

CLONE_ENGINES = ("f5",)
_AUDIO_DIR = subdir("audio")
_lock = threading.Lock()
_last_engine = "sapi"
_f5 = None


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def parse_voice_map(raw: str) -> Dict[str, str]:
    """解析 "小周=zhou.wav,旁白=narrator.wav"（相对路径按 REF_DIR 解析）。"""
    out: Dict[str, str] = {}
    for pair in (raw or "").split(","):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k, v = k.strip(), v.strip()
        if not k or not v:
            continue
        if not os.path.isabs(v):
            v = os.path.join(REF_DIR, v)
        out[k] = v
    return out


VOICE_MAP: Dict[str, str] = parse_voice_map(os.environ.get("TTS_VOICE_MAP", ""))


def ref_for(character: str, voice_map: Optional[Dict[str, str]] = None) -> str:
    """角色 → 参考音频路径。找不到返回空串（调用方据此退回 SAPI）。"""
    m = voice_map if voice_map is not None else VOICE_MAP
    if character in m and os.path.exists(m[character]):
        return m[character]
    for cand in (f"{safe_name(character, 'char')}.wav", f"{safe_name(character, 'char')}.flac"):
        p = os.path.join(REF_DIR, cand)
        if os.path.exists(p):
            return p
    return ""


def ref_text_for(ref_audio: str) -> str:
    """参考音频配套文字（可选）。没有就返回空串。"""
    if not ref_audio:
        return ""
    base = os.path.splitext(ref_audio)[0]
    p = base + ".txt"
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception:
            return ""
    return ""


# ---------------------------------------------------------------------------
# 引擎可用性
# ---------------------------------------------------------------------------
def _f5_ckpt() -> str:
    cand = os.path.join(F5_MODEL_DIR, "F5TTS_v1_Base", "model_1250000.safetensors")
    if os.path.exists(cand):
        return cand
    cand2 = os.path.join(F5_MODEL_DIR, "F5TTS_v1_Base", "model_1250000.pt")
    return cand2 if os.path.exists(cand2) else ""


def _f5_vocab() -> str:
    p = os.path.join(F5_MODEL_DIR, "F5TTS_v1_Base", "vocab.txt")
    return p if os.path.exists(p) else ""


def _clone_available() -> bool:
    """克隆引擎是否真的可用：包能导入 + 权重/词表/声码器都在。"""
    if not _f5_ckpt() or not _f5_vocab():
        return False
    if not os.path.exists(os.path.join(VOCOS_DIR, "pytorch_model.bin")):
        return False
    try:
        import f5_tts.api  # noqa: F401
        return True
    except Exception:
        return False


def resolve_engine(engine: Optional[str] = None) -> str:
    """把 auto/clone/sapi 归一到实际可用的引擎名（clone 或 sapi）。"""
    want = (engine or ENGINE or "auto").strip().lower()
    if want == "sapi":
        return "sapi"
    if want == "clone":
        return "clone" if _clone_available() else "sapi"
    return "clone" if _clone_available() else "sapi"


def last_engine() -> str:
    """最近一次实际使用的引擎（"clone" / "sapi"），供上层展示。"""
    return _last_engine


def cache_key(text: str, ref_audio: str, engine: str) -> str:
    """缓存键：文本 + 引擎 + 参考音频指纹（换音色必须换缓存）。"""
    ref_sig = ""
    if ref_audio:
        # 路径本身也要进 key：参考音频还没生成/被删掉时，换音色依然不能命中旧缓存
        size = os.path.getsize(ref_audio) if os.path.exists(ref_audio) else 0
        ref_sig = f"{os.path.abspath(ref_audio)}|{size}"
    return hashlib.md5(f"{engine}|{text}|{ref_sig}".encode("utf-8")).hexdigest()[:10]


# ---------------------------------------------------------------------------
# F5-TTS 封装
# ---------------------------------------------------------------------------
def _vocoder_local_path() -> Optional[str]:
    """声码器目录（F5-TTS 支持 `vocoder_local_path`，不需要联网）。

    目录里必须有 config.yaml + pytorch_model.bin；缺一个就返回 None，
    让 F5 走它自己的下载逻辑（此时通常也意味着没网，会给出清楚的错误）。
    """
    need = (os.path.join(VOCOS_DIR, "config.yaml"), os.path.join(VOCOS_DIR, "pytorch_model.bin"))
    return VOCOS_DIR if all(os.path.exists(p) for p in need) else None


def _patch_f5_audio_io() -> None:
    """修正 F5-TTS 的音频读取方式。

    F5 用 `torchaudio.load` 读参考音频，而 torchaudio 2.9 把读文件这件事交给了
    torchcodec —— 它需要 ffmpeg 的**共享** DLL。本项目用的是静态 ffmpeg（只有 exe，
    没有 dll），于是 torchcodec 在 Windows 上必挂（WinError 127）。
    这里把 `torchaudio.load` 换成 soundfile 实现（wav/flac 都能读），重采样仍用
    `torchaudio.functional.resample`（纯 torch 运算，不需要任何 DLL）。
    """
    try:
        import torchaudio
        from f5_tts.infer import utils_infer as ui
    except Exception:
        return
    if getattr(torchaudio, "_theater_soundfile_load", False):
        return

    def _load(path, *args, **kwargs):
        import soundfile as sf
        import torch

        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(data.T.copy()), sr

    torchaudio.load = _load
    torchaudio._theater_soundfile_load = True
    ui.torchaudio = torchaudio


def _get_f5():
    """惰性加载 F5-TTS（一次加载常驻，避免每句都重新初始化）。"""
    global _f5
    if _f5 is not None:
        return _f5
    with _lock:
        if _f5 is not None:
            return _f5
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        _patch_f5_audio_io()

        import torch
        from f5_tts.api import F5TTS

        device = TTS_DEVICE
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        _f5 = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=_f5_ckpt(),
            vocab_file=_f5_vocab(),
            # 传目录即进入"本地 vocos"分支，不碰 huggingface.co
            vocoder_local_path=_vocoder_local_path(),
            device=device,
            # 官方 F5TTS_v1_Base 的 ckpt 里只有 ema_model.* 权重，必须开 use_ema
            use_ema=True,
        )
        return _f5


def _ensure_ref_text(tts, ref_audio: str) -> str:
    """参考音频的配套文字：优先同名 .txt，没有就当场转写并缓存下来。

    F5 的 zero-shot 需要"参考音频 + 参考文字"。不提供文字时它会自己调 whisper
    转写——那要额外下一次模型，离线环境直接失败。所以这里显式处理：
    能转写就写回 .txt，下次零成本；转写不了就把原因打出来，让调用方走兜底。
    """
    text = ref_text_for(ref_audio)
    if text:
        return text

    print("[warn] 参考音频缺少同名 .txt，尝试自动转写（首次需要 whisper 模型，离线会失败）")
    try:
        text = (tts.transcribe(ref_audio) or "").strip()
    except Exception as e:
        print(f"[warn] 自动转写失败：{type(e).__name__}: {e}")
        return ""
    if text:
        try:
            with open(os.path.splitext(ref_audio)[0] + ".txt", "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[tts] 已缓存参考文字：{text[:30]}…")
        except Exception:
            pass
    return text


def _synth_clone(text: str, ref_audio: str, out_path: str, ref_text: str = "") -> str:
    """用参考音色合成一句话，写到 out_path。"""
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"参考音频不存在：{ref_audio}")

    tts = _get_f5()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ref_text = ref_text or _ensure_ref_text(tts, ref_audio)
    tts.infer(
        ref_file=ref_audio,
        ref_text=ref_text,
        gen_text=text,
        file_wave=out_path,
        speed=TTS_SPEED,
        show_info=lambda *a, **k: None,
    )
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("F5-TTS 未产出音频")
    return out_path


def _synth_sapi(text: str, out_path: str, character: str = "") -> str:
    """系统 SAPI 兜底（沿用 local_tts 已修好的逐句重开引擎逻辑）。"""
    from . import local_tts

    with _lock:
        local_tts._synthesize_once(text, out_path, _sapi_hint(character), local_tts.TTS_RATE)
        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            local_tts._synthesize_once(text, out_path, None, local_tts.TTS_RATE)
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(f"SAPI 未生成有效音频：{out_path}（文本：{text[:40]}）")
    return out_path


def _sapi_hint(character: str) -> Optional[str]:
    from . import local_tts

    return local_tts.VOICE_MAP.get(character) or local_tts.TTS_VOICE or None


# ---------------------------------------------------------------------------
# 对外
# ---------------------------------------------------------------------------
def synth(text: str, ref_audio: str = "", out_path: str = "", engine: Optional[str] = None,
          character: str = "", ref_text: str = "", force: bool = False) -> str:
    """合成一句台词，返回 wav 绝对路径。

    克隆失败会退回 SAPI（并在 `last_engine()` 里体现），不会让整条管线中断。
    """
    global _last_engine
    text = (text or "").strip()
    if not text:
        raise ValueError("文本为空")

    eng = resolve_engine(engine)
    out_path = out_path or os.path.join(_AUDIO_DIR, f"{eng}_{cache_key(text, ref_audio, eng)}.wav")

    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        _last_engine = eng
        return os.path.abspath(out_path)

    if eng == "clone" and ref_audio:
        try:
            _synth_clone(text, ref_audio, out_path, ref_text or ref_text_for(ref_audio))
            _last_engine = "clone"
            return os.path.abspath(out_path)
        except Exception as e:
            # 克隆失败不静默：写一条可见日志，然后退回 SAPI
            print(f"[warn] 声音克隆失败，退回 SAPI：{type(e).__name__}: {e}")

    _synth_sapi(text, out_path, character)
    _last_engine = "sapi"
    return os.path.abspath(out_path)


def gen_all_tts(script_struct: dict, voice_map: Optional[Dict[str, str]] = None,
                force: bool = False) -> Dict[int, List[dict]]:
    """批量配音。返回结构与 local_tts.gen_all_tts 完全一致（可直接替换）。"""
    audio_map: Dict[int, List[dict]] = {}
    eng = resolve_engine()
    for scene in script_struct.get("scenes", []):
        scene_id = int(scene["scene_id"])
        rows: List[dict] = []
        for idx, line in enumerate(scene.get("lines", [])):
            text = (line.get("text") or "").strip()
            if not text:
                continue
            char = line.get("character", "旁白")
            ref = ref_for(char, voice_map) if eng == "clone" else ""
            key = f"{char}{text}"
            fname = f"s{scene_id}_{idx}_{eng}_{text_key(key)}_{cache_key(text, ref, eng)}.wav"
            path = synth(text, ref_audio=ref, out_path=os.path.join(_AUDIO_DIR, fname),
                         engine=eng, character=char, force=force)
            rows.append({"character": char, "text": text, "audio_path": path,
                         "index": idx, "engine": last_engine(), "ref_audio": ref})
        audio_map[scene_id] = rows
    return audio_map


def status() -> dict:
    """给 check_env / MCP 工具用的一览。"""
    return {
        "engine_setting": ENGINE,
        "engine_resolved": resolve_engine(),
        "clone_available": _clone_available(),
        "f5_ckpt": _f5_ckpt(),
        "f5_vocab": _f5_vocab(),
        "vocos_dir": VOCOS_DIR,
        "ref_dir": REF_DIR,
        "voice_map": {k: v for k, v in VOICE_MAP.items()},
    }
