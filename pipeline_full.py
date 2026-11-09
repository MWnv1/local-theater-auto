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

"""一站式流水线：剧本 → 素材 → 配音 → 画面 → 成片。

既可以命令行单独跑：
    python pipeline_full.py --script script.txt --out output/demo.mp4
    python pipeline_full.py --requirement "两个程序员在机房聊本地AI小剧场" --out output/demo.mp4

也可以被 server.py（MCP）复用其中的函数做分步执行。
"""

import argparse
import json
import os
import time
from typing import Dict, List

from theater import CACHE_ROOT, MODEL_ROOT, ensure_dirs, subdir
from theater.script_parser import parse_full_script, format_scenes_brief

STEPS = ("parse", "assets", "tts", "sfx", "anim", "frames", "video")

# 剧本模型：torch 权重（Qwen3-4B 原始 safetensors）与 OpenVINO INT4 都能跑，
# 本机没有 Intel NPU，所以 NPU 只是候选，不可用就自动退 GPU/CPU。
LLM_BACKEND = os.environ.get("LLM_BACKEND", "auto")            # auto | torch | openvino
LLM_TORCH_DIR = os.environ.get("LLM_TORCH_DIR", os.path.join(MODEL_ROOT, "qwen3-4b"))
LLM_MODEL_DIR = os.environ.get("LLM_OV_DIR", os.path.join(MODEL_ROOT, "qwen3-4b-int4-ov"))
LLM_DEVICE = os.environ.get("LLM_DEVICE", "auto")               # auto | NPU | GPU | CPU | cuda

_llm = None
_tok = None
_llm_backend = ""

SYS_PROMPT = (
    "你是图文小剧场编剧。请按固定格式输出中文剧本，不要输出解释：\n"
    "【镜头1】\n【角色A：站位：左】\n【角色B：站位：右】\n角色A：台词\n角色B：台词\n"
    "每个镜头 2-4 句台词，共 6 个镜头，镜头之间空行。"
)


def _ov_device() -> str:
    """OpenVINO 设备：auto 时按 NPU → GPU → CPU 顺序挑第一个真实存在的。"""
    if LLM_DEVICE and LLM_DEVICE.lower() != "auto":
        return LLM_DEVICE
    try:
        import openvino as ov

        devices = set(ov.Core().available_devices)
    except Exception:
        return "CPU"
    for cand in ("NPU", "GPU", "CPU"):
        if any(d == cand or d.startswith(cand + ".") for d in devices):
            return cand
    return "CPU"


def _resolve_llm_backend() -> str:
    if LLM_BACKEND in ("torch", "openvino"):
        return LLM_BACKEND
    if os.path.isdir(LLM_TORCH_DIR) and os.listdir(LLM_TORCH_DIR):
        return "torch"
    if os.path.isdir(LLM_MODEL_DIR) and os.listdir(LLM_MODEL_DIR):
        return "openvino"
    raise RuntimeError(
        "没有可用的剧本生成模型。请下载 Qwen3-4B 到 "
        f"{LLM_TORCH_DIR}（torch 格式），或把 OpenVINO INT4 包放到 {LLM_MODEL_DIR}。"
    )


def _load_llm():
    """加载剧本模型（一次常驻），返回 (backend, model, tokenizer)。"""
    global _llm, _tok, _llm_backend
    backend = _resolve_llm_backend()
    if _llm is not None and _llm_backend == backend:
        return backend, _llm, _tok

    if backend == "torch":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        _tok = AutoTokenizer.from_pretrained(LLM_TORCH_DIR)
        _llm = AutoModelForCausalLM.from_pretrained(
            LLM_TORCH_DIR,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True,
        )
        _llm.eval()
    else:
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer

        dev = _ov_device()
        last_err = None
        for cand in (dev, "GPU", "CPU"):
            try:
                _llm = OVModelForCausalLM.from_pretrained(LLM_MODEL_DIR, device=cand, compile=True)
                dev = cand
                break
            except Exception as e:  # 比如本机根本没有 NPU
                last_err = e
                _llm = None
        if _llm is None:
            raise RuntimeError(f"OpenVINO 剧本模型加载失败（试过 {dev}/GPU/CPU）：{last_err}")
        print(f"[llm] OpenVINO 设备：{dev}")
        _tok = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)

    _llm_backend = backend
    return backend, _llm, _tok


def generate_script(requirement: str, max_new_tokens: int = 900) -> str:
    """用本地 Qwen3-4B 生成剧本（torch 或 OpenVINO 后端，全部离线）。"""
    backend, model, tok = _load_llm()
    user = f"需求：{requirement}"

    if backend == "torch":
        import torch

        msgs = [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": user}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.8, top_p=0.9, repetition_penalty=1.05)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return tok.decode(gen, skip_special_tokens=True).strip()

    inputs = tok(f"{SYS_PROMPT}\n{user}\n剧本：", return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                         temperature=0.8, top_p=0.9)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# 台词/场景描述里的关键词 → 内置音效名。命中不了就让素材库检索决定。
SFX_RULES = (
    (("雨", "下雨", "雨声"), "rain"),
    (("风", "呼啸", "转场"), "whoosh"),
    (("门", "开门", "关门", "走廊"), "door"),
    (("键盘", "打字", "代码", "敲"), "typing"),
    (("心跳", "紧张", "害怕"), "heartbeat"),
    (("消息", "通知", "电话", "提示"), "incoming"),
    (("撞", "砸", "摔", "震", "雷"), "thud"),
    (("钟", "铃", "响", "闹"), "chime"),
    (("点", "按", "键"), "click"),
)


def _sfx_name_for(text: str) -> str:
    for keys, name in SFX_RULES:
        if any(k in text for k in keys):
            return name
    return ""


def build_sfx(script_struct: dict, force: bool = False) -> Dict[int, str]:
    """给每个镜头挑一条音效（内置 9 种离线合成音效，先入库再取用）。"""
    from theater import library

    library.ensure_builtin_sfx()
    out: Dict[int, str] = {}
    for scene in script_struct.get("scenes", []):
        sid = int(scene["scene_id"])
        blob = " ".join((ln.get("text") or "") for ln in scene.get("lines", []))
        blob += " " + (scene.get("scene_desc") or "")
        name = _sfx_name_for(blob)
        if not name:
            hits = library.search(blob, kind="sfx", k=1)
            name = hits[0]["name"] if hits else ""
        out[sid] = library.synth_sfx(name) if name else ""
    return out


def _scene_subtitle(scene: dict) -> str:
    return scene.get("subtitle") or "　".join(
        f"{ln['character']}：{ln['text']}" for ln in scene.get("lines", [])
    )


def build_anims(script_struct: dict, assets: dict, tts: Dict[int, list],
                force: bool = False) -> Dict[int, dict]:
    """给每个镜头渲染"角色真的在动"的 PNG 序列（骨骼绑定 + 动作函数）。

    动作分配：本镜头有台词的角色说 `talk`，其余角色交替 `idle` / `nod`。
    时长以该镜头配音总长为准，保证动作和声音对得上。
    """
    from theater import animator, frame_compose, rigging, video_export

    fps = int(os.environ.get("ANIM_FPS", "24"))
    char_paths = (assets or {}).get("characters") or {}
    bg_map = (assets or {}).get("backgrounds") or {}
    bg_default = (assets or {}).get("background", "")
    rigs = {name: rigging.rig_for(p) for name, p in char_paths.items() if os.path.exists(p)}
    if not rigs:
        print("[anim] 没有可用立绘，跳过动画步骤")
        return {}

    out: Dict[int, dict] = {}
    for scene in script_struct.get("scenes", []):
        sid = int(scene["scene_id"])
        bg = bg_map.get(sid) or bg_default
        if not bg or not os.path.exists(bg):
            print(f"[anim] 镜头 {sid} 缺背景，跳过")
            continue

        audios = tts.get(sid, []) or []
        dur = 0.5 + sum(video_export._get_audio_duration(a["audio_path"]) + 0.15 for a in audios)
        dur = max(1.0, dur)
        speakers = {a.get("character") for a in audios}
        positions = scene.get("positions", {}) or {}
        subtitle = _scene_subtitle(scene)

        out_dir = os.path.join(subdir("anim"), f"scene_{sid}")
        os.makedirs(out_dir, exist_ok=True)
        frames = max(1, int(round(dur * fps)))
        files: List[str] = []
        for i in range(frames):
            t = i / float(fps)
            actors = {}
            for name, rig in rigs.items():
                if name in speakers:
                    act = "talk"
                else:
                    act = "nod" if (i // max(1, fps)) % 2 == 0 else "idle"
                actors[name] = animator.render_frame(rig, act, t)
            p = os.path.join(out_dir, f"f_{i:04d}.png")
            if force or not os.path.exists(p) or os.path.getsize(p) == 0:
                frame_compose.compose_actor_frame(bg, actors, positions, subtitle, p)
            files.append(p)

        out[sid] = {"files": files, "fps": fps, "dir": os.path.abspath(out_dir),
                    "duration": round(dur, 3), "manifest": os.path.join(out_dir, "anim_manifest.json")}
        _dump(out[sid]["manifest"], {"scene_id": sid, "fps": fps, "frames": frames,
                                     "duration": out[sid]["duration"], "files": files,
                                     "rigs": {k: v.get("signature") for k, v in rigs.items()}})
    return out


def run_pipeline(script_text: str, out_path: str, steps: List[str] = None,
                 force: bool = False, workdir: str = None) -> Dict:
    """执行流水线，返回各阶段结果摘要（路径均为绝对路径）。"""
    from theater import ai_draw, local_tts, frame_compose, video_export

    steps = steps or list(STEPS)
    unknown = [s for s in steps if s not in STEPS]
    if unknown:
        raise ValueError(f"未知步骤：{unknown}，可选 {STEPS}")

    ensure_dirs()
    workdir = workdir or subdir("work")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary: Dict = {"steps": steps, "workdir": workdir}

    # 1) 解析
    struct = parse_full_script(script_text)
    _dump(os.path.join(workdir, f"script_{stamp}.json"), struct)
    summary["parse"] = {
        "total_scenes": struct["total_scenes"],
        "characters": struct["all_characters"],
        "brief": format_scenes_brief(struct),
    }
    if "parse" in steps and len(steps) == 1:
        return summary

    # 2) 素材
    if "assets" in steps:
        assets = ai_draw.generate_scene_assets(struct, force=force)
        _dump(os.path.join(workdir, f"assets_{stamp}.json"), assets)
        summary["assets"] = {
            "background": assets["background"],
            "backgrounds": {str(k): v for k, v in assets.get("backgrounds", {}).items()},
            "characters": assets["characters"],
            "model_dir": assets.get("model_dir", ""),
        }
    else:
        assets = {}

    # 3) 配音
    if "tts" in steps:
        tts = local_tts.gen_all_tts(struct, force=force)
        _dump(os.path.join(workdir, f"tts_{stamp}.json"),
              {str(k): v for k, v in tts.items()})
        summary["tts"] = {str(k): [a["audio_path"] for a in v] for k, v in tts.items()}
    else:
        tts = {}

    # 4) 音效（垫在台词下面，不抢人声）
    if "sfx" in steps:
        sfx = build_sfx(struct, force=force)
        _dump(os.path.join(workdir, f"sfx_{stamp}.json"), {str(k): v for k, v in sfx.items()})
        summary["sfx"] = {str(k): v for k, v in sfx.items() if v}
    else:
        sfx = {}

    # 5) 骨骼动画（角色真的在动，不再是静帧）
    if "anim" in steps:
        anims = build_anims(struct, assets, tts, force=force)
        _dump(os.path.join(workdir, f"anim_{stamp}.json"),
              {str(k): {"dir": v["dir"], "fps": v["fps"], "frames": len(v["files"])}
               for k, v in anims.items()})
        summary["anim"] = {str(k): {"frames": len(v["files"]), "fps": v["fps"], "dir": v["dir"]}
                           for k, v in anims.items()}
    else:
        anims = {}

    # 6) 静帧画面（没有动画时作为兜底画面，也是镜头缩略图）
    if "frames" in steps:
        frames = frame_compose.build_frames(struct, assets, force=force)
        _dump(os.path.join(workdir, f"frames_{stamp}.json"),
              {str(k): v for k, v in frames.items()})
        summary["frames"] = {str(k): v for k, v in frames.items()}
    else:
        frames = {}

    # 7) 成片
    if "video" in steps:
        if not frames:
            # 只跑 video 时也得有画面：拿动画的第一帧顶上
            frames = {sid: a["files"][0] for sid, a in anims.items() if a.get("files")}
        out_path, duration = video_export.export_final_video(
            struct, frames, tts, out_path, anims=anims, sfx_map=sfx)
        summary["video"] = {"path": out_path, "duration": duration}

    summary["ok"] = True
    return summary


def main():
    ap = argparse.ArgumentParser(description="本地图文小剧场 一键流水线")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--script", help="剧本文本文件路径（UTF-8）")
    src.add_argument("--requirement", help="一句需求，交给本地 LLM 生成剧本")
    ap.add_argument("--out", default=os.path.join(CACHE_ROOT, "output.mp4"), help="成片输出路径")
    ap.add_argument("--steps", default=",".join(STEPS), help="只跑部分步骤，如 parse,assets")
    ap.add_argument("--force", action="store_true", help="忽略缓存，全部重算")
    args = ap.parse_args()

    if args.script:
        with open(args.script, "r", encoding="utf-8") as f:
            script_text = f.read()
    else:
        print("[1/1] 调用本地模型生成剧本…")
        script_text = generate_script(args.requirement)
        print(script_text)

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    print(f"[run] steps={steps}")
    res = run_pipeline(script_text, args.out, steps=steps, force=args.force)
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
