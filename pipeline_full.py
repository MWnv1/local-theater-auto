# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
# Generated: 2026-11-07 09:31:28 +08:00 (Asia/Shanghai) [UTC 2026-11-07T01:31:28Z]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

STEPS = ("parse", "assets", "tts", "frames", "video")

LLM_MODEL_DIR = os.environ.get("LLM_MODEL_DIR", os.path.join(MODEL_ROOT, "qwen3-4b-int4-ov"))
LLM_DEVICE = os.environ.get("LLM_DEVICE", "NPU")

_llm = None


def generate_script(requirement: str, max_new_tokens: int = 900) -> str:
    """用本地 Qwen3-4B（OpenVINO, 可跑 NPU）生成剧本。模型不存在时给出清晰报错。"""
    global _llm
    if not os.path.isdir(LLM_MODEL_DIR) or not os.listdir(LLM_MODEL_DIR):
        raise RuntimeError(
            f"剧本生成模型目录不存在或为空：{LLM_MODEL_DIR}\n"
            "请先把 Qwen3-4B INT4 OpenVINO 模型包解压到该目录，或直接手工提供剧本文本。"
        )

    from optimum.intel.openvino import OVModelForCausalLM  # 延迟导入
    from transformers import AutoTokenizer

    if _llm is None:
        _llm = OVModelForCausalLM.from_pretrained(LLM_MODEL_DIR, device=LLM_DEVICE, compile=True)
    tok = AutoTokenizer.from_pretrained(LLM_MODEL_DIR)

    sys_prompt = (
        "你是图文小剧场编剧。请按固定格式输出中文剧本，不要输出解释：\n"
        "【镜头1】\n【角色A：站位：左】\n【角色B：站位：右】\n角色A：台词\n角色B：台词\n"
        "每个镜头 2-4 句台词，共 6 个镜头，镜头之间空行。"
    )
    text = f"{sys_prompt}\n需求：{requirement}\n剧本："
    inputs = tok(text, return_tensors="pt")
    out = _llm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                        temperature=0.8, top_p=0.9)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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

    # 4) 画面
    if "frames" in steps:
        frames = frame_compose.build_frames(struct, assets, force=force)
        _dump(os.path.join(workdir, f"frames_{stamp}.json"),
              {str(k): v for k, v in frames.items()})
        summary["frames"] = {str(k): v for k, v in frames.items()}
    else:
        frames = {}

    # 5) 成片
    if "video" in steps:
        out_path, duration = video_export.export_final_video(struct, frames, tts, out_path)
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
