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

"""MCP Server（stdio）：把本地小剧场能力暴露成工具给 LobeChat / 其他 MCP 客户端。

工具（每个都是独立可调用的一步，便于"先生成剧本给我看，确认后再继续"）：
    theater_check_env()                         环境自检
    theater_parse_script(script_text)           剧本 → 结构化镜头（人读简报 + JSON）
    theater_generate_script(requirement)        本地 LLM 写剧本（可选）
    theater_generate_assets(script_text)        背景 + 角色透明立绘
    theater_generate_tts(script_text)           台词配音
    theater_compose_frames(script_text)         每镜头画面 + 字幕
    theater_export_video(script_text, out_name) 成片（NVENC/QSV→自动回退）
    theater_run_all(script_text, out_name)      一键全流程

接口设计说明：所有工具都以 script_text 为唯一"状态载体"，中间产物靠文件缓存
幂等复用。好处是 LobeChat 里分步调用时不用来回搬 JSON 路径；重跑时加 force=true。
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("local-theater-auto")


def _ok(payload) -> str:
    return json.dumps({"ok": True, "data": payload}, ensure_ascii=False, indent=2)


def _err(where: str, exc: Exception) -> str:
    return json.dumps(
        {
            "ok": False,
            "where": where,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc(limit=6),
        },
        ensure_ascii=False,
        indent=2,
    )


def _struct(script_text: str):
    from theater.script_parser import parse_full_script

    return parse_full_script(script_text)


@mcp.tool()
def theater_check_env() -> str:
    """检查本机环境（python/依赖/ffmpeg/模型/字体/语音），返回 JSON。"""
    try:
        import check_env as ce

        ce.results.clear()
        ce.check_python()
        ce.check_packages()
        ce.check_cache()
        ce.check_ffmpeg()
        ce.check_models()
        ce.check_fonts_and_tts()
        return _ok({"results": ce.results})
    except Exception as e:
        return _err("theater_check_env", e)


@mcp.tool()
def theater_parse_script(script_text: str) -> str:
    """解析剧本，返回镜头数、角色、每镜头站位与台词（含人读简报）。"""
    try:
        from theater.script_parser import format_scenes_brief

        struct = _struct(script_text)
        return _ok({"struct": struct, "brief": format_scenes_brief(struct)})
    except Exception as e:
        return _err("theater_parse_script", e)


@mcp.tool()
def theater_generate_script(requirement: str) -> str:
    """用本地 Qwen3-4B（OpenVINO/NPU）按需求生成剧本。"""
    try:
        from pipeline_full import generate_script

        return _ok({"script": generate_script(requirement)})
    except Exception as e:
        return _err("theater_generate_script", e)


@mcp.tool()
def theater_generate_assets(script_text: str, force: bool = False) -> str:
    """为剧本生成背景图与每个角色的透明立绘（已生成则复用缓存）。"""
    try:
        from theater import ai_draw

        struct = _struct(script_text)
        assets = ai_draw.generate_scene_assets(struct, force=force)
        return _ok(
            {
                "background": assets["background"],
                "backgrounds": {str(k): v for k, v in assets.get("backgrounds", {}).items()},
                "characters": assets["characters"],
                "model_dir": assets.get("model_dir", ""),
            }
        )
    except Exception as e:
        return _err("theater_generate_assets", e)


@mcp.tool()
def theater_generate_tts(script_text: str, force: bool = False) -> str:
    """把每句台词合成 wav 配音，返回每个镜头的音频路径。"""
    try:
        from theater import local_tts

        struct = _struct(script_text)
        tts = local_tts.gen_all_tts(struct, force=force)
        return _ok({str(k): v for k, v in tts.items()})
    except Exception as e:
        return _err("theater_generate_tts", e)


@mcp.tool()
def theater_compose_frames(script_text: str, force: bool = False) -> str:
    """合成每个镜头的画面（背景 + 角色按站位摆放 + 底部字幕）。"""
    try:
        from theater import ai_draw, frame_compose

        struct = _struct(script_text)
        assets = ai_draw.generate_scene_assets(struct, force=False)
        frames = frame_compose.build_frames(struct, assets, force=force)
        return _ok({str(k): v for k, v in frames.items()})
    except Exception as e:
        return _err("theater_compose_frames", e)


@mcp.tool()
def theater_export_video(script_text: str, out_name: str = "output.mp4", force: bool = False) -> str:
    """导出成片：自动补齐素材/配音/画面，再按镜头拼接编码。"""
    try:
        from pipeline_full import run_pipeline

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", out_name)
        res = run_pipeline(script_text, out_path, steps=["assets", "tts", "frames", "video"], force=force)
        return _ok(res)
    except Exception as e:
        return _err("theater_export_video", e)


@mcp.tool()
def theater_run_all(script_text: str, out_name: str = "output.mp4", force: bool = False) -> str:
    """一键跑完整流水线：解析 → 素材 → 配音 → 画面 → 成片。"""
    try:
        from pipeline_full import run_pipeline

        out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", out_name)
        return _ok(run_pipeline(script_text, out_path, force=force))
    except Exception as e:
        return _err("theater_run_all", e)


if __name__ == "__main__":
    mcp.run()
