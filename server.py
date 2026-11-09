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
    theater_generate_script(requirement)        本地 Qwen3-4B 写剧本（torch / OpenVINO）
    theater_generate_assets(script_text)        背景 + 角色透明立绘（先查素材库，再自绘）
    theater_generate_tts(script_text)           台词配音（F5-TTS 克隆，缺参考音退 SAPI）
    theater_compose_frames(script_text)         每镜头画面 + 字幕
    theater_export_video(script_text, out_name) 成片（NVENC/QSV/AMF→自动回退 x264）
    theater_run_all(script_text, out_name)      一键全流程
    theater_clone_voice(text, ref_audio)        参考语音 → 同音色合成新台词
    theater_animate_character(char_image, ...)  立绘骨骼绑定 + 动作序列渲染
    theater_generate_similar_video(video, ...)  参考视频 → 相似视频（逐帧 img2img）
    theater_library_search(query)               检索素材库（图片 / 音效）
    theater_make_sfx(name)                      合成内置音效并入库存档

接口设计说明：所有工具都以 script_text 为唯一"状态载体"，中间产物靠文件缓存
幂等复用。好处是 LobeChat 里分步调用时不用来回搬 JSON 路径；重跑时加 force=true。
"""

import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# mcp 2.x 把 FastMCP 改名成 MCPServer（mcp.server.mcpserver），1.x 还是 FastMCP。
# 两代都支持 @server.tool() 和 run()，所以按可用的那个来，别把用户钉在某一版。
try:
    from mcp.server.fastmcp import FastMCP as _Server  # mcp < 2
except Exception:  # pragma: no cover - 取决于本机装的是哪一代
    from mcp.server.mcpserver import MCPServer as _Server  # mcp >= 2

mcp = _Server("local-theater-auto")


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


def _out_path(out_name: str, default_ext: str = "") -> str:
    """成片/音频输出目录：项目下 output/，允许传绝对路径。

    `default_ext` 不是空的，而 out_name 又没带扩展名时补上。pydub/ffmpeg 都靠
    扩展名判格式，`clone_voice(out_name="probe")` 这种写法会直接报
    "No format specified"，所以在入参侧就把后缀补齐，别让它跑到下游才炸。
    """
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(d, exist_ok=True)
    p = out_name if os.path.isabs(out_name) else os.path.join(d, out_name)
    if default_ext and not os.path.splitext(p)[1]:
        p += default_ext
    return p


@mcp.tool()
def theater_export_video(script_text: str, out_name: str = "output.mp4", force: bool = False,
                         animate: bool = True) -> str:
    """导出成片：自动补齐素材/配音/音效/动画/画面，再按镜头拼接编码。"""
    try:
        from pipeline_full import run_pipeline

        steps = ["assets", "tts", "sfx", "frames", "video"]
        if animate:
            steps.insert(3, "anim")
        return _ok(run_pipeline(script_text, _out_path(out_name, ".mp4"), steps=steps, force=force))
    except Exception as e:
        return _err("theater_export_video", e)


@mcp.tool()
def theater_run_all(script_text: str, out_name: str = "output.mp4", force: bool = False) -> str:
    """一键跑完整流水线：解析 → 素材 → 配音 → 音效 → 动画 → 画面 → 成片。"""
    try:
        from pipeline_full import run_pipeline

        return _ok(run_pipeline(script_text, _out_path(out_name, ".mp4"), force=force))
    except Exception as e:
        return _err("theater_run_all", e)


@mcp.tool()
def theater_clone_voice(text: str, ref_audio: str, out_name: str = "clone.wav",
                        ref_text: str = "", engine: str = "clone") -> str:
    """给定一段参考语音，用同音色合成新台词（F5-TTS zero-shot，全本地离线）。

    ref_audio 旁放同名 .txt 写参考文字最稳（没写会自动尝试转写并缓存）。
    """
    try:
        from theater import tts_clone

        out = tts_clone.synth(text, ref_audio=ref_audio, out_path=_out_path(out_name, ".wav"),
                              engine=engine, ref_text=ref_text, force=True)
        return _ok({"audio": out, "engine": tts_clone.last_engine(),
                    "bytes": os.path.getsize(out), "status": tts_clone.status()})
    except Exception as e:
        return _err("theater_clone_voice", e)


@mcp.tool()
def theater_animate_character(char_image: str, action: str = "wave", seconds: float = 2.0,
                              fps: int = 24) -> str:
    """给单张角色立绘做 2D 伪骨骼绑定，并渲染一段动作 PNG 序列。

    action 可选 idle / nod / wave / walk / talk。
    """
    try:
        from theater import animator, rigging

        if action not in animator.ACTIONS:
            return _ok({"warning": f"未知动作 {action}，已按 idle 处理",
                        "actions": sorted(animator.ACTIONS)})
        rig = rigging.rig_for(char_image)
        clip = animator.render_clip(rig, action, float(seconds), int(fps))
        return _ok({"rig": {"signature": rig.get("signature"), "parts": len(rig.get("parts", []))},
                    "action": action, "clip": clip})
    except Exception as e:
        return _err("theater_animate_character", e)


@mcp.tool()
def theater_generate_similar_video(video_path: str, prompt: str = "", out_name: str = "similar.mp4",
                                   max_frames: int = 0, keep_audio: bool = True) -> str:
    """按参考视频生成"相似"视频：逐帧 img2img 保留构图与运动，并给出相似度报告。"""
    try:
        from theater import video_ref

        mf = int(max_frames) if max_frames else None
        out, info = video_ref.generate_similar_video(
            video_path, prompt, _out_path(out_name, ".mp4"), keep_audio=keep_audio, max_frames=mf)
        report = video_ref.similarity_report(video_path, out)
        return _ok({"video": out, "info": info, "report": report})
    except Exception as e:
        return _err("theater_generate_similar_video", e)


@mcp.tool()
def theater_library_search(query: str, kind: str = "", k: int = 5) -> str:
    """检索素材库（图片/音效）：AI 出图出音前先问库，命中就直接复用。"""
    try:
        from theater import library

        hits = library.search(query, kind=(kind or None), k=int(k))
        return _ok({"query": query, "hits": hits, "stats": library.stats()})
    except Exception as e:
        return _err("theater_library_search", e)


@mcp.tool()
def theater_make_sfx(name: str = "rain") -> str:
    """合成一条内置音效（rain/whoosh/click/typing/door/thud/incoming/heartbeat/chime）并入库存档。"""
    try:
        from theater import library

        path = library.synth_sfx(name)
        return _ok({"sfx": path, "name": name, "bytes": os.path.getsize(path),
                    "stats": library.stats()})
    except Exception as e:
        return _err("theater_make_sfx", e)


if __name__ == "__main__":
    mcp.run()
