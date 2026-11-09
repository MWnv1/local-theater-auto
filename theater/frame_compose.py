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

"""图层合成 + 字幕（PIL）。

修复的点：
1. 原稿在 RGB 图上用 `fill=(0,0,0,180)` 画半透明黑条 —— PIL 的 RGB 图不接受
   4 元组颜色，会抛 TypeError / alpha 被丢弃。这里改为 RGBA 覆盖层
   + `alpha_composite`，真正做出半透明底。
2. 字体改为**全路径候选列表**（msyh.ttc / msyhbd.ttc / simhei.ttf），
   并去掉裸 `except:`，找不到时给出可见的告警而不是静默退化成位图字体。
3. 字幕不再只取第一句：一个镜头的所有台词按像素宽度自动换行，最多 3 行。
4. 多角色按站位槽位分布（左 / 居中偏左 / 居中 / 居中偏右 / 右），
   并对重叠做了横向避让，不再是全部堆在正中。
5. 画面按"背景 + 角色 + 文案"签名缓存，重复调用不重算。
"""

import os
import hashlib
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont

from . import subdir

_FRAME_DIR = subdir("frames")
VIDEO_W = int(os.environ.get("VIDEO_W", "1920"))
VIDEO_H = int(os.environ.get("VIDEO_H", "1080"))

FONT_CANDIDATES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
]
_FONT_CACHE: Dict[int, ImageFont.FreeTypeFont] = {}

# 站位 → 横向锚点（占画面宽度比例）
_POS_ANCHOR = {
    "左": 0.20,
    "居中偏左": 0.36,
    "居中": 0.50,
    "中": 0.50,
    "居中偏右": 0.64,
    "右": 0.80,
}

_SUBTITLE_FONT_SIZE = int(os.environ.get("SUBTITLE_FONT_SIZE", "44"))
_SUBTITLE_MAX_LINES = 3
_BAND_ALPHA = 165


def _load_font(size: int = _SUBTITLE_FONT_SIZE) -> ImageFont.FreeTypeFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except Exception:
                continue
    print("[warn] 未找到中文字体，字幕将使用默认位图字体（可能显示为方块）")
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> List[str]:
    """按像素宽度折行（中文按字切，英文按词优先）。"""
    if not text:
        return []
    lines: List[str] = []
    cur = ""
    for ch in text:
        trial = cur + ch
        if draw.textlength(trial, font=font) <= max_width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = ch
            if len(lines) >= _SUBTITLE_MAX_LINES:
                break
    if cur and len(lines) < _SUBTITLE_MAX_LINES:
        lines.append(cur)
    if len(lines) == _SUBTITLE_MAX_LINES:
        # 末行加省略号，避免无限截断后看不出被截
        last = lines[-1]
        while last and draw.textlength(last + "…", font=font) > max_width:
            last = last[:-1]
        lines[-1] = last + "…"
    return lines


def _paste_character(canvas: Image.Image, char_img: Image.Image, anchor: float, char_top: int) -> int:
    """把角色贴到画布，返回其左边 x（用于重叠避让）。"""
    target_h = int(VIDEO_H * 0.72)
    w, h = char_img.size
    ratio = target_h / float(h)
    new_w = max(1, int(w * ratio))
    char_img = char_img.resize((new_w, target_h), Image.Resampling.LANCZOS)

    x = int(VIDEO_W * anchor - new_w / 2)
    x = max(20, min(x, VIDEO_W - new_w - 20))
    canvas.alpha_composite(char_img, (x, char_top))
    return x


def _subtitle_layout(subtitle: str):
    """算出字幕行数/行高/字幕条高度（角色区据此让位）。"""
    font = _load_font()
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    lines = _wrap_text(probe, subtitle, font, VIDEO_W - 220)
    line_h = _SUBTITLE_FONT_SIZE + 12
    band_h = (len(lines) * line_h + 44) if lines else 0
    return font, lines, line_h, band_h


def _char_top_for(band_h: int) -> int:
    char_h = int(VIDEO_H * 0.72)
    return max(10, min(VIDEO_H - band_h - char_h - 10, VIDEO_H - band_h - char_h))


def _scale_actor(img: Image.Image) -> Image.Image:
    """角色统一缩到画面高度的 72%，保持等比。"""
    target_h = int(VIDEO_H * 0.72)
    new_w = max(1, int(img.width * (target_h / float(img.height))))
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


def _blit_grouped(canvas: Image.Image, items: list, char_top: int) -> None:
    """按站位分组贴角色：同组（同锚点）内均匀铺开，整组超宽时整体回收到画面内。"""
    groups: Dict[float, list] = {}
    for item in items:
        groups.setdefault(item[2], []).append(item)

    for anchor, group in groups.items():
        n = len(group)
        widths = [it[1].width for it in group]
        avg_w = sum(widths) / n
        base_center = VIDEO_W * anchor
        if n == 1:
            centers = [base_center]
        else:
            spacing = min(avg_w * 0.9, (VIDEO_W - 80) / n)
            centers = [base_center + (i - (n - 1) / 2.0) * spacing for i in range(n)]

        left_edge = centers[0] - widths[0] / 2
        right_edge = centers[-1] + widths[-1] / 2
        shift = 0
        if left_edge < 20:
            shift = 20 - left_edge
        if right_edge + shift > VIDEO_W - 20:
            shift = max(0, VIDEO_W - 20 - right_edge)
        centers = [c + shift for c in centers]

        for (name, img, _), center in zip(group, centers):
            x = int(center - img.width / 2)
            x = max(20, min(x, VIDEO_W - img.width - 20))
            canvas.alpha_composite(img, (x, char_top))


def _draw_subtitle(canvas: Image.Image, subtitle: str, font, lines, line_h: int, band_h: int) -> None:
    if not lines:
        return
    band = Image.new("RGBA", (VIDEO_W, band_h), (0, 0, 0, _BAND_ALPHA))
    canvas.alpha_composite(band, (0, VIDEO_H - band_h))
    draw = ImageDraw.Draw(canvas)
    y = VIDEO_H - band_h + 16
    for line in lines:
        tw = draw.textlength(line, font=font)
        draw.text(((VIDEO_W - tw) / 2, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h


def _load_bg(bg_path: str) -> Image.Image:
    bg = Image.open(bg_path).convert("RGBA")
    return bg.resize((VIDEO_W, VIDEO_H), Image.Resampling.LANCZOS)


def compose_scene_frame(
    bg_path: str,
    char_paths: Dict[str, str],
    positions: Dict[str, str],
    subtitle: str,
    scene_id: int,
    force: bool = False,
) -> str:
    """合成单镜头画面：背景 + 角色 + 底部半透明字幕条。"""
    sig_src = "|".join(
        [bg_path, str(scene_id), subtitle]
        + [f"{k}:{v}:{os.path.getsize(v) if os.path.exists(v) else 0}" for k, v in char_paths.items()]
        + [f"{k}={v}" for k, v in positions.items()]
    )
    sig = hashlib.md5(sig_src.encode("utf-8")).hexdigest()[:12]
    out_path = os.path.join(_FRAME_DIR, f"frame_s{scene_id}_{sig}.png")
    if not force and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return os.path.abspath(out_path)

    canvas = _load_bg(bg_path)
    font, sub_lines, line_h, band_h = _subtitle_layout(subtitle)
    char_top = _char_top_for(band_h)

    prepared = []  # (name, image, anchor)
    for char_name, char_path in char_paths.items():
        if not os.path.exists(char_path):
            continue
        anchor = _POS_ANCHOR.get(positions.get(char_name, "居中"), 0.5)
        prepared.append((char_name, _scale_actor(Image.open(char_path).convert("RGBA")), anchor))

    _blit_grouped(canvas, prepared, char_top)
    _draw_subtitle(canvas, subtitle, font, sub_lines, line_h, band_h)

    canvas.convert("RGB").save(out_path, "PNG")
    return os.path.abspath(out_path)


def compose_actor_frame(
    bg_path: str,
    actor_images: Dict[str, Image.Image],
    positions: Dict[str, str],
    subtitle: str,
    out_path: str,
) -> str:
    """用"某一时刻的角色图"合成整帧（角色可以带动作），写盘并返回路径。

    这是动画成片的单帧入口：调用方（pipeline 的 anim 步骤）对每个时间点先用
    `animator.render_frame` 拿到该角色当下的立绘，再交给这里排版 + 叠加字幕。
    """
    canvas = _load_bg(bg_path)
    font, sub_lines, line_h, band_h = _subtitle_layout(subtitle)
    char_top = _char_top_for(band_h)

    prepared = []
    for char_name, img in actor_images.items():
        if img is None:
            continue
        anchor = _POS_ANCHOR.get(positions.get(char_name, "居中"), 0.5)
        prepared.append((char_name, _scale_actor(img.convert("RGBA")), anchor))

    _blit_grouped(canvas, prepared, char_top)
    _draw_subtitle(canvas, subtitle, font, sub_lines, line_h, band_h)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.convert("RGB").save(out_path, "PNG")
    return os.path.abspath(out_path)


def build_frames(script_struct: dict, assets: dict, force: bool = False) -> Dict[int, str]:
    """批量生成所有镜头画面。

    背景优先用该镜头的专属背景（剧本写了【场景：xxx】时），否则用通用背景。
    """
    frame_map: Dict[int, str] = {}
    bg_default = assets.get("background", "")
    bg_map = assets.get("backgrounds", {}) or {}
    char_paths = assets.get("characters", {}) or {}

    for scene in script_struct.get("scenes", []):
        scene_id = int(scene["scene_id"])
        bg_path = bg_map.get(scene_id) or bg_default
        if not bg_path or not os.path.exists(bg_path):
            raise RuntimeError(f"镜头 {scene_id} 缺少背景图，请先生成素材")

        subtitle = scene.get("subtitle") or "　".join(
            f"{ln['character']}：{ln['text']}" for ln in scene.get("lines", [])
        )
        frame_map[scene_id] = compose_scene_frame(
            bg_path=bg_path,
            char_paths=char_paths,
            positions=scene.get("positions", {}),
            subtitle=subtitle,
            scene_id=scene_id,
            force=force,
        )
    return frame_map
