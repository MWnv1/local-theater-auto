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

"""剧本结构化解析。

输入：图文小剧场剧本纯文本，形如

    【镜头1】
    【小林：站位：左】
    【阿杰：站位：右】
    小林：这个模型真能离线跑？
    阿杰：4bit 的，NPU 上 30 token/s。

输出：结构化镜头数据，供 ai_draw / local_tts / frame_compose / video_export 使用。

与原始稿的差异（均为修复项）：
1. 【镜头1】之前的文本不再丢弃，会自动归入第 1 个镜头；
2. `场景：机房` 这类元信息行不再被误判成台词；
3. 未显式写【站位】的角色会自动分配位置，避免全部堆在"居中"糊成一团；
4. 角色名单按首次出场顺序去重返回。
"""

import re
from typing import Dict, List

# 会被识别为"镜头元信息"而不是台词的关键字
META_KEYS = {
    "场景", "地点", "时间", "画面", "背景", "道具", "音效", "备注",
    "风格", "画风", "镜头类型", "摘要", "人物", "服装", "灯光",
}

_SCENE_RE = re.compile(r"^【\s*镜头\s*(\d+)\s*】\s*(.*)$")
_POS_RE = re.compile(r"^【\s*(.+?)\s*[：:]\s*站位\s*[：:]\s*(左|右|居中|中)\s*】\s*$")
# 【旁白】夜色渐深。  —— 方括号直接跟文本的台词/旁白写法
_BRACKET_LINE_RE = re.compile(r"^【([^【】]{1,12})】\s*(.+)$")
# 角色名限长，避免把长句里的冒号误判成台词
_DIALOG_RE = re.compile(r"^(.{1,20}?)[：:]\s*(.+)$")

_DEFAULT_POSITIONS = {
    1: ["居中"],
    2: ["左", "右"],
    3: ["左", "居中", "右"],
    4: ["左", "居中偏左", "居中偏右", "右"],
}

# 旁白/画外音：参与字幕与配音，但**不当作角色**（不画立绘、不占站位）
NARRATORS = {"旁白", "画外音", "旁白君", "解说", "Narrator", "narrator"}


def _new_scene(scene_id: int) -> Dict:
    return {
        "scene_id": int(scene_id),
        "scene_desc": "",
        "characters": [],
        "lines": [],
        "positions": {},
    }


def _add_line(scene: Dict, speaker: str, text: str) -> None:
    speaker = speaker.strip(" 　")
    text = text.strip()
    if not speaker or not text:
        return
    is_narrator = speaker in NARRATORS
    scene["lines"].append({"character": speaker, "text": text, "narrator": is_narrator})
    if not is_narrator and speaker not in scene["characters"]:
        scene["characters"].append(speaker)


def _assign_default_positions(scene: Dict) -> None:
    """给没有写站位的角色分配默认位置。"""
    chars = list(scene["characters"])
    missing = [c for c in chars if c not in scene["positions"]]
    if not missing:
        return
    n = len(chars)
    slots = _DEFAULT_POSITIONS.get(n)
    if slots is None:
        # 角色过多：在左->右之间均匀铺开
        slots = []
        for i in range(n):
            ratio = i / max(n - 1, 1)
            if ratio < 0.34:
                slots.append("左")
            elif ratio > 0.66:
                slots.append("右")
            else:
                slots.append("居中")
    for i, c in enumerate(chars):
        if c in scene["positions"]:
            continue
        scene["positions"][c] = slots[min(i, len(slots) - 1)]
        if scene["positions"][c] == "中":
            scene["positions"][c] = "居中"


def parse_full_script(script_text: str) -> Dict:
    """解析图文小剧场剧本，返回结构化镜头数据。"""
    if not script_text or not str(script_text).strip():
        raise ValueError("剧本内容为空，无法解析")

    raw_lines = str(script_text).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    lines: List[str] = []
    for l in raw_lines:
        s = l.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)

    scenes: List[Dict] = []
    current: Dict = None

    for line in lines:
        scene_match = _SCENE_RE.match(line)
        if scene_match:
            new_id = int(scene_match.group(1))
            if current is not None and current.get("_auto") and current["lines"]:
                # 第一个【镜头N】出现前已有正文：把它并进这个镜头，避免出现两个同号镜头
                current["scene_id"] = new_id
                current["_auto"] = False
            else:
                if current is not None:
                    scenes.append(current)
                current = _new_scene(new_id)
            tail = scene_match.group(2).strip()
            if tail:
                # 【镜头3】后面直接跟台词
                dm = _DIALOG_RE.match(tail)
                if dm and dm.group(1).strip() not in META_KEYS:
                    _add_line(current, dm.group(1), dm.group(2))
                else:
                    current["scene_desc"] = tail
            continue

        # 还没有任何镜头头：把前面这段文字归入镜头 1
        if current is None:
            current = _new_scene(1)
            current["_auto"] = True

        pos_match = _POS_RE.match(line)
        if pos_match:
            char_name = pos_match.group(1).strip()
            pos = pos_match.group(2).strip()
            if pos == "中":
                pos = "居中"
            current["positions"][char_name] = pos
            if char_name not in current["characters"]:
                current["characters"].append(char_name)
            continue

        # 【旁白】xxx / 【角色】xxx 这种方括号台词
        bracket_match = _BRACKET_LINE_RE.match(line)
        if bracket_match:
            name = bracket_match.group(1).strip()
            text = bracket_match.group(2).strip()
            if name in META_KEYS:
                if name in ("场景", "地点", "背景", "画面") and not current["scene_desc"]:
                    current["scene_desc"] = text
                continue
            _add_line(current, name, text)
            continue

        dialog_match = _DIALOG_RE.match(line)
        if dialog_match:
            speaker = dialog_match.group(1).strip(" 【】")
            text = dialog_match.group(2).strip()
            if speaker in META_KEYS:
                # 元信息：场景描述留作背景提示
                if speaker in ("场景", "地点", "背景", "画面"):
                    current["scene_desc"] = text
                continue
            if "【" in speaker or "】" in speaker:
                continue
            _add_line(current, speaker, text)
            continue

        # 纯描述行（既不是台词也不是站位）：当作场景描述
        if not current["scene_desc"]:
            current["scene_desc"] = line.strip("【】")

    if current is not None:
        scenes.append(current)

    # 收尾处理
    for s in scenes:
        s.pop("_auto", None)
        _assign_default_positions(s)
        s["subtitle"] = "　".join(
            f"{ln['character']}：{ln['text']}" for ln in s["lines"]
        )

    all_chars: List[str] = []
    for s in scenes:
        for c in s["characters"]:
            if c not in all_chars:
                all_chars.append(c)

    if not scenes:
        raise ValueError("未解析到任何镜头，请检查剧本格式（应包含【镜头1】）")

    return {
        "total_scenes": len(scenes),
        "all_characters": all_chars,
        "scenes": scenes,
    }


def format_scenes_brief(struct: Dict) -> str:
    """把解析结果渲染成便于人看/模型读的简报。"""
    out = [f"共 {struct['total_scenes']} 个镜头，角色：{'、'.join(struct['all_characters'])}"]
    for s in struct["scenes"]:
        out.append(f"\n【镜头{s['scene_id']}】" + (f"（{s['scene_desc']}）" if s.get("scene_desc") else ""))
        for c, p in s["positions"].items():
            out.append(f"  · {c} → {p}")
        for ln in s["lines"]:
            out.append(f"  {ln['character']}：{ln['text']}")
    return "\n".join(out)
