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

"""素材库 + 音效库。

为什么需要它：纯自绘的管线每次都要跑一遍大模型（512px 一张 20 分钟），
而小剧场里大量镜头用的是"同一间屋子、同一种雨声"，重复自绘纯属浪费。
这里做一个本地资产库：

  * 图片 / 音效 / 音乐 / 视频 / 参考语音分目录存放，`index.json` 记元数据；
  * `add()` 用 sha256 去重，重复文件不会在库里长两份；
  * `search()` / `pick_for_prompt()` 支持中文查询 + 同义词扩展，命中就复用；
  * `synth_sfx()` 用 numpy 现场合成一批常用音效（离线、无版权风险），
    所以"刚装好的库"里就已经有东西可调，不是空壳；
  * `mix_sfx()` 用 ffmpeg 把音效叠到台词轨上，时长以台词为准。

环境变量：
  THEATER_LIBRARY      库根目录（默认 <项目根>/library）
  LIBRARY_MIN_SCORE    检索命中阈值（默认 0.45）
  LIBRARY_FIRST        出图/选素材时是否优先查库（默认 1）
"""

import hashlib
import json
import os
import shutil
import time
import wave
from typing import Dict, List, Optional

import numpy as np

from . import PROJECT_ROOT, safe_name, ffmpeg_path

LIB_ROOT = os.environ.get("THEATER_LIBRARY", os.path.join(PROJECT_ROOT, "library"))
INDEX_PATH = os.path.join(LIB_ROOT, "index.json")
LIBRARY_MIN_SCORE = float(os.environ.get("LIBRARY_MIN_SCORE", "0.45"))
LIBRARY_FIRST = os.environ.get("LIBRARY_FIRST", "1") not in ("0", "false", "False")

KINDS = ("image", "sfx", "music", "video", "voice_ref")
SFX_RATE = 24000

# 中文查询词 → 库内候选词（英文名/别名）。只做"够用"的扩展，不是完整词典。
SYNONYMS: Dict[str, List[str]] = {
    "雨声": ["rain", "raindrop", "drizzle"],
    "下雨": ["rain", "raindrop"],
    "雨": ["rain"],
    "敲门": ["door", "knock"],
    "门": ["door"],
    "点击": ["click", "tap"],
    "按键": ["click", "keyboard", "typing"],
    "打字": ["typing", "keyboard"],
    "键盘": ["typing", "keyboard"],
    "风声": ["wind", "whoosh"],
    "呼啸": ["whoosh", "wind"],
    "转场": ["whoosh", "transition"],
    "撞击": ["thud", "impact"],
    "心跳": ["heartbeat"],
    "提示音": ["incoming", "notify", "chime"],
    "消息": ["incoming", "notify"],
    "房间": ["room", "indoor"],
    "室内": ["indoor", "room"],
    "机房": ["server", "room", "datacenter"],
    "办公室": ["office", "room"],
    "街道": ["street", "city"],
    "夜景": ["night", "city"],
    "森林": ["forest", "tree"],
    "rain": ["雨声", "下雨"],
    "door": ["敲门", "门"],
    "click": ["点击", "按键"],
    "typing": ["打字", "键盘"],
    "whoosh": ["风声", "转场"],
    "room": ["房间", "室内"],
    "indoor": ["室内", "房间"],
    "office": ["办公室"],
    "night": ["夜景"],
}

_INDEX_CACHE: Optional[dict] = None


# ---------------------------------------------------------------------------
# 索引
# ---------------------------------------------------------------------------
def _index_file() -> str:
    return INDEX_PATH


def _load_index() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    path = _index_file()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "entries" in data:
                _INDEX_CACHE = data
                return _INDEX_CACHE
        except Exception:
            pass  # 索引坏了就从目录重建，不让它挡住整个管线
    _INDEX_CACHE = {"version": 1, "entries": []}
    return _INDEX_CACHE


def _save_index() -> None:
    global _INDEX_CACHE
    os.makedirs(os.path.dirname(_index_file()), exist_ok=True)
    tmp = _index_file() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_INDEX_CACHE, f, ensure_ascii=False, indent=2)
    os.replace(tmp, _index_file())


def _sha256(path: str, block: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            buf = f.read(block)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 增 / 查
# ---------------------------------------------------------------------------
def add(kind: str, path: str, tags: Optional[List[str]] = None,
        source: str = "local", meta: Optional[dict] = None) -> dict:
    """把文件收进库（按 sha256 去重），返回条目。

    重复入库不会复制第二份，只把新标签并进旧条目。
    """
    if kind not in KINDS:
        raise ValueError(f"未知类型 {kind}，可选 {KINDS}")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    idx = _load_index()
    sha = _sha256(path)
    for e in idx["entries"]:
        if e["sha256"] == sha and e["kind"] == kind:
            for t in (tags or []):
                if t not in e["tags"]:
                    e["tags"].append(t)
            _save_index()
            return dict(e)

    eid = f"{kind}_{sha[:10]}"
    ext = os.path.splitext(path)[1].lower() or ""
    name = os.path.splitext(os.path.basename(path))[0]
    dest_dir = os.path.join(LIB_ROOT, kind)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{eid}_{safe_name(name, kind)}{ext}")

    if os.path.abspath(path) != os.path.abspath(dest):
        shutil.copy2(path, dest)

    entry = {
        "id": eid,
        "kind": kind,
        "name": safe_name(name, kind),
        "path": os.path.abspath(dest),
        "tags": sorted(set((tags or []) + ([kind] if kind == "sfx" else []))),
        "source": source,
        "sha256": sha,
        "size": os.path.getsize(dest),
        "ext": ext,
        "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "uses": 0,
        "meta": meta or {},
    }
    idx["entries"].append(entry)
    _save_index()
    return dict(entry)


def list_entries(kind: Optional[str] = None) -> List[dict]:
    idx = _load_index()
    if kind:
        return [e for e in idx["entries"] if e["kind"] == kind]
    return list(idx["entries"])


def rebuild_index() -> int:
    """扫目录重建索引（手工往库里丢文件后调用）。返回条目数。"""
    global _INDEX_CACHE
    entries = []
    for kind in KINDS:
        d = os.path.join(LIB_ROOT, kind)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            p = os.path.join(d, fn)
            if not os.path.isfile(p):
                continue
            sha = _sha256(p)
            name = os.path.splitext(fn)[0]
            if name.startswith(f"{kind}_") and "_" in name:
                name = name.split("_", 1)[1]
            entries.append(
                {
                    "id": f"{kind}_{sha[:10]}",
                    "kind": kind,
                    "name": safe_name(name, kind),
                    "path": os.path.abspath(p),
                    "tags": [kind],
                    "source": "rebuild",
                    "sha256": sha,
                    "size": os.path.getsize(p),
                    "ext": os.path.splitext(fn)[1].lower(),
                    "added_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "uses": 0,
                    "meta": {},
                }
            )
    _INDEX_CACHE = {"version": 1, "entries": entries}
    _save_index()
    return len(entries)


def _tokens(query: str) -> List[str]:
    raw = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in str(query).lower())
    words = [w for w in raw.split() if w]
    extra: List[str] = []
    for block in raw.split():
        if any("\u4e00" <= ch <= "\u9fff" for ch in block) and len(block) > 1:
            extra.extend(block[i:i + 2] for i in range(len(block) - 1))
    return words + extra


def _score(entry: dict, tokens: List[str]) -> float:
    hay = (entry.get("name", "") + " " + " ".join(entry.get("tags", []))).lower()
    score = 0.0
    for t in tokens:
        if not t:
            continue
        if t in hay:
            score = max(score, 0.40) if score == 0 else score + 0.10
        for syn in SYNONYMS.get(t, []):
            if syn.lower() in hay:
                score = max(score, 0.30) if score == 0 else score + 0.10
                break
    if score and entry.get("kind") == "sfx":
        score += 0.05
    return round(min(score, 0.95), 3)


def search(query: str, kind: Optional[str] = None, k: int = 5,
           min_score: Optional[float] = None) -> List[dict]:
    """按查询词检索库；返回带 `_score` 的条目（低于阈值的不返回）。"""
    thr = LIBRARY_MIN_SCORE if min_score is None else min_score
    tokens = _tokens(query)
    out = []
    for e in list_entries(kind):
        s = _score(e, tokens)
        if s >= thr:
            item = dict(e)
            item["_score"] = s
            out.append(item)
    out.sort(key=lambda x: (-x["_score"], x.get("kind", ""), x.get("name", "")))
    return out[:k]


def pick_for_prompt(prompt: str, kind: str = "image",
                    min_score: Optional[float] = None) -> Optional[dict]:
    """出图/配音前先问库：有没有现成可用的素材。"""
    hits = search(prompt, kind=kind, k=1, min_score=min_score)
    return hits[0] if hits else None


def touch(entry_id: str) -> None:
    """记一次使用（供"常用素材优先"排序，不强求）。"""
    idx = _load_index()
    for e in idx["entries"]:
        if e["id"] == entry_id:
            e["uses"] = int(e.get("uses", 0)) + 1
            _save_index()
            return


# ---------------------------------------------------------------------------
# 程序化音效（离线合成，装好即有内容）
# ---------------------------------------------------------------------------
def _write_wav(path: str, samples: np.ndarray, rate: int = SFX_RATE) -> str:
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return path


def _env(n: int, attack: float, decay: float) -> np.ndarray:
    a = max(1, int(n * attack))
    d = max(1, int(n * decay))
    e = np.ones(n)
    e[:a] = np.linspace(0, 1, a)
    e[-d:] = np.linspace(1, 0, d)
    return e


def _synth_samples(name: str) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(name)) % (2 ** 32))
    if name == "rain":
        n = int(SFX_RATE * 6.0)
        nz = rng.normal(0, 1, n)
        # 两级一阶低通，做出"沙沙"而非白噪
        for _ in range(2):
            nz = np.convolve(nz, np.ones(6) / 6, mode="same")
        return 0.35 * nz * (1 + 0.15 * np.sin(2 * np.pi * 0.3 * np.arange(n) / SFX_RATE))
    if name == "whoosh":
        n = int(SFX_RATE * 0.9)
        t = np.arange(n) / SFX_RATE
        sweep = np.sin(2 * np.pi * (80 + 700 * t) * t)
        nz = np.convolve(rng.normal(0, 1, n), np.ones(4) / 4, mode="same")
        return 0.45 * (0.6 * nz + 0.4 * sweep) * _env(n, 0.35, 0.45)
    if name == "click":
        n = int(SFX_RATE * 0.06)
        sig = rng.normal(0, 1, n) * np.exp(-np.arange(n) / (n / 6))
        return 0.5 * sig
    if name == "typing":
        n = int(SFX_RATE * 2.5)
        out = np.zeros(n)
        ln = min(int(SFX_RATE * 0.03), n)
        for _ in range(18):
            # 末尾要留够 burst 的长度，否则切片短于 burst 会广播失败
            pos = int(rng.integers(0, max(1, n - ln)))
            burst = rng.normal(0, 1, ln) * np.exp(-np.arange(ln) / (ln / 5))
            out[pos:pos + ln] += 0.35 * burst
        return out
    if name == "door":
        n = int(SFX_RATE * 1.6)
        t = np.arange(n) / SFX_RATE
        thump = np.sin(2 * np.pi * (60 - 25 * t) * t) * np.exp(-t * 6)
        creak = 0.25 * np.sin(2 * np.pi * (300 + 120 * t) * t) * np.exp(-t * 2.2)
        return 0.7 * thump + creak
    if name == "thud":
        n = int(SFX_RATE * 0.5)
        t = np.arange(n) / SFX_RATE
        return 0.8 * np.sin(2 * np.pi * 70 * t) * np.exp(-t * 10)
    if name == "incoming":
        n = int(SFX_RATE * 0.8)
        t = np.arange(n) / SFX_RATE
        tone = np.sin(2 * np.pi * (600 + 300 * t) * t)
        return 0.5 * tone * _env(n, 0.1, 0.5)
    if name == "heartbeat":
        n = int(SFX_RATE * 1.2)
        t = np.arange(n) / SFX_RATE
        beat = np.sin(2 * np.pi * 55 * t) * np.exp(-((t % 0.6) * 12))
        return 0.6 * beat
    if name == "chime":
        n = int(SFX_RATE * 1.4)
        t = np.arange(n) / SFX_RATE
        return 0.4 * (np.sin(2 * np.pi * 880 * t) + 0.6 * np.sin(2 * np.pi * 1320 * t)) * _env(n, 0.05, 0.6)
    # 未知名字 → 一段温和的低频噪声，保证永远能返回可用文件
    n = int(SFX_RATE * 1.0)
    return 0.2 * np.convolve(rng.normal(0, 1, n), np.ones(8) / 8, mode="same")


BUILTIN_SFX = ("rain", "whoosh", "click", "typing", "door", "thud", "incoming", "heartbeat", "chime")


def synth_sfx(name: str, tags: Optional[List[str]] = None) -> str:
    """合成（或复用）一个内置音效，返回库内路径。"""
    tags = list(tags or [])
    for zh, en in (("雨声", "rain"), ("转场", "whoosh"), ("风", "whoosh"), ("点击", "click"),
                   ("打字", "typing"), ("门", "door"), ("撞击", "thud"), ("提示音", "incoming")):
        if en == name:
            tags.append(zh)

    # 已合成过就直接复用（同名同内容，不必重复生成）
    for e in list_entries("sfx"):
        if e["name"] == name:
            return e["path"]

    # 临时文件用"库里最终想要的名字"，否则入库后 name 会带 _tmp_ 前缀
    tmp_dir = os.path.join(LIB_ROOT, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp = os.path.join(tmp_dir, f"{safe_name(name)}.wav")
    _write_wav(tmp, _synth_samples(name))
    try:
        entry = add("sfx", tmp, tags=tags, source="synth")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return entry["path"]


def ensure_builtin_sfx() -> List[str]:
    """确保内置音效都在库里（首次安装时调用一次）。"""
    return [synth_sfx(n) for n in BUILTIN_SFX]


# ---------------------------------------------------------------------------
# 组合（配乐/音效叠到台词轨）
# ---------------------------------------------------------------------------
def _duration(path: str) -> float:
    """音频时长（秒）。wav 直接读头，其它交给 ffprobe。"""
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        pass
    import subprocess
    import re

    from . import ffprobe_path

    r = subprocess.run([ffprobe_path(), "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=nw=1:nk=1", path], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def mix_sfx(base_wav: str, sfx: str, at_sec: float = 0.0, gain: float = 0.6,
            out_path: Optional[str] = None) -> str:
    """把音效叠到台词轨（from 0 起算的 at_sec 处），输出时长以台词轨为准。"""
    import subprocess

    if out_path is None:
        out_path = os.path.join(os.path.dirname(base_wav),
                                os.path.splitext(os.path.basename(base_wav))[0] + "_sfx.wav")
    delay_ms = max(0, int(at_sec * 1000))
    fc = (
        f"[1:a]adelay={delay_ms}|{delay_ms},volume={gain}[s];"
        f"[0:a][s]amix=inputs=2:duration=first:dropout_transition=0[a]"
    )
    cmd = [ffmpeg_path(), "-y", "-i", base_wav, "-i", sfx, "-filter_complex", fc,
           "-map", "[a]", out_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"音效混音失败：{r.stderr[-400:]}")
    return os.path.abspath(out_path)


def stats() -> dict:
    entries = list_entries()
    by_kind: Dict[str, int] = {}
    size = 0
    for e in entries:
        by_kind[e["kind"]] = by_kind.get(e["kind"], 0) + 1
        size += int(e.get("size", 0))
    return {"root": LIB_ROOT, "total": len(entries), "by_kind": by_kind,
            "size_mb": round(size / 1024 / 1024, 1)}
