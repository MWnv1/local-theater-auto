# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
#
# Acceptance: A4（素材库 / 音效库，AI 可调库）

import json
import os
import shutil
import tempfile
import unittest

from theater import library


class TestLibraryIndex(unittest.TestCase):
    """索引读写、去重、检索、程序化音效。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="libtest_")
        self._old_root = library.LIB_ROOT
        library.LIB_ROOT = self.tmp
        library.INDEX_PATH = os.path.join(self.tmp, "index.json")
        library._INDEX_CACHE = None

    def tearDown(self):
        library.LIB_ROOT = self._old_root
        library.INDEX_PATH = os.path.join(self._old_root, "index.json")
        library._INDEX_CACHE = None
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_png(self, name="bg.png", color=(200, 120, 60)):
        from PIL import Image

        p = os.path.join(self.tmp, name)
        Image.new("RGB", (64, 64), color).save(p)
        return p

    def test_add_dedupes_by_sha256(self):
        src = self._make_png()
        e1 = library.add("image", src, tags=["房间", "背景"], source="ai")
        e2 = library.add("image", src, tags=["重复"], source="ai")
        self.assertEqual(e1["id"], e2["id"], "同一文件重复入库必须返回同一条目")
        self.assertEqual(len(library.list_entries("image")), 1)

    def test_search_chinese_and_synonym(self):
        self._make_png("rain_scene.png")
        library.synth_sfx("rain")
        library.synth_sfx("door")
        hits = library.search("雨声", kind="sfx", k=3)
        self.assertTrue(hits, "查询'雨声'必须命中内置雨声")
        self.assertEqual(hits[0]["name"], "rain")
        self.assertGreaterEqual(hits[0]["_score"], library.LIBRARY_MIN_SCORE)

    def test_search_returns_empty_for_unrelated(self):
        library.synth_sfx("rain")
        self.assertEqual(library.search("外星飞船引擎轰鸣", kind="sfx", k=3), [])

    def test_index_rebuild_keeps_entries(self):
        self._make_png()
        library.add("image", self._make_png("second.png", (10, 20, 30)), tags=["a"])
        before = len(library.list_entries())
        library.rebuild_index()
        self.assertEqual(len(library.list_entries()), before)

    def test_synth_sfx_is_valid_wav(self):
        import wave

        p = library.synth_sfx("click")
        self.assertTrue(os.path.exists(p))
        with wave.open(p, "rb") as w:
            self.assertGreater(w.getnframes(), 0)
            self.assertEqual(w.getnchannels(), 1)

    def test_mix_sfx_keeps_duration(self):
        from theater import local_tts

        base = local_tts._text_to_audio("测试混音", "mix_test.wav", "", force=True)
        sfx = library.synth_sfx("whoosh")
        out = library.mix_sfx(base, sfx, at_sec=0.2, gain=0.5)
        self.assertTrue(os.path.exists(out))
        d1 = library._duration(base)
        d2 = library._duration(out)
        self.assertLess(abs(d1 - d2), 0.35, "混音不应改变主体时长")

    def test_pick_for_prompt_respects_kind(self):
        self._make_png("room.png")
        library.add("image", os.path.join(self.tmp, "room.png"), tags=["房间", "室内"], source="local")
        hit = library.pick_for_prompt("anime style indoor room, soft lighting", kind="image")
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit["_score"], library.LIBRARY_MIN_SCORE)


class TestLibraryDefaults(unittest.TestCase):
    def test_default_root_on_d_drive(self):
        root = library.LIB_ROOT
        if os.path.isabs(root):
            self.assertTrue(root[:2].upper() == "D:", f"库根目录应在 D 盘：{root}")


if __name__ == "__main__":
    unittest.main()
