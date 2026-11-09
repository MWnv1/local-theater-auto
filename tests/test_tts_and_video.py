# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
#
# Acceptance: A1（AI 声音克隆替换 SAPI）+ A3（参考视频 → 相似视频）

import os
import shutil
import tempfile
import unittest
from unittest import mock

from theater import tts_clone


class TestTtsEngineSelection(unittest.TestCase):
    def test_default_engine_is_auto(self):
        self.assertIn(tts_clone.ENGINE, ("auto", "clone", "sapi"))

    def test_resolve_engine_falls_back(self):
        with mock.patch.object(tts_clone, "_clone_available", return_value=False):
            self.assertEqual(tts_clone.resolve_engine("auto"), "sapi")
            self.assertEqual(tts_clone.resolve_engine("clone"), "sapi")
        with mock.patch.object(tts_clone, "_clone_available", return_value=True):
            self.assertEqual(tts_clone.resolve_engine("auto"), "clone")

    def test_cache_key_depends_on_ref_audio(self):
        k1 = tts_clone.cache_key("你好", "refA.wav", "clone")
        k2 = tts_clone.cache_key("你好", "refB.wav", "clone")
        self.assertNotEqual(k1, k2)
        k3 = tts_clone.cache_key("你好", "refA.wav", "clone")
        self.assertEqual(k1, k3)

    def test_voice_map_parsing(self):
        m = tts_clone.parse_voice_map("小周=zhou.wav,小林=lin.wav")
        self.assertEqual(len(m), 2)
        # 相对路径按 TTS_REF_DIR 解析（多机器、多目录下都能定位到同一个音色）
        self.assertTrue(os.path.isabs(m["小周"]))
        self.assertTrue(m["小周"].endswith("zhou.wav"))
        self.assertTrue(m["小林"].endswith("lin.wav"))


class TestTtsFallback(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ttstest_")
        self._old = (tts_clone._AUDIO_DIR, tts_clone.REF_DIR)
        tts_clone._AUDIO_DIR = self.tmp
        tts_clone.REF_DIR = self.tmp

    def tearDown(self):
        tts_clone._AUDIO_DIR, tts_clone.REF_DIR = self._old
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_synth_without_ref_falls_back_to_sapi(self):
        out = tts_clone.synth("这句话用系统音色先兜底。", ref_audio="", engine="auto")
        self.assertTrue(os.path.exists(out))
        self.assertGreater(os.path.getsize(out), 0)
        self.assertEqual(tts_clone.last_engine(), "sapi")


class TestVideoRefPure(unittest.TestCase):
    def test_parse_fps_string(self):
        from theater import video_ref

        self.assertAlmostEqual(video_ref._parse_fps("30000/1001"), 29.97, places=2)
        self.assertAlmostEqual(video_ref._parse_fps("25/1"), 25.0)
        self.assertEqual(video_ref._parse_fps("0/0"), 25.0)

    def test_plan_frames_matches_duration(self):
        from theater import video_ref

        plan = video_ref.plan_frames(duration=3.0, fps=24, max_frames=48)
        self.assertEqual(plan["count"], 48)
        self.assertAlmostEqual(plan["step_frames"], 3.0 * 24 / 48, places=2)


if __name__ == "__main__":
    unittest.main()
