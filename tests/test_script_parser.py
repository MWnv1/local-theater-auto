# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
"""剧本解析的单元测试（纯标准库，无需安装任何依赖）。

运行：
    python -m unittest discover -s tests -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theater.script_parser import format_scenes_brief, parse_full_script


class TestScriptParser(unittest.TestCase):
    def test_basic_two_scenes(self):
        text = """
【镜头1】
【小林：站位：左】
【阿杰：站位：右】
小林：这个模型真能离线跑？
阿杰：4bit 的，NPU 上 30 token/s。

【镜头2】
小林：走吧。
"""
        s = parse_full_script(text)
        self.assertEqual(s["total_scenes"], 2)
        self.assertEqual(s["all_characters"], ["小林", "阿杰"])
        self.assertEqual(s["scenes"][0]["positions"], {"小林": "左", "阿杰": "右"})
        self.assertEqual(len(s["scenes"][0]["lines"]), 2)

    def test_narrator_not_a_character(self):
        text = """
【镜头1】
【旁白】夜色渐深。
小林：这句是我说的。
"""
        s = parse_full_script(text)
        self.assertNotIn("旁白", s["all_characters"])
        self.assertEqual(s["all_characters"], ["小林"])
        first = s["scenes"][0]["lines"][0]
        self.assertEqual(first["character"], "旁白")
        self.assertTrue(first["narrator"])

    def test_meta_line_not_dialogue(self):
        text = """
【镜头1】
场景：机房
小林：好吵。
"""
        s = parse_full_script(text)
        self.assertEqual(s["scenes"][0]["scene_desc"], "机房")
        self.assertEqual(len(s["scenes"][0]["lines"]), 1)

    def test_text_before_first_scene_kept(self):
        text = """
小林：开场白。
【镜头1】
阿杰：接话。
"""
        s = parse_full_script(text)
        self.assertEqual(s["total_scenes"], 1)
        self.assertEqual(len(s["scenes"][0]["lines"]), 2)

    def test_default_positions_by_count(self):
        text = """
【镜头1】
小林：一
阿杰：二
小张：三
"""
        s = parse_full_script(text)
        self.assertEqual(
            s["scenes"][0]["positions"],
            {"小林": "左", "阿杰": "居中", "小张": "右"},
        )

    def test_empty_script_raises(self):
        with self.assertRaises(ValueError):
            parse_full_script("   \n  ")

    def test_brief_render(self):
        s = parse_full_script("【镜头1】\n小林：hi\n")
        brief = format_scenes_brief(s)
        self.assertIn("镜头1", brief)
        self.assertIn("小林", brief)


if __name__ == "__main__":
    unittest.main(verbosity=2)
