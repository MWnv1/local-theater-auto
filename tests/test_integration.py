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

"""新能力与旧接口的接缝测试：只断言"接上了、签名没破"，不依赖模型。"""

import inspect
import os
import tempfile
import unittest

from PIL import Image

from theater import frame_compose, video_export


class TestVideoExportContract(unittest.TestCase):
    """export_final_video 必须向后兼容：老调用只传 4 个参数也能用。"""

    def test_export_accepts_anims_and_sfx(self):
        sig = inspect.signature(video_export.export_final_video)
        self.assertIn("anims", sig.parameters)
        self.assertIn("sfx_map", sig.parameters)
        # 新增参数必须有默认值，否则旧调用会 TypeError
        self.assertIsNone(sig.parameters["anims"].default)
        self.assertIsNone(sig.parameters["sfx_map"].default)

    def test_build_clip_accepts_anim_and_sfx(self):
        sig = inspect.signature(video_export._build_clip)
        self.assertIn("anim", sig.parameters)
        self.assertIn("sfx", sig.parameters)


class TestSfxPicking(unittest.TestCase):
    """台词关键词 → 内置音效，不落库、不合成，纯映射。"""

    def test_keyword_maps(self):
        from pipeline_full import _sfx_name_for

        self.assertEqual(_sfx_name_for("外面下着大雨"), "rain")
        self.assertEqual(_sfx_name_for("他一直在敲键盘写代码"), "typing")
        self.assertEqual(_sfx_name_for("她推开门走进来"), "door")
        self.assertEqual(_sfx_name_for("手机响了，是消息提示"), "incoming")
        self.assertEqual(_sfx_name_for("今天天气不错"), "")


class TestActorFrameCompose(unittest.TestCase):
    """compose_actor_frame：给定"动作中的角色图"，必须照样出 1920×1080 整帧。"""

    def test_output_size_and_subtitle(self):
        with tempfile.TemporaryDirectory() as td:
            bg = Image.new("RGB", (640, 360), (30, 40, 60))
            bg_path = os.path.join(td, "bg.png")
            bg.save(bg_path)

            char = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
            for x in range(60, 140):
                for y in range(40, 380):
                    char.putpixel((x, y), (240, 200, 160, 255))

            out = os.path.join(td, "frame.png")
            frame_compose.compose_actor_frame(
                bg_path, {"小周": char}, {"小周": "左"}, "夜里三点，机房还亮着灯。", out
            )
            self.assertTrue(os.path.exists(out))
            with Image.open(out) as im:
                self.assertEqual(im.size, (frame_compose.VIDEO_W, frame_compose.VIDEO_H))
                self.assertEqual(im.mode, "RGB")

    def test_unknown_position_falls_back_to_center(self):
        with tempfile.TemporaryDirectory() as td:
            bg_path = os.path.join(td, "bg.png")
            Image.new("RGB", (320, 180), (0, 0, 0)).save(bg_path)
            out = os.path.join(td, "f2.png")
            # 站位写成剧本里没出现过的地方 → 不报错，按居中处理
            frame_compose.compose_actor_frame(
                bg_path, {"甲": Image.new("RGBA", (100, 200), (255, 0, 0, 255))},
                {"甲": "天上"}, "测试", out,
            )
            self.assertTrue(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
