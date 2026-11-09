# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
#
# Acceptance: A2（骨骼绑定 + 动作驱动，人物真的会动）

import json
import os
import shutil
import tempfile
import unittest

from theater import animator, rigging


def _make_char(path: str, w: int = 256, h: int = 512) -> str:
    """造一个近似人形的测试立绘：头 + 躯干 + 两臂 + 两腿。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = w // 2
    d.ellipse([cx - w // 7, 8, cx + w // 7, 8 + w // 4], fill=(240, 200, 170, 255))  # 头
    d.rectangle([cx - w // 6, h // 4, cx + w // 6, int(h * 0.62)], fill=(80, 120, 200, 255))  # 躯干
    d.rectangle([cx - w // 4, h // 4 + 6, cx - w // 6, int(h * 0.55)], fill=(60, 90, 160, 255))  # 左臂
    d.rectangle([cx + w // 6, h // 4 + 6, cx + w // 4, int(h * 0.55)], fill=(60, 90, 160, 255))  # 右臂
    d.rectangle([cx - w // 6, int(h * 0.62), cx - 4, h - 8], fill=(40, 60, 90, 255))  # 左腿
    d.rectangle([cx + 4, int(h * 0.62), cx + w // 6, h - 8], fill=(40, 60, 90, 255))  # 右腿
    img.save(path)
    return path


class TestRigging(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rigtest_")
        self.char = _make_char(os.path.join(self.tmp, "char.png"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_build_rig_has_parts_and_joints(self):
        rig = rigging.build_rig(self.char, out_dir=self.tmp)
        for key in ("head", "torso", "arm_l", "arm_r", "leg_l", "leg_r"):
            self.assertIn(key, rig["parts"], f"缺少部件 {key}")
        self.assertGreaterEqual(len(rig["joints"]), 4)
        self.assertTrue(os.path.exists(rig["rig_path"]))
        loaded = rigging.load_rig(rig["rig_path"])
        self.assertEqual(len(loaded["parts"]), len(rig["parts"]))

    def test_parts_cover_pixels(self):
        rig = rigging.build_rig(self.char, out_dir=self.tmp)
        total = sum(p["bbox"][2] - p["bbox"][0] for p in rig["parts"].values())
        self.assertGreater(total, 0)

    def test_rig_is_stable(self):
        r1 = rigging.build_rig(self.char, out_dir=self.tmp)
        r2 = rigging.build_rig(self.char, out_dir=self.tmp)
        self.assertEqual(r1["signature"], r2["signature"])


class TestAnimator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="animtest_")
        self.char = _make_char(os.path.join(self.tmp, "char.png"))
        self.rig = rigging.build_rig(self.char, out_dir=self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_actions_registered(self):
        for a in ("idle", "nod", "wave", "walk", "talk"):
            self.assertIn(a, animator.ACTIONS)

    def test_frames_differ_over_time(self):
        f0 = animator.render_frame(self.rig, "wave", 0.0)
        f1 = animator.render_frame(self.rig, "wave", 0.5)
        diff = animator.frame_diff(f0, f1)
        self.assertGreater(diff, 0.005, "动作在 0.5 秒内必须产生可见位移")

    def test_idle_breathing_still_moves(self):
        f0 = animator.render_frame(self.rig, "idle", 0.0)
        f1 = animator.render_frame(self.rig, "idle", 1.2)
        self.assertGreater(animator.frame_diff(f0, f1), 0.0005)

    def test_render_clip_writes_sequence(self):
        info = animator.render_clip(self.rig, "walk", seconds=1.0, fps=12, out_dir=os.path.join(self.tmp, "clip"))
        self.assertTrue(os.path.isdir(info["dir"]))
        self.assertEqual(info["frames"], 12)
        names = sorted(os.listdir(info["dir"]))
        self.assertEqual(len([n for n in names if n.endswith(".png")]), 12)
        self.assertTrue(os.path.exists(info["manifest"]))
        with open(info["manifest"], encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(m["fps"], 12)
        self.assertGreaterEqual(len(m["joint_tracks"]), 1)

    def test_pose_at_is_deterministic(self):
        a = animator.pose_at("wave", 0.37)
        b = animator.pose_at("wave", 0.37)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
