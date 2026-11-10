# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
#
# Acceptance: A2b（人物只做平移/旋转）+ A3b（逐帧算偏移量再生成）

import json
import os
import random
import shutil
import tempfile
import unittest

from PIL import Image, ImageDraw

from theater import animator, video_ref


def _busy_image(path: str, w: int = 320, h: int = 180) -> str:
    """造一张高纹理图：相位相关需要非平坦画面，纯色图测不出位移。"""
    img = Image.new("RGB", (w, h), (28, 32, 38))
    d = ImageDraw.Draw(img)
    rng = random.Random(11)
    for _ in range(50):
        x, y = rng.randint(0, w - 30), rng.randint(0, h - 30)
        d.rectangle([x, y, x + rng.randint(8, 26), y + rng.randint(8, 26)],
                    fill=(rng.randint(60, 240), rng.randint(60, 240), rng.randint(60, 240)))
    img.save(path)
    return path


class TestTransformActions(unittest.TestCase):
    def test_new_actions_registered(self):
        for a in ("pan_left", "pan_right", "pan_up", "pan_down", "rotate", "spin"):
            self.assertIn(a, animator.ACTIONS)

    def test_pan_left_is_negative_dx(self):
        p = animator.pose_at("pan_left", 1.0)   # 4 秒周期的 1/4 → sin=1
        self.assertLess(p["dx"], 0)
        self.assertAlmostEqual(p["dy"], 0.0, places=5)
        self.assertAlmostEqual(p["rot"], 0.0, places=5)

    def test_pan_up_is_negative_dy(self):
        p = animator.pose_at("pan_up", 1.0)
        self.assertLess(p["dy"], 0)
        self.assertAlmostEqual(p["dx"], 0.0, places=5)

    def test_pan_right_opposite_of_left(self):
        l = animator.pose_at("pan_left", 1.0)["dx"]
        r = animator.pose_at("pan_right", 1.0)["dx"]
        self.assertGreater(r, 0)
        self.assertLess(l, 0)
        self.assertAlmostEqual(abs(r), abs(l), delta=0.001)

    def test_rotate_grows_with_time(self):
        a = animator.pose_at("rotate", 0.2)["rot"]
        b = animator.pose_at("rotate", 0.75)["rot"]   # 3 秒周期，0.75s 时 sin 最大
        self.assertGreater(b, a)
        self.assertGreater(b, 9.0)

    def test_spin_wraps_within_one_turn(self):
        p = animator.pose_at("spin", 5.9)
        self.assertGreaterEqual(p["rot"], 0.0)
        self.assertLessEqual(p["rot"], 360.0)

    def test_pose_is_pure(self):
        self.assertEqual(animator.pose_at("pan_left", 0.37), animator.pose_at("pan_left", 0.37))

    def test_transform_px_scales_to_canvas(self):
        px = animator.transform_px({"dx": -0.18, "dy": 0.1, "rot": 5.0}, (1000, 500))
        self.assertAlmostEqual(px["dx"], -180.0, delta=0.1)
        self.assertAlmostEqual(px["dy"], 50.0, delta=0.1)
        self.assertAlmostEqual(px["rot"], 5.0)


class TestWarp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="warptest_")
        self.base = _busy_image(os.path.join(self.tmp, "a.png"))
        self.img = Image.open(self.base).convert("RGBA")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shift_moves_content_right(self):
        moved = animator.warp(self.img, 24.0, 0.0)
        p = os.path.join(self.tmp, "b.png")
        moved.convert("RGB").save(p)
        off = video_ref.frame_offset(self.base, p)
        self.assertAlmostEqual(off["dx"], 24.0, delta=2.5)
        self.assertAlmostEqual(off["dy"], 0.0, delta=1.5)

    def test_rotate_is_recovered_in_same_sign(self):
        turned = animator.warp(self.img, 0.0, 0.0, 5.0)
        p = os.path.join(self.tmp, "c.png")
        turned.convert("RGB").save(p)
        off = video_ref.frame_offset(self.base, p)
        self.assertAlmostEqual(off["rot"], 5.0, delta=2.0)
        self.assertGreater(off["conf"], 0.6)

    def test_no_motion_returns_zero(self):
        off = video_ref.frame_offset(self.base, self.base)
        self.assertAlmostEqual(off["dx"], 0.0, delta=1.0)
        self.assertAlmostEqual(off["rot"], 0.0, delta=1.0)


class TestMotionAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mottest_")
        self.base = _busy_image(os.path.join(self.tmp, "f0.png"))
        self.frames = [self.base]
        img = Image.open(self.base).convert("RGBA")
        for i in range(1, 4):
            p = os.path.join(self.tmp, f"f{i}.png")
            animator.warp(img, 10.0 * i, 0.0).convert("RGB").save(p)
            self.frames.append(p)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_analyze_motion_per_frame_offsets(self):
        m = video_ref.analyze_motion(self.frames)
        self.assertEqual(len(m["offsets"]), 3)
        for off in m["offsets"]:
            self.assertAlmostEqual(off["dx"], 10.0, delta=2.5)
        self.assertEqual(m["frames"], 4)

    def test_describe_motion_calls_it_a_right_pan(self):
        m = video_ref.analyze_motion(self.frames)
        s = m["summary"]
        self.assertIn("向右", s["label"])
        self.assertIn("right", s["prompt_hint"])
        self.assertGreater(s["speed_px"], 5.0)

    def test_static_offsets_labelled_static(self):
        s = video_ref.describe_motion([{"dx": 0.2, "dy": 0.1, "rot": 0.0, "conf": 0.9}] * 4)
        self.assertIn("静止", s["label"])
        self.assertGreaterEqual(s["static_ratio"], 0.6)

    def test_rotation_mentioned_in_summary(self):
        s = video_ref.describe_motion([{"dx": 0.0, "dy": 0.0, "rot": 3.0, "conf": 0.9}] * 3)
        self.assertIn("顺时针", s["label"])
        self.assertAlmostEqual(s["total_rot"], 9.0, delta=0.01)

    def test_llm_hint_off_by_default(self):
        os.environ["SIM_MOTION_LLM"] = "0"
        self.assertEqual(video_ref.motion_hint_llm({"mean_dx": 5, "mean_dy": 0}), "")

    def test_local_motion_labelled_local(self):
        # 整帧不动、但有 3% 像素在变 → 主体自己在动，不能说成"静止"
        s = video_ref.describe_motion([{"dx": 0.0, "dy": 0.0, "rot": 0.0,
                                       "conf": 0.9, "changed": 0.03}] * 4)
        self.assertEqual(s["kind"], "local")
        self.assertIn("局部动作", s["label"])
        self.assertIn("only the subject moves", s["prompt_hint"])

    def test_frame_offset_reports_changed_ratio(self):
        moved = video_ref.frame_offset(self.frames[0], self.frames[1])
        same = video_ref.frame_offset(self.base, self.base)
        self.assertGreater(moved["changed"], 0.0)
        self.assertLess(same["changed"], 0.02)


class TestRenderTransform(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="rendertest_")
        self.char = os.path.join(self.tmp, "char.png")
        img = Image.new("RGBA", (200, 400), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([70, 20, 130, 120], fill=(240, 200, 170, 255))
        d.rectangle([60, 130, 140, 300], fill=(80, 120, 200, 255))
        d.rectangle([60, 300, 95, 390], fill=(40, 60, 90, 255))
        d.rectangle([105, 300, 140, 390], fill=(40, 60, 90, 255))
        img.save(self.char)
        self.rig = {"image": self.char, "signature": "test", "parts": {}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pan_left_has_no_gap_when_small(self):
        f0 = animator.render_frame(self.rig, "pan_left", 0.0)
        f1 = animator.render_frame(self.rig, "pan_left", 1.0)
        self.assertGreater(animator.frame_diff(f0, f1), 0.02)

    def test_clip_manifest_records_motion_tracks(self):
        info = animator.render_clip(self.rig, "rotate", seconds=0.5, fps=12,
                                    out_dir=os.path.join(self.tmp, "clip"))
        with open(info["manifest"], encoding="utf-8") as f:
            m = json.load(f)
        self.assertEqual(len(m["motion_tracks"]), 6)
        self.assertIn("px", m["motion_tracks"][0])


if __name__ == "__main__":
    unittest.main()