# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
#
# Acceptance: A3b 的回归项 —— "果冻"事故（去噪步数不足 / 尺度太小 / init 累积）

import os
import tempfile
import unittest

from PIL import Image, ImageDraw

from theater import video_ref


class FakePipe:
    """假管道：只记录参数，不加载任何权重。"""

    def __init__(self, output: Image.Image):
        self.output = output
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)

        class R:
            images = [self.output]

        return R()


def _sample_frame(path: str, size=(768, 432)) -> str:
    im = Image.new("RGB", size, (40, 40, 40))
    d = ImageDraw.Draw(im)
    for x in range(0, size[0], 48):
        d.rectangle([x, 40, x + 24, size[1] - 40], fill=(200, 180, 120))
    d.ellipse([size[0] // 2 - 40, size[1] // 2 - 40, size[0] // 2 + 40, size[1] // 2 + 40],
              fill=(240, 240, 240))
    im.save(path)
    return path


class TestSimQuality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="simq_")
        self.src = _sample_frame(os.path.join(self.tmp, "src.png"))
        # 单测里不许碰真权重：ControlNet 一旦被判可用就会去加载 720 MB 模型
        self._cn_mode = video_ref.SIM_CONTROLNET
        video_ref._cn_pipe = None
        video_ref.SIM_CONTROLNET = "off"

    def tearDown(self):
        video_ref.SIM_CONTROLNET = self._cn_mode

    def test_defaults_are_the_fixed_ones(self):
        self.assertEqual(video_ref.SIM_SIZE, 768)          # 512 太小，细节被抹平
        self.assertGreaterEqual(video_ref.SIM_STEPS, 24)   # 8 步是"果冻"的直接原因
        self.assertFalse(video_ref.MOTION_WARP_ON)         # 逐帧累积瑕疵，默认关
        self.assertGreaterEqual(video_ref.SIM_MIN_DENOISE, 12)

    def test_restyle_enforces_minimum_denoise_steps(self):
        # 传入 8 步也要被抬到 steps*strength>=SIM_MIN_DENOISE
        fake = FakePipe(Image.new("RGB", (768, 432), (90, 90, 90)))
        old_pipe = video_ref._pipe
        video_ref._pipe = fake
        try:
            out = os.path.join(self.tmp, "a.png")
            video_ref._restyle(self.src, out, "p", "n", 0.35, 8)
        finally:
            video_ref._pipe = old_pipe
        kw = fake.calls[-1]
        self.assertGreaterEqual(kw["num_inference_steps"] * kw["strength"], 12)
        self.assertEqual(kw["strength"], 0.35)

    def test_restyle_passes_fixed_seed(self):
        fake = FakePipe(Image.new("RGB", (768, 432), (90, 90, 90)))
        old_pipe = video_ref._pipe
        video_ref._pipe = fake
        try:
            video_ref._restyle(self.src, os.path.join(self.tmp, "b.png"), "p", "n", 0.35, 32)
        finally:
            video_ref._pipe = old_pipe
        self.assertIn("generator", fake.calls[-1])

    def test_guard_blends_flat_output_back_to_source(self):
        # 纯色输出没有任何边缘 → IoU 极低 → 必须混回原帧，不能交一张"糊块"
        fake = FakePipe(Image.new("RGB", (768, 432), (10, 220, 10)))
        old_pipe = video_ref._pipe
        video_ref._pipe = fake
        try:
            out = os.path.join(self.tmp, "c.png")
            video_ref._restyle(self.src, out, "p", "n", 0.9, 32)
        finally:
            video_ref._pipe = old_pipe
        im = Image.open(out).convert("RGB").resize((768, 432))
        # 原帧是条纹+底色（3 色），假管道的纯绿只有 1 色；混合后必须还有结构
        colors = im.getcolors(maxcolors=1000000) or []
        self.assertGreaterEqual(len(colors), 2, "纯色输出说明 guard 没生效")
        import numpy as np

        arr = np.asarray(im, dtype=np.int16)
        self.assertGreater(float(np.abs(arr - np.array([10, 220, 10])).mean()), 5.0)

    def test_guard_keeps_good_output(self):
        src = Image.open(self.src).convert("RGB")
        self.assertGreater(video_ref._edge_iou(src, src), 0.95)
        same, blended = video_ref._guard_blend(src, src, 1.0)
        self.assertFalse(blended)
        self.assertEqual(list(same.size), list(src.size))

    def test_edge_iou_detects_garbage(self):
        good = Image.open(self.src).convert("RGB")
        bad = Image.new("RGB", good.size, (255, 0, 255))
        self.assertLess(video_ref._edge_iou(good, bad), 0.20)

    def test_canny_edges_is_black_white_and_same_size(self):
        im = Image.open(self.src).convert("RGB")
        edges = video_ref._canny_edges(im)
        self.assertEqual(edges.size, im.size)
        self.assertEqual(edges.mode, "RGB")
        grey = [p[0] for p in edges.convert("RGB").getdata()]
        self.assertGreater(max(grey), 200)      # 有白线
        self.assertGreater(min(grey), -1)       # 只是确认取值合法
        flat = video_ref._canny_edges(Image.new("RGB", (256, 144), (128, 128, 128)))
        self.assertEqual(max(p[0] for p in flat.getdata()), 0)   # 纯色无边缘 → 全黑

    def test_controlnet_available_when_weights_present(self):
        import tempfile as tf

        old_dir = video_ref.SIM_CONTROLNET_DIR
        try:
            d = tf.mkdtemp(prefix="cn_")
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            with open(os.path.join(d, "diffusion_pytorch_model.fp16.safetensors"), "wb") as f:
                f.write(b"x")
            video_ref.SIM_CONTROLNET_DIR = d
            video_ref.SIM_CONTROLNET = "auto"
            self.assertTrue(video_ref._controlnet_available())
            self.assertTrue(video_ref._controlnet_on())
            self.assertAlmostEqual(video_ref._default_strength(), 0.35)  # 0.45 会刷颗粒
            video_ref.SIM_CONTROLNET = "off"
            self.assertFalse(video_ref._controlnet_on())                 # 显式关掉
        finally:
            video_ref.SIM_CONTROLNET_DIR = old_dir

    def test_controlnet_off_when_dir_missing(self):
        import tempfile as tf

        old_dir = video_ref.SIM_CONTROLNET_DIR
        old_mode = video_ref.SIM_CONTROLNET
        try:
            video_ref.SIM_CONTROLNET_DIR = tf.mkdtemp(prefix="nocn_")
            video_ref.SIM_CONTROLNET = "on"
            self.assertFalse(video_ref._controlnet_available())
            self.assertFalse(video_ref._controlnet_on())      # on 但没权重 → 回落
            self.assertAlmostEqual(video_ref._default_strength(), 0.35)
            video_ref.SIM_CONTROLNET = "auto"
            self.assertFalse(video_ref._controlnet_on())
        finally:
            video_ref.SIM_CONTROLNET_DIR = old_dir
            video_ref.SIM_CONTROLNET = old_mode


if __name__ == "__main__":
    unittest.main()