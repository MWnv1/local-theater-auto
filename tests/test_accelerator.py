# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
#
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Project: local-theater-auto
# Websites: https://imworkingnow.cc.cd | https://imwn.cc.cd | https://imwn.dpdns.org
#
# Acceptance: A8（CUDA / Tensor Core 加速 + 按本机配置量化）

import os
import unittest

from theater import accelerator


class TestDetect(unittest.TestCase):
    def test_detect_reports_machine(self):
        cfg = accelerator.detect()
        for key in ("cuda", "tier", "torch", "cores", "ram_gb"):
            self.assertIn(key, cfg)
        self.assertIn(cfg["tier"], ("cpu", "low", "mid", "high", "xl"))

    def test_tier_follows_vram(self):
        cfg = accelerator.detect()
        vram = float(cfg["vram_gb"] or 0)
        if not cfg["cuda"]:
            self.assertEqual(cfg["tier"], "cpu")
        elif vram >= 24:
            self.assertEqual(cfg["tier"], "xl")
        elif vram >= 12:
            self.assertEqual(cfg["tier"], "high")
        elif vram >= 8:
            self.assertEqual(cfg["tier"], "mid")
        else:
            self.assertEqual(cfg["tier"], "low")


class TestPlan(unittest.TestCase):
    def test_plan_shape(self):
        p = accelerator.plan()
        for key in ("enabled", "tier", "dtype", "attention", "tf32", "llm_quant",
                    "channels_last", "vram_gb"):
            self.assertIn(key, p)
        self.assertIn(p["dtype"], ("fp16", "bf16", "fp32"))
        self.assertIn(p["attention"], ("flash", "mem_efficient", "math"))
        self.assertIn(p["llm_quant"], ("4bit", "8bit", "none"))

    def test_attention_is_actually_runnable(self):
        # 报出来的内核必须真能跑，不能只是"开关打开了"
        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        if not torch.cuda.is_available():
            self.skipTest("没有 CUDA")
        mapping = {"flash": SDPBackend.FLASH_ATTENTION,
                   "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
                   "math": SDPBackend.MATH}
        q = torch.randn(1, 4, 128, 32, device="cuda", dtype=torch.float16)
        with sdpa_kernel(mapping[accelerator.plan()["attention"]]):
            F.scaled_dot_product_attention(q, q, q)
        torch.cuda.synchronize()

    def test_apply_global_sets_tf32_and_is_idempotent(self):
        p = accelerator.apply_global()
        if not p["enabled"]:
            self.skipTest("未启用加速")
        import torch

        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertEqual(torch.get_float32_matmul_precision(), "high")
        self.assertEqual(accelerator.apply_global(), p)

    def test_llm_kwargs_has_quantization_on_gpu(self):
        kw, note = accelerator.llm_kwargs()
        p = accelerator.plan()
        if not p["enabled"]:
            self.skipTest("未启用加速")
        if p["llm_quant"] == "none":
            self.assertIn("torch_dtype", kw)
        else:
            self.assertIn("quantization_config", kw)
            self.assertEqual(kw["quantization_config"].load_in_4bit, p["llm_quant"] == "4bit")
        self.assertTrue(note)

    def test_accel_off_disables_everything(self):
        os.environ["ACCEL"] = "off"
        accelerator._PLAN = None
        try:
            p = accelerator.plan()
            self.assertFalse(p["enabled"])
            self.assertEqual(p["llm_quant"], "none")
            self.assertFalse(p["tf32"])
        finally:
            os.environ.pop("ACCEL", None)
            accelerator._PLAN = None

    def test_tensor_tflops_positive_on_gpu(self):
        if not accelerator.cuda_available():
            self.skipTest("没有 CUDA")
        out = accelerator.tensor_tflops(measure=True)
        self.assertGreater(float(out["bf16"]), 1.0)
        self.assertGreater(float(out["fp32"]), 1.0)

    def test_no_measure_returns_empty(self):
        self.assertEqual(accelerator.tensor_tflops(measure=False), {})

    def test_optimize_pipe_is_tolerant(self):
        class FakePipe:
            def __init__(self):
                self.calls = []

            def to(self, *a, **kw):
                self.calls.append(kw)
                return self

        pipe = FakePipe()
        applied = accelerator.optimize_pipe(pipe)
        self.assertIn("channels_last", applied)


if __name__ == "__main__":
    unittest.main()