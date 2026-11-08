# SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Commercial
# Copyright 2026 MWnv1 (GitHub: https://github.com/MWnv1)
# Dual-licensed: GNU AGPL-3.0-only (LICENSE) or commercial (COMMERCIAL-LICENSE.md)
"""硬件适配的决策逻辑单测（纯标准库，不探测真实硬件）。

覆盖：厂商归一、主显卡挑选、三家编码器优先级、编码器画质参数、
绘图后端选择。这些函数一旦判错，NVIDIA/AMD 机器上就会拼出错误的
ffmpeg 参数（比如给 NVENC 传 -crf），所以必须锁死。
"""

import os
import unittest

from theater import hardware


class VendorTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("GPU_VENDOR", None)

    def test_normalize_nvidia(self):
        self.assertEqual(hardware.normalize_vendor("NVIDIA GeForce RTX 5060 Ti"), "nvidia")
        self.assertEqual(hardware.normalize_vendor("Quadro RTX 4000"), "nvidia")

    def test_normalize_amd(self):
        self.assertEqual(hardware.normalize_vendor("AMD Radeon RX 7800 XT"), "amd")
        self.assertEqual(hardware.normalize_vendor("Radeon (TM) Graphics"), "amd")

    def test_normalize_intel(self):
        self.assertEqual(hardware.normalize_vendor("Intel(R) Arc(TM) A770 Graphics"), "intel")
        self.assertEqual(hardware.normalize_vendor("Intel(R) UHD Graphics 630"), "intel")

    def test_normalize_unknown(self):
        self.assertEqual(hardware.normalize_vendor("Microsoft Basic Display Adapter"), "unknown")
        self.assertEqual(hardware.normalize_vendor(""), "unknown")
        self.assertEqual(hardware.normalize_vendor(None), "unknown")

    def test_primary_prefers_discrete(self):
        adapters = ["Intel(R) UHD Graphics 630", "NVIDIA GeForce RTX 4060"]
        self.assertEqual(hardware.primary_vendor(adapters), "nvidia")
        adapters = ["Intel(R) UHD Graphics 630", "AMD Radeon RX 7600"]
        self.assertEqual(hardware.primary_vendor(adapters), "amd")
        self.assertEqual(hardware.primary_vendor(["Intel(R) UHD Graphics 630"]), "intel")
        self.assertEqual(hardware.primary_vendor([]), "unknown")

    def test_env_override(self):
        os.environ["GPU_VENDOR"] = "amd"
        try:
            self.assertEqual(hardware.primary_vendor(["NVIDIA GeForce RTX 4060"]), "amd")
        finally:
            os.environ.pop("GPU_VENDOR", None)


FFMPEG_ALL = """
 V..... h264_nvenc           NVIDIA NVENC H.264 encoder
 V..... hevc_nvenc           NVIDIA NVENC hevc encoder
 V..... h264_amf             AMD AMF H.264 Encoder
 V..... hevc_amf             AMD AMF HEVC encoder
 V..... h264_qsv             Intel Quick Sync Video H.264 encoder
 V..... hevc_qsv             Intel Quick Sync Video HEVC encoder
 V..... h264_videotoolbox    VideoToolbox H.264 Encoder
 V..... libx264              libx264 H.264
 V..... libx265              libx265 H.265
 V..... aac                  AAC (Advanced Audio Coding)
"""

FFMPEG_X264_ONLY = " V..... libx264  libx264 H.264\n V..... aac  AAC\n"


class EncoderTest(unittest.TestCase):
    def test_vendor_preferred_encoder(self):
        self.assertEqual(hardware.select_encoder("nvidia", FFMPEG_ALL), "h264_nvenc")
        self.assertEqual(hardware.select_encoder("amd", FFMPEG_ALL), "h264_amf")
        self.assertEqual(hardware.select_encoder("intel", FFMPEG_ALL), "h264_qsv")

    def test_hevc_codec(self):
        self.assertEqual(hardware.select_encoder("nvidia", FFMPEG_ALL, codec="hevc"), "hevc_nvenc")
        self.assertEqual(hardware.select_encoder("amd", FFMPEG_ALL, codec="hevc"), "hevc_amf")

    def test_software_fallback(self):
        self.assertEqual(hardware.select_encoder("nvidia", FFMPEG_X264_ONLY), "libx264")
        self.assertEqual(hardware.select_encoder("amd", ""), "libx264")
        self.assertEqual(hardware.select_encoder("unknown", ""), "libx264")
        self.assertEqual(hardware.select_encoder("unknown", FFMPEG_X264_ONLY, codec="hevc"), "libx265")

    def test_explicit_preference_wins(self):
        self.assertEqual(hardware.select_encoder("nvidia", FFMPEG_ALL, preferred="h264_qsv"), "h264_qsv")
        # 指定了但不可用 -> 按厂商回退
        self.assertEqual(hardware.select_encoder("nvidia", FFMPEG_ALL, preferred="h264_amf_typo"), "h264_nvenc")
        # auto 不是真编码器名，不应被当成可用
        self.assertEqual(hardware.select_encoder("intel", FFMPEG_ALL, preferred="auto"), "h264_qsv")

    def test_quality_args_are_per_encoder(self):
        qsv = hardware.encoder_quality_args("h264_qsv", "23")
        nvenc = hardware.encoder_quality_args("h264_nvenc", "23")
        amf = hardware.encoder_quality_args("h264_amf", "22")
        x264 = hardware.encoder_quality_args("libx264", "20")

        self.assertIn("-global_quality", qsv)
        self.assertNotIn("-crf", qsv)

        self.assertIn("-cq", nvenc)
        self.assertIn("-rc", nvenc)
        self.assertNotIn("-crf", nvenc)

        self.assertIn("-qp_i", amf)
        self.assertNotIn("-crf", amf)

        self.assertIn("-crf", x264)

    def test_is_hardware_encoder(self):
        self.assertTrue(hardware.is_hardware_encoder("h264_nvenc"))
        self.assertTrue(hardware.is_hardware_encoder("h264_amf"))
        self.assertTrue(hardware.is_hardware_encoder("hevc_qsv"))
        self.assertFalse(hardware.is_hardware_encoder("libx264"))


class DrawBackendTest(unittest.TestCase):
    def test_nvidia_uses_cuda(self):
        got = hardware.pick_draw_backend("nvidia", {"openvino": True, "torch": True})
        self.assertEqual(got["backend"], "torch")
        self.assertEqual(got["device"], "cuda")

    def test_amd_uses_directml_on_windows(self):
        got = hardware.pick_draw_backend(
            "amd", {"openvino": True, "torch": True, "directml": True}, True
        )
        self.assertEqual(got["device"], "dml")

    def test_amd_without_directml_falls_to_openvino_cpu(self):
        got = hardware.pick_draw_backend("amd", {"openvino": True, "torch": True, "directml": False}, True)
        self.assertEqual(got["backend"], "openvino")
        self.assertEqual(got["device"], "CPU")

    def test_intel_uses_openvino_gpu(self):
        got = hardware.pick_draw_backend("intel", {"openvino": True, "torch": False})
        self.assertEqual(got["device"], "GPU")

    def test_no_backend(self):
        got = hardware.pick_draw_backend("unknown", {})
        self.assertIsNone(got["backend"])

    def test_openvino_device_suggestion(self):
        self.assertEqual(hardware.suggest_openvino_device("intel"), "GPU")
        self.assertEqual(hardware.suggest_openvino_device("nvidia"), "CPU")
        self.assertEqual(hardware.suggest_openvino_device("nvidia", "GPU"), "GPU")
        self.assertEqual(hardware.suggest_openvino_device("intel", "AUTO"), "GPU")


if __name__ == "__main__":
    unittest.main()
