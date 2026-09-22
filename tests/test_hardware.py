"""Unit tests for hardware profiling and memory calculation."""

import unittest
from modelfit.hardware import (
    SystemSpecs,
    estimate_required_memory_gb,
    can_model_run,
    detect_system_specs,
)


class TestHardware(unittest.TestCase):
    def test_estimate_required_memory_fp16(self):
        # 1 Billion params @ FP16 (2 bytes) = ~1.86 GB * 1.25 = ~2.33 GB
        mem_fp16 = estimate_required_memory_gb(1.0, precision="fp16")
        self.assertAlmostEqual(mem_fp16, 2.33, delta=0.1)

    def test_estimate_required_memory_int4(self):
        # 7 Billion params @ INT4 (0.5 bytes) = ~3.26 GB * 1.25 = ~4.07 GB
        mem_int4 = estimate_required_memory_gb(7.0, precision="int4")
        self.assertAlmostEqual(mem_int4, 4.07, delta=0.2)

    def test_can_model_run_gpu_target(self):
        specs = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=32.0,
            ram_available_gb=20.0,
            gpu_name="NVIDIA RTX 4090",
            vram_total_gb=24.0,
            vram_available_gb=20.0,
            has_cuda=True
        )
        res = can_model_run(0.5, specs, precision="fp16")
        self.assertTrue(res["can_run"])
        self.assertEqual(res["target"], "gpu")

    def test_can_model_run_cpu_fallback(self):
        specs = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=12.0,
            has_cuda=False
        )
        res = can_model_run(0.5, specs, precision="fp16")
        self.assertTrue(res["can_run"])
        self.assertEqual(res["target"], "cpu")

    def test_can_model_run_oom_prevention(self):
        specs = SystemSpecs(
            os_name="Linux",
            cpu_count=2,
            ram_total_gb=4.0,
            ram_available_gb=1.5,
            has_cuda=False
        )
        # 7B model in FP16 needs ~16 GB -> Must be rejected
        res = can_model_run(7.0, specs, precision="fp16")
        self.assertFalse(res["can_run"])
        self.assertEqual(res["target"], "none")
        self.assertIn("requires", res["reason"].lower())

    def test_detect_system_specs_live(self):
        specs = detect_system_specs()
        self.assertIsNotNone(specs.os_name)
        self.assertGreater(specs.cpu_count, 0)
        self.assertGreater(specs.ram_total_gb, 0)


if __name__ == "__main__":
    unittest.main()
