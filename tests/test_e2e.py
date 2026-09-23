"""End-to-end integration test simulating the plant detection workflow."""

import unittest
from unittest.mock import patch
from modelfit import server
from modelfit.hardware import SystemSpecs


class TestPlantDetectionWorkflowE2E(unittest.TestCase):
    @patch.object(server, "detect_system_specs")
    @patch.object(server.hf_client, "search_models")
    def test_plant_detection_workflow(self, mock_search, mock_specs):
        # 1. Deterministic system specs (8GB RAM, CPU fallback)
        mock_specs.return_value = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            has_cuda=False
        )

        # 2. Mock Hugging Face search response
        mock_search.return_value = [
            {
                "id": "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification",
                "pipeline_tag": "image-classification",
                "downloads": 15000,
                "likes": 45,
                "tags": ["params:3.5M", "image-classification"]
            },
            {
                "id": "huge-giant/giant-plant-model-100b",
                "pipeline_tag": "image-classification",
                "downloads": 500,
                "likes": 2,
                "tags": ["params:100B", "image-classification"]
            }
        ]

        # 3. Query hardware specs
        specs_res = server.handle_tool_call("get_hardware_specs", {})
        self.assertEqual(specs_res["status"], "success")
        self.assertIn("specs", specs_res)

        # 4. Search compatible models
        search_res = server.handle_tool_call(
            "search_compatible_models",
            {"query": "plant disease", "pipeline_tag": "image-classification"}
        )
        self.assertEqual(search_res["status"], "success")
        # Ensure the 100B giant model was pruned, keeping the lightweight model
        self.assertEqual(len(search_res["models"]), 1)
        self.assertEqual(
            search_res["models"][0]["model_id"],
            "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
        )

        # 5. One-shot recommend and scaffold
        scaffold_res = server.handle_tool_call(
            "recommend_and_scaffold",
            {"query": "plant disease", "pipeline_tag": "image-classification"}
        )
        self.assertEqual(scaffold_res["status"], "success")
        self.assertIn("linkanjarad/mobilenet_v2_1.0_224", scaffold_res["integration_code"])
        self.assertIn("pipeline(", scaffold_res["integration_code"])


if __name__ == "__main__":
    unittest.main()
