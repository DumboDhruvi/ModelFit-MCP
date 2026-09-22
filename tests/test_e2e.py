"""End-to-end integration test simulating the plant detection workflow."""

import unittest
from unittest.mock import patch
from modelfit.server import handle_tool_call


class TestPlantDetectionWorkflowE2E(unittest.TestCase):
    @patch("modelfit.hf_client.HFHardwareClient.search_models")
    def test_plant_detection_workflow(self, mock_search):
        # 1. Mock Hugging Face search response
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

        # 2. Query hardware specs
        specs_res = handle_tool_call("get_hardware_specs", {})
        self.assertEqual(specs_res["status"], "success")
        self.assertIn("specs", specs_res)

        # 3. Search compatible models
        search_res = handle_tool_call(
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

        # 4. One-shot recommend and scaffold
        scaffold_res = handle_tool_call(
            "recommend_and_scaffold",
            {"query": "plant disease", "pipeline_tag": "image-classification"}
        )
        self.assertEqual(scaffold_res["status"], "success")
        self.assertIn("linkanjarad/mobilenet_v2_1.0_224", scaffold_res["integration_code"])
        self.assertIn("pipeline(", scaffold_res["integration_code"])


if __name__ == "__main__":
    unittest.main()
