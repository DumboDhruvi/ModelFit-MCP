"""Unit tests for Hugging Face client querying and filtering."""

import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
from modelfit.hf_client import HFHardwareClient
from modelfit.hardware import SystemSpecs


class TestHFClient(unittest.TestCase):
    def setUp(self):
        self.client = HFHardwareClient()
        self.specs = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            gpu_name="NVIDIA RTX 3060",
            vram_total_gb=12.0,
            vram_available_gb=8.0,
            has_cuda=True
        )

    @patch("urllib.request.urlopen")
    def test_search_models_mocked(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = b'[{"id": "test/model-1", "downloads": 1000, "likes": 50}]'
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        models = self.client.search_models("test-query", pipeline_tag="image-classification")
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["id"], "test/model-1")

    def test_filter_and_rank_models(self):
        candidate_models = [
            {
                "id": "small/plant-net",
                "downloads": 10000,
                "likes": 100,
                "pipeline_tag": "image-classification",
                "tags": ["params:25M"]
            },
            {
                "id": "huge/super-plant-90b",
                "downloads": 500,
                "likes": 10,
                "pipeline_tag": "image-classification",
                "tags": ["params:90B"]
            }
        ]

        compatible = self.client.filter_and_rank_models(candidate_models, self.specs)
        # The 90B model must be filtered out due to 8GB VRAM limit
        self.assertEqual(len(compatible), 1)
        self.assertEqual(compatible[0]["model_id"], "small/plant-net")
        self.assertEqual(compatible[0]["target_device"], "gpu")


if __name__ == "__main__":
    unittest.main()
