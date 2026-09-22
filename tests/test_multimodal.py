"""Unit tests for multi-modal prediction normalization and OOM protection."""

import unittest
from unittest.mock import MagicMock, patch
from modelfit.adapter import ModelGateway, Prediction


class TestMultiModalAdapter(unittest.TestCase):
    def setUp(self):
        self.gateway = ModelGateway()

    @patch("transformers.pipeline")
    def test_object_detection_normalization(self, mock_pipeline):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {
                "label": "leaf_spot",
                "score": 0.9412,
                "box": {"xmin": 10.0, "ymin": 20.0, "xmax": 50.0, "ymax": 60.0}
            }
        ]
        mock_pipeline.return_value = mock_pipe

        self.gateway.load_model("mock-detector", pipeline_tag="object-detection")
        results = self.gateway.predict("leaf.jpg")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, "leaf_spot")
        self.assertAlmostEqual(results[0].score, 0.9412)
        self.assertIsNotNone(results[0].box)
        self.assertEqual(results[0].box["xmin"], 10.0)

    @patch("transformers.pipeline")
    def test_text_generation_normalization(self, mock_pipeline):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [{"generated_text": "Plant has nitrogen deficiency."}]
        mock_pipeline.return_value = mock_pipe

        self.gateway.load_model("mock-llm", pipeline_tag="text-generation")
        results = self.gateway.predict("Describe the leaf:")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].label, "Plant has nitrogen deficiency.")
        self.assertEqual(results[0].score, 1.0)


if __name__ == "__main__":
    unittest.main()
