"""Unit tests for ModelGateway loading, predicting, and hot-swapping."""

import unittest
from unittest.mock import MagicMock, patch
from modelfit.adapter import ModelGateway, Prediction


class TestModelGateway(unittest.TestCase):
    def setUp(self):
        self.gateway = ModelGateway()

    @patch("transformers.pipeline")
    def test_load_and_predict(self, mock_pipeline):
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            {"label": "Apple___Black_rot", "score": 0.9821},
            {"label": "Apple___healthy", "score": 0.0125}
        ]
        mock_pipeline.return_value = mock_pipe

        res = self.gateway.load_model("mock-plant-model", pipeline_tag="image-classification")
        self.assertEqual(res["status"], "loaded")
        self.assertEqual(self.gateway.active_model_id, "mock-plant-model")

        preds = self.gateway.predict("sample_leaf.jpg")
        self.assertEqual(len(preds), 2)
        self.assertIsInstance(preds[0], Prediction)
        self.assertEqual(preds[0].label, "Apple___Black_rot")
        self.assertAlmostEqual(preds[0].score, 0.9821)

    @patch("transformers.pipeline")
    def test_hot_swapping_frees_memory(self, mock_pipeline):
        mock_pipe_1 = MagicMock()
        mock_pipe_2 = MagicMock()
        mock_pipeline.side_effect = [mock_pipe_1, mock_pipe_2]

        # Load first model
        self.gateway.load_model("model-v1")
        self.assertEqual(self.gateway.active_model_id, "model-v1")

        # Hot-swap to second model
        self.gateway.load_model("model-v2")
        self.assertEqual(self.gateway.active_model_id, "model-v2")
        self.assertIs(self.gateway.pipeline, mock_pipe_2)


if __name__ == "__main__":
    unittest.main()
