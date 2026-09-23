"""Unit tests for high-speed ensemble gateway and aggregators."""

import unittest
from unittest.mock import MagicMock, patch
from modelfit.ensemble import EnsembleGateway, EnsemblePrediction
from modelfit.adapter import ModelGateway, Prediction
from modelfit.hardware import SystemSpecs


class TestEnsembleGateway(unittest.TestCase):
    def setUp(self):
        self.gateway = ModelGateway()
        self.ensemble = EnsembleGateway(gateway=self.gateway)

    @patch("modelfit.adapter.pipeline")
    def test_weighted_average_aggregation(self, mock_pipeline):
        # Model A predicts: "healthy" (0.9), "blight" (0.1)
        pipe_a = MagicMock()
        pipe_a.return_value = [{"label": "healthy", "score": 0.90}, {"label": "blight", "score": 0.10}]

        # Model B predicts: "healthy" (0.8), "blight" (0.2)
        pipe_b = MagicMock()
        pipe_b.return_value = [{"label": "healthy", "score": 0.80}, {"label": "blight", "score": 0.20}]

        mock_pipeline.side_effect = [pipe_a, pipe_b]

        preds = self.ensemble.predict_ensemble(
            input_data="leaf.jpg",
            model_ids=["model-a", "model-b"],
            strategy="weighted_average"
        )

        self.assertGreaterEqual(len(preds), 1)
        self.assertEqual(preds[0].label, "healthy")
        self.assertAlmostEqual(preds[0].score, 0.85, places=2)
        self.assertEqual(preds[0].votes, 2)
        self.assertIn("model-a", preds[0].participating_models)
        self.assertIn("model-b", preds[0].participating_models)

    @patch("modelfit.adapter.pipeline")
    def test_majority_vote_aggregation(self, mock_pipeline):
        # Model A: "rust"
        pipe_a = MagicMock()
        pipe_a.return_value = [{"label": "rust", "score": 0.85}]

        # Model B: "rust"
        pipe_b = MagicMock()
        pipe_b.return_value = [{"label": "rust", "score": 0.75}]

        # Model C: "healthy"
        pipe_c = MagicMock()
        pipe_c.return_value = [{"label": "healthy", "score": 0.95}]

        mock_pipeline.side_effect = [pipe_a, pipe_b, pipe_c]

        preds = self.ensemble.predict_ensemble(
            input_data="leaf.jpg",
            model_ids=["model-a", "model-b", "model-c"],
            strategy="majority_vote"
        )

        self.assertEqual(preds[0].label, "rust")
        self.assertEqual(preds[0].votes, 2)

    @patch("modelfit.adapter.pipeline")
    def test_top_confidence_aggregation(self, mock_pipeline):
        pipe_a = MagicMock()
        pipe_a.return_value = [{"label": "mildew", "score": 0.65}]

        pipe_b = MagicMock()
        pipe_b.return_value = [{"label": "rot", "score": 0.98}]

        mock_pipeline.side_effect = [pipe_a, pipe_b]

        preds = self.ensemble.predict_ensemble(
            input_data="leaf.jpg",
            model_ids=["model-a", "model-b"],
            strategy="top_confidence"
        )

        self.assertEqual(len(preds), 1)
        self.assertEqual(preds[0].label, "rot")
        self.assertEqual(preds[0].score, 0.98)
        self.assertEqual(preds[0].participating_models, ["model-b"])

    def test_find_ensemble_candidates(self):
        specs = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            has_cuda=False
        )

        mock_hf = MagicMock()
        mock_hf.search_models.return_value = []
        mock_hf.filter_and_rank_models.return_value = [
            {"model_id": "fast-1", "memory_required_gb": 0.5},
            {"model_id": "fast-2", "memory_required_gb": 0.8},
            {"model_id": "giant-heavy", "memory_required_gb": 6.5}  # Too heavy for ensemble
        ]

        ens = EnsembleGateway(hf_client=mock_hf)
        candidates = ens.find_ensemble_candidates("plant", specs=specs, max_models=2)

        # Only fast-1 and fast-2 fit within 35% of 8GB (2.8 GB limit)
        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0]["model_id"], "fast-1")
        self.assertEqual(candidates[1]["model_id"], "fast-2")


if __name__ == "__main__":
    unittest.main()
