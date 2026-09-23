"""Unit tests for empirical model evaluator and benchmarking."""

import unittest
from unittest.mock import MagicMock, patch
from modelfit.evaluator import ModelEvaluator, BenchmarkResult
from modelfit.adapter import ModelGateway, Prediction


class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        self.gateway = ModelGateway()
        self.evaluator = ModelEvaluator(gateway=self.gateway)

    @patch("modelfit.adapter.pipeline")
    def test_benchmark_ranking(self, mock_pipeline):
        # Mock two pipelines:
        # Model 1: High confidence (0.95), fast
        pipe_fast = MagicMock()
        pipe_fast.return_value = [{"label": "healthy", "score": 0.95}]

        # Model 2: Lower confidence (0.70)
        pipe_slow = MagicMock()
        pipe_slow.return_value = [{"label": "healthy", "score": 0.70}]

        mock_pipeline.side_effect = [pipe_fast, pipe_slow]

        results = self.evaluator.benchmark_models(
            model_ids=["fast-model", "slow-model"],
            test_samples=["leaf_1.jpg", "leaf_2.jpg"],
            pipeline_tag="image-classification"
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].model_id, "fast-model")
        self.assertEqual(results[0].rank, 1)
        self.assertGreater(results[0].composite_score, results[1].composite_score)
        self.assertEqual(results[0].samples_evaluated, 2)

    def test_benchmark_empty_models(self):
        results = self.evaluator.benchmark_models([], ["sample.jpg"])
        self.assertEqual(results, [])

    @patch("modelfit.adapter.pipeline")
    def test_benchmark_error_handling(self, mock_pipeline):
        mock_pipeline.side_effect = RuntimeError("Failed to load weights")

        results = self.evaluator.benchmark_models(["broken-model"], ["sample.jpg"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].composite_score, 0.0)
        self.assertEqual(results[0].samples_evaluated, 0)


if __name__ == "__main__":
    unittest.main()
