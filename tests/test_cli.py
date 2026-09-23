"""Unit tests for the CLI commands."""

import unittest
from unittest.mock import patch, MagicMock
from modelfit import cli
from modelfit import server
from modelfit.hardware import SystemSpecs


class TestCLI(unittest.TestCase):
    @patch.object(cli, "detect_system_specs")
    def test_cmd_specs(self, mock_specs):
        mock_specs.return_value = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            has_cuda=False
        )
        args = MagicMock()
        # Ensure it runs without exception
        cli.cmd_specs(args)

    @patch.object(cli, "detect_system_specs")
    @patch.object(cli.HFHardwareClient, "search_models")
    @patch.object(cli.HFHardwareClient, "filter_and_rank_models")
    def test_cmd_search(self, mock_rank, mock_search, mock_specs):
        mock_specs.return_value = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            has_cuda=False
        )
        mock_search.return_value = []
        mock_rank.return_value = [
            {
                "model_id": "test/model",
                "target_device": "cpu",
                "memory_required_gb": 0.5,
                "downloads": 1000,
                "ranking_score": 95.0
            }
        ]

        args = MagicMock()
        args.query = "plant"
        args.task = "image-classification"
        args.precision = "fp16"
        args.limit = 5

        cli.cmd_search(args)
        mock_search.assert_called_once()
        mock_rank.assert_called_once()

    @patch.object(cli.ModelEvaluator, "benchmark_models")
    def test_cmd_benchmark(self, mock_bench):
        mock_bench.return_value = []
        args = MagicMock()
        args.models = "model-1,model-2"
        args.samples = "sample.jpg"
        args.task = "image-classification"

        cli.cmd_benchmark(args)
        mock_bench.assert_called_once()

    @patch.object(cli.EnsembleGateway, "predict_ensemble")
    def test_cmd_ensemble(self, mock_ens):
        mock_ens.return_value = []
        args = MagicMock()
        args.models = "model-1,model-2"
        args.input = "sample.jpg"
        args.strategy = "weighted_average"
        args.task = "image-classification"

        cli.cmd_ensemble(args)
        mock_ens.assert_called_once()


if __name__ == "__main__":
    unittest.main()
