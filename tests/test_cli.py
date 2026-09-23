"""Unit tests for the ModelFit Command Line Interface."""

import unittest
from unittest.mock import patch
from io import StringIO
from modelfit import cli
from modelfit.hardware import SystemSpecs


class TestCLI(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_cli_specs(self, mock_stdout):
        with patch("sys.argv", ["modelfit", "specs"]):
            cli.main()
            output = mock_stdout.getvalue()
            self.assertIn("ModelFit Hardware Profile", output)
            self.assertIn("RAM Total", output)

    @patch("sys.stdout", new_callable=StringIO)
    @patch.object(cli, "detect_system_specs")
    @patch("modelfit.hf_client.HFHardwareClient.search_models")
    def test_cli_search(self, mock_search, mock_specs, mock_stdout):
        mock_specs.return_value = SystemSpecs(
            os_name="Linux",
            cpu_count=8,
            ram_total_gb=16.0,
            ram_available_gb=8.0,
            has_cuda=False
        )
        mock_search.return_value = [
            {
                "id": "small/plant-net",
                "downloads": 5000,
                "likes": 20,
                "pipeline_tag": "image-classification",
                "tags": ["params:10M"]
            }
        ]
        with patch("sys.argv", ["modelfit", "search", "plant"]):
            cli.main()
            output = mock_stdout.getvalue()
            self.assertIn("small/plant-net", output)
            self.assertIn("Found", output)


if __name__ == "__main__":
    unittest.main()
