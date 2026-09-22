"""Unit tests for the ModelFit Command Line Interface."""

import unittest
from unittest.mock import patch
from io import StringIO
from modelfit.cli import main


class TestCLI(unittest.TestCase):
    @patch("sys.stdout", new_callable=StringIO)
    def test_cli_specs(self, mock_stdout):
        with patch("sys.argv", ["modelfit", "specs"]):
            main()
            output = mock_stdout.getvalue()
            self.assertIn("ModelFit Hardware Profile", output)
            self.assertIn("RAM Total", output)

    @patch("sys.stdout", new_callable=StringIO)
    @patch("modelfit.hf_client.HFHardwareClient.search_models")
    def test_cli_search(self, mock_search, mock_stdout):
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
            main()
            output = mock_stdout.getvalue()
            self.assertIn("small/plant-net", output)
            self.assertIn("Found", output)


if __name__ == "__main__":
    unittest.main()
