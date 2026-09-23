"""Unit tests for code generation snippets."""

import unittest
from modelfit.code_gen import (
    generate_transformers_snippet,
    generate_ensemble_snippet,
    generate_client_snippet,
    generate_flutter_snippet,
    generate_node_snippet,
)


class TestCodeGen(unittest.TestCase):
    def test_generate_transformers_snippet(self):
        code = generate_transformers_snippet("test/model", "image-classification", target_device="gpu")
        self.assertIn('device="cuda"', code)
        self.assertIn("test/model", code)

    def test_generate_ensemble_snippet(self):
        code = generate_ensemble_snippet(["model-1", "model-2"], "image-classification", strategy="weighted_average")
        self.assertIn("EnsembleGateway", code)
        self.assertIn("model-1", code)
        self.assertIn("weighted_average", code)

    def test_generate_flutter_snippet(self):
        code = generate_flutter_snippet(host="127.0.0.1", port=7860)
        self.assertIn("class ModelFitClient", code)
        self.assertIn("http://127.0.0.1:7860", code)

    def test_generate_node_snippet(self):
        code = generate_node_snippet(host="127.0.0.1", port=7860)
        self.assertIn("GATEWAY_URL", code)
        self.assertIn("predict(", code)


if __name__ == "__main__":
    unittest.main()
