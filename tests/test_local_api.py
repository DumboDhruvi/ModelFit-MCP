"""Unit tests for the local micro-API server."""

import io
import json
import unittest
from unittest.mock import patch, MagicMock
from modelfit.local_api import GatewayRequestHandler
from modelfit.adapter import ModelGateway, Prediction


class MockServer:
    pass


class TestLocalAPI(unittest.TestCase):
    def setUp(self):
        self.gateway = ModelGateway.get_instance()

    def _create_handler(self, method: str, path: str, body: dict = None):
        request = MagicMock()
        client_address = ("127.0.0.1", 12345)
        server = MockServer()

        handler = GatewayRequestHandler.__new__(GatewayRequestHandler)
        handler.command = method
        handler.path = path
        handler.request = request
        handler.client_address = client_address
        handler.server = server
        handler.headers = {}
        handler.wfile = io.BytesIO()

        if body is not None:
            body_bytes = json.dumps(body).encode("utf-8")
            handler.rfile = io.BytesIO(body_bytes)
            handler.headers["Content-Length"] = str(len(body_bytes))
        else:
            handler.rfile = io.BytesIO(b"")

        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()

        return handler

    def test_get_status(self):
        self.gateway.active_model_id = "test-model"
        self.gateway.pipeline_tag = "image-classification"
        self.gateway.target_device = "cpu"

        handler = self._create_handler("GET", "/status")
        handler.do_GET()

        output = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertEqual(output["active_model"], "test-model")
        self.assertEqual(output["task"], "image-classification")

    @patch("modelfit.adapter.ModelGateway.predict")
    def test_post_predict(self, mock_predict):
        mock_predict.return_value = [Prediction(label="healthy", score=0.99)]

        handler = self._create_handler("POST", "/predict", {"input": "leaf.jpg"})
        handler.do_POST()

        output = json.loads(handler.wfile.getvalue().decode("utf-8"))
        self.assertIn("predictions", output)
        self.assertEqual(output["predictions"][0]["label"], "healthy")


if __name__ == "__main__":
    unittest.main()
