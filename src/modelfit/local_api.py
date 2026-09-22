"""Lightweight local HTTP API endpoint for cross-language integration."""

import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from modelfit.adapter import gateway


class GatewayRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/status":
            self._send_json(200, {
                "active_model": gateway.active_model_id,
                "task": gateway.pipeline_tag,
                "device": gateway.target_device
            })
        else:
            self._send_json(404, {"error": "Not found"})

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        payload = json.loads(body.decode("utf-8")) if body else {}

        if self.path == "/predict":
            input_data = payload.get("input")
            if not input_data:
                self._send_json(400, {"error": "Missing 'input' field"})
                return
            try:
                predictions = gateway.predict(input_data)
                self._send_json(200, {
                    "predictions": [p.to_dict() for p in predictions]
                })
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.path == "/swap":
            model_id = payload.get("model_id")
            task = payload.get("task", "image-classification")
            if not model_id:
                self._send_json(400, {"error": "Missing 'model_id'"})
                return
            try:
                res = gateway.load_model(model_id, pipeline_tag=task)
                self._send_json(200, res)
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(404, {"error": "Endpoint not found"})

    def log_message(self, format, *args):
        # Suppress noisy standard HTTP logs during daemon execution
        pass


def run_api_server(host: str = "127.0.0.1", port: int = 7860):
    server = HTTPServer((host, port), GatewayRequestHandler)
    print(f"ModelFit Local Gateway running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run_api_server()
