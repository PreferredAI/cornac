# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""CLI entry point for model serving.
"""

import argparse
import sys
import json
import pickle
import http.server
import socketserver


class ModelRequestHandler(http.server.BaseHTTPRequestHandler):
    def _set_response(self, status_code=200, content_type="application/json"):
        self.send_response(status_code)
        self.send_header("Content-type", content_type)
        self.end_headers()

    def do_GET(self):
        if self.path == "/":
            self._set_response()
            response_data = {"message": "Cornac model serving."}
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        if self.path == "/recommend":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length).decode("utf-8")
            post_data = json.loads(post_data)

            # TODO: input validation
            user_id = str(post_data["uid"])
            k = -1 if "k" not in post_data else int(post_data["k"])
            remove_seen = (
                False
                if "remove_seen" not in post_data
                else bool(post_data["remove_seen"])
            )

            response_data = {
                "recommendations": self.server.model.recommend(
                    user_id=user_id,
                    k=k,
                    remove_seen=remove_seen,
                    train_set=self.server.train_set,
                ),
                "data_received": post_data,
            }

            self._set_response()
            self.wfile.write(json.dumps(response_data).encode())
        else:
            self.send_error(404, "Endpoint not found")


def import_model_class(model_class):
    components = model_class.split(".")
    mod = __import__(".".join(components[:-1]), fromlist=[components[-1]])
    klass = getattr(mod, components[-1])
    return klass


def parse_args():
    parser = argparse.ArgumentParser(description="Cornac model serving")
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to directory where the model was saved",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="cornac.models.Recommender",
        help="Cornac class of the model which is being deployed",
    )
    parser.add_argument(
        "--train_set",
        type=str,
        default=None,
        help="Path to pickled file of the train_set (used to remove seen items)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Service port",
    )

    return parser.parse_args(sys.argv[1:])


def main():
    args = parse_args()

    # Load model/train_set if provided
    httpd = socketserver.TCPServer(("", args.port), ModelRequestHandler)
    httpd.model = import_model_class(args.model_class).load(args.model_dir)
    httpd.train_set = None
    if args.train_set is not None:
        with open(args.train_set, "rb") as f:
            httpd.train_set = pickle.load(f)

    # Start service
    try:
        print(f"Serving {httpd.model.name} at port {args.port}")
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    print("Server stopped.")


if __name__ == "__main__":
    main()
