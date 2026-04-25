#!/usr/bin/env python3
"""Minimal HTTP server for the CoT Faithfulness Dashboard."""

import http.server
import os

PORT = 8000
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == '__main__':
    with http.server.HTTPServer(('', PORT), Handler) as httpd:
        print(f"\n  🧠 CoT Faithfulness Dashboard")
        print(f"  → http://localhost:{PORT}\n")
        httpd.serve_forever()
