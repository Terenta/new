#!/usr/bin/env python3
"""Serve an explicit file allow-list behind a one-time unguessable URL token."""

from __future__ import annotations

import argparse
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--allow", action="append", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--max-downloads", type=int, default=1)
    parser.add_argument("--idle-timeout", type=int, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", args.token):
        raise SystemExit("token must contain 32-128 URL-safe characters")
    if args.max_downloads < 1 or args.idle_timeout < 1:
        raise SystemExit("max-downloads and idle-timeout must be positive")

    root = args.root.expanduser().resolve(strict=True)
    allowed: dict[str, Path] = {}
    for name in args.allow:
        if name != Path(name).name:
            raise SystemExit(f"allow entries must be basenames: {name}")
        path = (root / name).resolve(strict=True)
        if path.parent != root or not path.is_file():
            raise SystemExit(f"refusing file outside root: {name}")
        allowed[name] = path

    prefix = f"/download/{args.token}/"
    state = {"downloads": 0}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            request_path = unquote(urlsplit(self.path).path)
            if not request_path.startswith(prefix):
                self.send_error(404)
                return
            name = request_path[len(prefix) :]
            path = allowed.get(name)
            if path is None:
                self.send_error(404)
                return
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
            state["downloads"] += 1
            if state["downloads"] >= args.max_downloads:
                threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, fmt: str, *values: object) -> None:
            print(f"{self.client_address[0]} {fmt % values}", flush=True)

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    timer = threading.Timer(args.idle_timeout, server.shutdown)
    timer.daemon = True
    timer.start()
    try:
        print(
            f"one-time transfer ready on {args.bind}:{args.port}; "
            f"files={len(allowed)} max_downloads={args.max_downloads}",
            flush=True,
        )
        server.serve_forever()
    finally:
        timer.cancel()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
