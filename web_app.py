"""Localhost web interface for IrisLens AI (stdlib HTTP server + NCNN)."""

from __future__ import annotations

import base64
import hmac
import json
import os
import secrets
import socket
import time
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("IRISLENS_DATA_DIR", BASE_DIR / "doctor_data")).expanduser().resolve()
IMAGE_DIR = DATA_DIR / "captures"
USERNAME = os.environ.get("IRISLENS_USERNAME", "")
PASSWORD = os.environ.get("IRISLENS_PASSWORD", "")
SESSION_TTL_SECONDS = 8 * 60 * 60
sessions: dict[str, float] = {}


def load_examinations() -> list[dict]:
    index = DATA_DIR / "examinations.json"
    if not index.exists():
        return []
    try:
        payload = json.loads(index.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def save_examinations(items: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp = DATA_DIR / "examinations.json.tmp"
    temp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(DATA_DIR / "examinations.json")


class IrisLensHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR / "web"), **kwargs)

    def _json(self, status: int, payload: dict, headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def _has_session(self) -> bool:
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        morsel = cookie.get("iris_session")
        if morsel is None:
            return False
        expires_at = sessions.get(morsel.value, 0)
        if expires_at <= time.time():
            sessions.pop(morsel.value, None)
            return False
        return True

    def _require_login(self, path: str) -> bool:
        public_paths = {"/login", "/login.html", "/login.css", "/login.js", "/api/login"}
        if path in public_paths or (path == "/api/examinations" and self.command == "POST"):
            return True
        if self._has_session():
            return True
        if path.startswith("/api/") or path.startswith("/captures/"):
            self._json(401, {"error": "Please log in to continue."})
        else:
            self.send_response(302)
            self.send_header("Location", "/login")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
        return False

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if not self._require_login(path):
            return
        if path == "/login":
            self.path = "/login.html"
            super().do_GET()
            return
        if path == "/api/status":
            self._json(200, {
                "service": "ready",
            })
            return
        if path == "/api/examinations":
            self._json(200, {"examinations": sorted(load_examinations(), key=lambda item: item["created_at"], reverse=True)})
            return
        if path.startswith("/api/examinations/"):
            examination_id = path.rsplit("/", 1)[-1]
            item = next((entry for entry in load_examinations() if entry["id"] == examination_id), None)
            if item is None:
                self._json(404, {"error": "Examination not found."})
            else:
                self._json(200, item)
            return
        if path.startswith("/captures/"):
            filename = Path(path).name
            image_path = IMAGE_DIR / filename
            if not filename.endswith(".jpg") or not image_path.is_file():
                self._json(404, {"error": "Image not found."})
                return
            image = image_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(image)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(image)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/login":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length)) if 0 < length < 20_000 else {}
                username = str(payload.get("username", ""))
                password = str(payload.get("password", ""))
                if not hmac.compare_digest(username, USERNAME) or not hmac.compare_digest(password, PASSWORD):
                    self._json(401, {"error": "Username or password is wrong."})
                    return
                token = secrets.token_urlsafe(32)
                sessions[token] = time.time() + SESSION_TTL_SECONDS
                self._json(200, {"ok": True}, {
                    "Set-Cookie": f"iris_session={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age={SESSION_TTL_SECONDS}"
                })
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "Invalid login data."})
            return
        if path == "/api/logout":
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            morsel = cookie.get("iris_session")
            if morsel:
                sessions.pop(morsel.value, None)
            self._json(200, {"ok": True}, {
                "Set-Cookie": "iris_session=; Path=/; HttpOnly; SameSite=Strict; Max-Age=0"
            })
            return
        if not self._require_login(path):
            return
        if path != "/api/examinations":
            self._json(404, {"error": "Endpoint not found."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 20_000_000:
                self._json(400, {"error": "Invalid examination size."})
                return
            payload = json.loads(self.rfile.read(length))
            patient = payload.get("patient") or {}
            prediction = payload.get("prediction") or {}
            image_data = payload.get("image", "")
            if "," in image_data:
                image_data = image_data.split(",", 1)[1]
            image_bytes = base64.b64decode(image_data, validate=True)
            frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            required = ("id", "name", "age", "gender")
            if frame is None or any(not str(patient.get(key, "")).strip() for key in required):
                self._json(400, {"error": "Patient information or captured image is incomplete."})
                return
            examination_id = uuid.uuid4().hex
            created_at = datetime.now().astimezone().isoformat(timespec="seconds")
            item = {
                "id": examination_id,
                "created_at": created_at,
                "patient": {key: str(patient[key]).strip() for key in required},
                "prediction": {
                    "class_name": str(prediction.get("class_name", "Not detected")),
                    "label": str(prediction.get("label", "Not detected")),
                    "confidence": float(prediction.get("confidence", 0)),
                    "object_count": int(prediction.get("object_count", 0)),
                },
                "image_url": f"/captures/{examination_id}.jpg",
            }
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)
            (IMAGE_DIR / f"{examination_id}.jpg").write_bytes(image_bytes)
            records = load_examinations()
            records.append(item)
            save_examinations(records)
            self._json(201, {"id": examination_id, "created_at": created_at})
        except (ValueError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"Invalid request: {exc}"})
        except Exception as exc:
            self._json(500, {"error": f"Inference failed: {exc}"})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if not self._require_login(path):
            return
        prefix = "/api/examinations/"
        examination_id = path[len(prefix):] if path.startswith(prefix) else ""
        if len(examination_id) != 32 or any(char not in "0123456789abcdef" for char in examination_id):
            self._json(404, {"error": "Examination not found."})
            return

        records = load_examinations()
        remaining = [item for item in records if item.get("id") != examination_id]
        if len(remaining) == len(records):
            self._json(404, {"error": "Examination not found."})
            return

        try:
            save_examinations(remaining)
            (IMAGE_DIR / f"{examination_id}.jpg").unlink(missing_ok=True)
            self._json(200, {"ok": True, "id": examination_id})
        except OSError as exc:
            self._json(500, {"error": f"Could not delete examination: {exc}"})


def main() -> None:
    if not USERNAME or not PASSWORD:
        raise SystemExit(
            "Set IRISLENS_USERNAME and IRISLENS_PASSWORD environment variables before starting the web app."
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    host = os.environ.get("IRISLENS_HOST", "0.0.0.0")
    server = ThreadingHTTPServer((host, int(os.environ.get("IRISLENS_PORT", "8000"))), IrisLensHandler)
    print(f"IrisLens AI local server: http://127.0.0.1:{server.server_port}")
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("192.0.2.1", 80))
            wifi_ip = sock.getsockname()[0]
        print(f"Access from a device on the same Wi-Fi network: http://{wifi_ip}:{server.server_port}")
    except OSError:
        print(f"Server listening on {host}:{server.server_port}; check the laptop Wi-Fi IP address for network access.")
    print("Portal login credentials are read from IRISLENS_USERNAME and IRISLENS_PASSWORD.")
    print("Press Ctrl+C to stop the server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
