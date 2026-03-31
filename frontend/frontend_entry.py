"""
frontend_entry.py — Entry point for the PyInstaller-compiled frontend.
Adds the correct paths and starts Flask.
"""
import sys
import os

# When frozen, set up paths correctly
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    os.chdir(os.path.dirname(sys.executable))
    
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)
    
    exe_dir = os.path.dirname(sys.executable)
    if exe_dir not in sys.path:
        sys.path.insert(0, exe_dir)

    # Set template folder for Flask (templates are bundled in _MEIPASS)
    os.environ.setdefault("FLASK_TEMPLATE_DIR", os.path.join(BASE_DIR, "templates"))

# Suppress warnings
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import warnings
warnings.filterwarnings("ignore")

import requests
from flask import Flask, render_template, request, jsonify

# Determine template folder
if getattr(sys, 'frozen', False):
    template_dir = os.path.join(sys._MEIPASS, "templates") if hasattr(sys, '_MEIPASS') else os.path.join(os.path.dirname(sys.executable), "templates")
    app = Flask(__name__, template_folder=template_dir)
else:
    app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ── Pages ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("landing.html")

@app.route("/admin")
def admin():
    return render_template("index.html")

@app.route("/user")
def user():
    return render_template("user.html")


# ── API proxies ────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question cannot be empty."}), 400
    try:
        resp = requests.post(f"{BACKEND_URL}/chat", json=data, timeout=300)
        if not resp.ok:
            try:
                err_detail = resp.json().get("detail", resp.text)
            except Exception:
                err_detail = resp.text
            return jsonify({"error": err_detail}), resp.status_code
        return jsonify(resp.json())
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach the RAG backend. Is it running on port 8000?"}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/status")
def status():
    try:
        resp = requests.get(f"{BACKEND_URL}/status", timeout=5)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/upload", methods=["POST"])
def upload():
    uploads = [f for f in request.files.getlist("files") if f and f.filename]
    if not uploads and "file" in request.files:
        legacy_file = request.files["file"]
        if legacy_file and legacy_file.filename:
            uploads.append(legacy_file)

    if not uploads:
        return jsonify({"error": "No file provided."}), 400

    request_files = [
        ("files", (uploaded.filename, uploaded.stream, uploaded.mimetype or "application/octet-stream"))
        for uploaded in uploads
    ]

    try:
        resp = requests.post(
            f"{BACKEND_URL}/upload",
            files=request_files,
            timeout=600,
        )
        try:
            payload = resp.json()
        except Exception:
            payload = {"error": resp.text}
        if not resp.ok:
            return jsonify(payload), resp.status_code
        return jsonify(payload)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach the RAG backend."}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/documents")
def documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        resp.raise_for_status()
        response = jsonify(resp.json())
        response.headers["Cache-Control"] = "no-store"
        return response
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


import urllib.parse

@app.route("/api/documents/<path:filename>", methods=["DELETE"])
def delete_document(filename):
    try:
        safe_filename = urllib.parse.quote(filename)
        resp = requests.delete(f"{BACKEND_URL}/documents/{safe_filename}", timeout=10)
        if not resp.ok:
            try:
                err = resp.json().get("detail", resp.text)
            except Exception:
                err = resp.text
            return jsonify({"error": err}), resp.status_code
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── Chat history (PostgreSQL) ──────────────────────────────────────────────
@app.route("/api/db-status")
def db_status():
    try:
        resp = requests.get(f"{BACKEND_URL}/db-status", timeout=3)
        return jsonify(resp.json())
    except Exception:
        return jsonify({"db_enabled": False})


@app.route("/api/chats")
def list_chats():
    try:
        resp = requests.get(f"{BACKEND_URL}/chats", timeout=5)
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"db_enabled": False, "sessions": [], "error": str(exc)})


@app.route("/api/chats", methods=["POST"])
def create_chat():
    try:
        resp = requests.post(
            f"{BACKEND_URL}/chats",
            json=request.get_json() or {},
            timeout=5,
        )
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/chats/<chat_uuid>/messages")
def get_chat_messages(chat_uuid):
    try:
        resp = requests.get(f"{BACKEND_URL}/chats/{chat_uuid}/messages", timeout=5)
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"db_enabled": False, "messages": [], "error": str(exc)})


@app.route("/api/chats/<chat_uuid>", methods=["DELETE"])
def delete_chat(chat_uuid):
    try:
        resp = requests.delete(f"{BACKEND_URL}/chats/{chat_uuid}", timeout=5)
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/settings/model", methods=["PATCH"])
def update_model_settings():
    try:
        resp = requests.patch(
            f"{BACKEND_URL}/settings/model",
            json=request.get_json() or {},
            timeout=5,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ollama/pull", methods=["POST"])
def pull_ollama_model():
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ollama/pull",
            json=request.get_json() or {},
            timeout=10,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ollama/models")
def list_ollama_models():
    try:
        resp = requests.get(f"{BACKEND_URL}/ollama/models", timeout=10)
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/settings/ollama", methods=["PATCH"])
def update_ollama_settings():
    try:
        resp = requests.patch(
            f"{BACKEND_URL}/settings/ollama",
            params=request.args,
            timeout=5,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ollama/check", methods=["POST"])
def check_ollama_model():
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ollama/check",
            json=request.get_json() or {},
            timeout=10,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/ollama/progress/<model_name>")
def get_model_progress(model_name):
    try:
        resp = requests.get(
            f"{BACKEND_URL}/ollama/progress/{model_name}",
            timeout=5,
        )
        return jsonify(resp.json()), resp.status_code
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "False").lower() == "true"
    print(f"Starting MyAIAssistant Frontend on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
