"""
app.py  —  Flask frontend for the RAG Chatbot
Talks to the FastAPI backend running on http://localhost:8000
"""
import os
# Suppress urllib3/chardet version warnings in all processes (incl. Flask reloader)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import warnings
warnings.filterwarnings("ignore")

import requests
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


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
        resp = requests.post(
            f"{BACKEND_URL}/chat",
            json=data,  # Pass entire payload including chat_uuid
            timeout=60,
        )
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
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400
    f = request.files["file"]
    try:
        resp = requests.post(
            f"{BACKEND_URL}/upload",
            files={"file": (f.filename, f.stream, f.mimetype)},
            timeout=120,
        )
        resp.raise_for_status()
        return jsonify(resp.json())
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "Cannot reach the RAG backend."}), 503
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/documents")
def documents():
    try:
        resp = requests.get(f"{BACKEND_URL}/documents", timeout=5)
        resp.raise_for_status()
        return jsonify(resp.json())
    except Exception as exc:
        return jsonify({"error": str(exc)}), 503


@app.route("/api/documents/<filename>", methods=["DELETE"])
def delete_document(filename):
    try:
        resp = requests.delete(f"{BACKEND_URL}/documents/{filename}", timeout=10)
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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
