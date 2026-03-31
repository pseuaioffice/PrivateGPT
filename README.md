# MyAIAssistant

MyAIAssistant is a local-first RAG assistant for private documents. It ships
as a FastAPI backend + Flask frontend, and includes a Windows desktop launcher
and installer via PyInstaller + Inno Setup.

## Quick Start (Dev)

Backend:
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open: http://127.0.0.1:5000

## Supported Document Types

PDF, DOCX, TXT/MD/CSV, XLS/XLSX.

## Packaging (Windows)

Build the desktop bundle:
```bash
python packaging\build.py
```

Compile the installer (requires Inno Setup):
```bash
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
```

The installer will be created under `packaging\installer_output`.

## Configuration

Runtime config is stored in `backend\.env`. The installer writes this file
from defaults in `packaging\runtime_config.py`, and can override values during
setup.

