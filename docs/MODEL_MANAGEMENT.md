# Model Management System Documentation

## Overview

The Model Management System provides dynamic Ollama model management for the RAG (Retrieval-Augmented Generation) system. It allows administrators to:

- **Select Chat Models**: Choose which Ollama model to use for chat responses
- **Select Embedding Models**: Choose which Ollama model to use for document embeddings
- **Download Models**: Automatically pull models from Ollama registry if not installed
- **Track Progress**: View real-time download progress with WebSocket or SSE
- **Persistent Configuration**: Model selections persist across server restarts

---

## Architecture

### Components

| Component | Description | Location |
|-----------|-------------|----------|
| **Model Manager** | Core service for model discovery, download, and status tracking | `backend/model_manager.py` |
| **Backend API** | FastAPI endpoints for model management | `backend/main.py` |
| **Frontend Proxy** | Flask endpoints that forward to backend | `frontend/app.py` |
| **Admin UI** | Model selection interface in admin dashboard | `frontend/templates/index.html` |

### Data Flow

```
User (Admin UI)
    ↓
Frontend Flask App (proxy)
    ↓
FastAPI Backend
    ↓
Model Manager Service
    ↓
Ollama API (local:11434)
```

### Persistent Storage

Model configuration is stored in SQLite at `{CHROMA_PERSIST_DIR}/model_config.db`:

- `model_config` table: key-value pairs for active models and provider
- `ollama_models` table: cache of installed Ollama models

---

## API Endpoints

### Model Discovery

#### List Installed Models
```
GET /api/ollama/models
GET /api/ollama/tags
```

Response:
```json
{
  "models": [
    {
      "name": "qwen2.5:latest",
      "size": 4937959808,
      "modified_at": "2024-01-15T10:30:00Z",
      "digest": "sha256:abc123...",
      "status": "installed"
    }
  ],
  "active_models": {
    "chat_model": "qwen2.5",
    "embedding_model": "nomic-embed-text",
    "provider": "ollama"
  }
}
```

#### Check Model Status
```
POST /api/ollama/check
```

Request:
```json
{
  "model_name": "qwen2.5"
}
```

Response:
```json
{
  "model": "qwen2.5",
  "installed": true,
  "status": "installed"
}
```

### Model Download

#### Start Model Pull (Background)
```
POST /api/ollama/pull
```

Request:
```json
{
  "model_name": "qwen2.5"
}
```

Response:
```json
{
  "message": "Started pull for qwen2.5."
}
```

Progress is streamed via WebSocket to `/ws/ollama`.

#### Stream Download Progress (SSE)
```
GET /api/ollama/pull-stream?model_name=qwen2.5
```

Server-Sent Events stream:
```
data: {"model": "qwen2.5", "status": "pulling", "progress": 25}
data: {"model": "qwen2.5", "status": "pulling", "progress": 50}
data: {"model": "qwen2.5", "status": "success", "progress": 100}
```

#### Get Download Progress
```
GET /api/ollama/progress/{model_name}
```

Response:
```json
{
  "model": "qwen2.5",
  "status": "pulling",
  "progress": 45,
  "error": null
}
```

### Model Configuration

#### Get Active Models
Included in `/api/status` response:
```json
{
  "total_vectors": 1500,
  "chat_model": "qwen2.5",
  "embedding_model": "nomic-embed-text",
  "model_provider": "ollama"
}
```

#### Update Model Settings (Persistent)
```
PATCH /api/settings/ollama?chat_model=qwen2.5&embedding_model=nomic-embed-text
```

Response:
```json
{
  "chat_model": "qwen2.5",
  "embedding_model": "nomic-embed-text",
  "provider": "ollama"
}
```

#### Switch Provider
```
PATCH /api/settings/model
```

Request:
```json
{
  "provider": "ollama",
  "model_name": "qwen2.5"
}
```

---

## Admin UI Usage

### Accessing Model Management

1. Navigate to `/admin` in your browser
2. Click "Admin Dashboard" in the sidebar
3. Scroll to the "Model Orchestration" panel

### Selecting Ollama Models

1. Ensure "Ollama Local" is selected (not "Hugging Face")
2. Enter the **Chat Model** name (e.g., `qwen2.5`, `llama3.1`)
3. Enter the **Embedding Model** name (e.g., `nomic-embed-text`)
4. Click "Apply" to save settings

### Installing Missing Models

If a model is not installed:

1. The status will show "Not Installed" with gray background
2. An "Install" button will appear
3. Click "Install" to start the download
4. Watch the progress bar fill in real-time
5. Status changes to "Installed" (green) when complete

### Real-Time Progress

- Download progress is shown via WebSocket connection
- Progress bar updates automatically
- WebSocket status shown at bottom: "Connected" (green) or "Disconnected" (red)

---

## RAG Integration

### Embedding Model Consistency

The system ensures the **same embedding model** is used for:

1. **Document Indexing**: When files are uploaded or indexed
2. **Query Embedding**: When user questions are converted to vectors
3. **Similarity Search**: When finding relevant document chunks

This is critical because different embedding models produce different vector spaces - mixing them would break retrieval accuracy.

### Model Selection in RAG Pipeline

**Document Ingestion Flow**:
```
PDF/Text File → Chunking → Embedding Model (Ollama) → Vectors → SQLite Store
```

**Query Flow**:
```
User Question → Embedding Model (Same Ollama Model) → Similarity Search → Retrieved Context
                                     ↓
                              Chat Model (Ollama) → Generated Answer
```

### Configuration Loading

On server startup (`lifespan` in `main.py`):

1. Load persistent configuration from `model_config.db`
2. Apply to `settings` object
3. RAG pipeline uses these settings automatically

---

## Configuration Reference

### Environment Variables (Default)

```env
# Default models used if no persistent config exists
CHAT_MODEL_LOCAL=qwen
EMBEDDING_MODEL_LOCAL=nomic-embed-text
MODEL_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

### Persistent Configuration

Stored in `{CHROMA_PERSIST_DIR}/model_config.db`:

| Key | Value Example | Description |
|-----|---------------|-------------|
| `provider` | `ollama` | Active provider |
| `chat_model` | `qwen2.5` | Active chat model |
| `embedding_model` | `nomic-embed-text` | Active embedding model |

### Recommended Models

**Chat Models**:
- `qwen2.5` - Alibaba's Qwen 2.5 (good balance of quality/speed)
- `llama3.1` - Meta's Llama 3.1 (strong performance)
- `mistral` - Mistral AI's model (efficient)

**Embedding Models**:
- `nomic-embed-text` - Nomic AI (optimized for retrieval)
- `all-minilm` - Lightweight, fast embeddings

---

## Troubleshooting

### Model Not Found

**Symptom**: Status shows "Not Installed" for valid model name

**Solution**:
1. Ensure Ollama is running: `ollama serve`
2. Check model name includes correct tag (e.g., `:latest`)
3. Click "Install" button to pull model

### Download Stuck

**Symptom**: Progress bar stuck, no updates

**Solution**:
1. Check WebSocket status (should show "Connected")
2. Check Ollama logs: `ollama serve` in terminal
3. Restart download if needed

### Embedding Mismatch Warning

**Symptom**: Poor retrieval quality after switching embedding models

**Solution**:
1. Clear existing vectors: Admin → "Wipe" button
2. Re-index all documents with new embedding model
3. Use same embedding model consistently

### Connection Issues

**Symptom**: "Cannot reach Ollama" errors

**Solution**:
1. Verify Ollama is running on port 11434
2. Check `OLLAMA_BASE_URL` in `.env`
3. Ensure no firewall blocking localhost:11434

---

## Security Considerations

1. **Local Only**: Ollama runs locally - models are not exposed externally
2. **No API Keys**: Unlike HuggingFace/OpenAI, no API keys required for Ollama
3. **Admin Only**: Model management is in Admin Dashboard - restrict access appropriately
4. **Model Size**: Large models (70B+) require significant RAM/VRAM - monitor resources

---

## Development

### Adding New Endpoints

Add to `backend/main.py`:

```python
@app.get("/ollama/custom-endpoint", tags=["System"])
async def custom_endpoint():
    result = model_manager.custom_method()
    return {"result": result}
```

### Frontend Integration

Add proxy in `frontend/app.py`:

```python
@app.route("/api/ollama/custom", methods=["GET"])
def custom_proxy():
    resp = requests.get(f"{BACKEND_URL}/ollama/custom-endpoint")
    return jsonify(resp.json())
```

---

## Changelog

### v1.0 - Model Management System

- Persistent configuration storage
- Real-time download progress (WebSocket + SSE)
- Model status checking
- Admin UI with progress bars
- Automatic configuration loading on startup
- Duplicate function cleanup
