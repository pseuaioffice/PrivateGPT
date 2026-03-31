"""
model_manager.py
Service layer for Ollama model management.
Handles model discovery, download progress tracking, and status monitoring.
"""
import json
import logging
import sqlite3
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import requests
import subprocess
import sys
import os
import time
from config import settings

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    DOWNLOADING = "downloading"
    ERROR = "error"


@dataclass
class ModelInfo:
    name: str
    size: int = 0
    modified_at: str = ""
    digest: str = ""
    status: str = ModelStatus.NOT_INSTALLED.value
    progress: int = 0
    model_type: str = "chat"  # 'chat' or 'embedding'


class ModelManager:
    """
    Manages Ollama models for the RAG system.
    Tracks installed models, handles downloads, and persists configuration.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        # Read CHROMA_PERSIST_DIR from the environment at init time (not import time).
        # This ensures the path is the correct absolute installed path set by backend_entry.py,
        # not a stale relative or temp path from module load time.
        persist_dir = settings.CHROMA_PERSIST_DIR
        self._db_path = os.path.join(persist_dir, 'model_config.db')
        self._ensure_db()
        self._download_progress: Dict[str, Dict] = {}
        self._progress_callbacks: List[Callable] = []
        self._ollama_url = settings.OLLAMA_BASE_URL
    
    def _ensure_db(self):
        """Initialize the configuration database."""
        import os
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS model_config (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ollama_models (
                name TEXT PRIMARY KEY,
                size INTEGER,
                modified_at TEXT,
                digest TEXT,
                model_type TEXT,
                status TEXT
            )
        """)
        conn.commit()
        conn.close()
        logger.info("Model configuration database initialized.")
    
    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
    
    # ------------------------------------------------------------------
    # Configuration Management
    # ------------------------------------------------------------------
    
    def get_config(self, key: str, default: str = "") -> str:
        """Get a configuration value."""
        conn = self._get_conn()
        res = default
        try:
            row = conn.execute(
                "SELECT value FROM model_config WHERE key = ?", (key,)
            ).fetchone()
            if row:
                res = row[0]
        except Exception as e:
            logger.error("Failed to get config %s: %s", key, e)
        finally:
            conn.close()
        return res
    
    def set_config(self, key: str, value: str):
        """Set a configuration value."""
        conn = self._get_conn()
        try:
            conn.execute(
                """INSERT INTO model_config (key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value)
            )
            conn.commit()
            logger.info("Config updated: %s = %s", key, value)
        finally:
            conn.close()
    
    def get_active_models(self) -> Dict[str, str]:
        """Get currently active chat and embedding models."""
        return {
            "chat_model": self.get_config("chat_model", settings.CHAT_MODEL_LOCAL),
            "embedding_model": self.get_config("embedding_model", settings.EMBEDDING_MODEL_LOCAL),
            "provider": self.get_config("provider", settings.MODEL_PROVIDER),
        }
    
    def set_active_models(self, chat_model: Optional[str] = None, 
                         embedding_model: Optional[str] = None,
                         provider: Optional[str] = None):
        """Update active model configuration."""
        if chat_model:
            self.set_config("chat_model", chat_model)
            settings.CHAT_MODEL_LOCAL = chat_model
        if embedding_model:
            self.set_config("embedding_model", embedding_model)
            settings.EMBEDDING_MODEL_LOCAL = embedding_model
        if provider:
            self.set_config("provider", provider)
            settings.MODEL_PROVIDER = provider
    
    # ------------------------------------------------------------------
    # Ollama API Integration
    # ------------------------------------------------------------------
    
    def list_local_models(self) -> List[ModelInfo]:
        """List all models installed in Ollama."""
        try:
            url = f"{self._ollama_url}/api/tags"
            res = requests.get(url, timeout=5)
            res.raise_for_status()
            data = res.json()
            
            models = []
            for m in data.get("models", []):
                model = ModelInfo(
                    name=m.get("name", ""),
                    size=m.get("size", 0),
                    modified_at=m.get("modified_at", ""),
                    digest=m.get("digest", ""),
                    status=ModelStatus.INSTALLED.value
                )
                models.append(model)
            
            # Update local cache
            self._update_model_cache(models)
            return models
            
        except requests.exceptions.ConnectionError as e:
            logger.warning("Ollama not running at %s: %s", self._ollama_url, e)
            return []
        except requests.exceptions.RequestException as e:
            logger.error("Failed to list Ollama models: %s", e)
            return []
        except Exception as e:
            logger.error("Unexpected error listing Ollama models: %s", e)
            return []
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if a model is installed in Ollama."""
        try:
            # Normalize model name
            if ":" not in model_name:
                model_name = f"{model_name}:latest"
            
            url = f"{self._ollama_url}/api/tags"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            data = res.json()
            
            installed = [m.get("name", "") for m in data.get("models", [])]
            return model_name in installed
            
        except Exception as e:
            logger.error("Failed to check model existence: %s", e)
            return False
    
    def pull_model(self, model_name: str, progress_callback: Optional[Callable] = None):
        """
        Pull a model from Ollama registry with progress tracking.
        Returns a generator that yields progress updates.
        """
        try:
            url = f"{self._ollama_url}/api/pull"
            
            # Register callback if provided
            if progress_callback:
                self._progress_callbacks.append(progress_callback)
            
            # Initialize progress tracking
            self._download_progress[model_name] = {
                "status": "downloading",
                "progress": 0,
                "total": 0,
                "completed": 0
            }
            
            logger.info("Starting pull for model: %s", model_name)
            
            with requests.post(
                url, 
                json={"name": model_name, "stream": True}, 
                stream=True, 
                timeout=None
            ) as res:
                res.raise_for_status()
                
                for line in res.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            status = data.get("status", "")
                            
                            # Calculate progress
                            progress = 0
                            if "completed" in data and "total" in data:
                                completed = data.get("completed", 0)
                                total = data.get("total", 0)
                                if total > 0:
                                    progress = int((completed / total) * 100)
                            
                            # Update progress tracking
                            self._download_progress[model_name] = {
                                "status": status,
                                "progress": progress,
                                "completed": data.get("completed", 0),
                                "total": data.get("total", 0)
                            }
                            
                            # Notify callbacks
                            update = {
                                "model": model_name,
                                "status": status,
                                "progress": progress
                            }
                            
                            for callback in self._progress_callbacks:
                                try:
                                    callback(update)
                                except Exception as e:
                                    logger.warning("Progress callback error: %s", e)
                            
                            # Yield the update for streaming
                            yield update
                            
                        except json.JSONDecodeError:
                            continue
            
            # Mark as complete
            self._download_progress[model_name]["status"] = "success"
            self._download_progress[model_name]["progress"] = 100
            
            # Refresh model list
            self.list_local_models()
            
            logger.info("Successfully pulled model: %s", model_name)
            
        except Exception as e:
            logger.error("Failed to pull model %s: %s", model_name, e)
            self._download_progress[model_name] = {
                "status": "error",
                "progress": 0,
                "error": str(e)
            }
            yield {
                "model": model_name,
                "status": "error",
                "progress": 0,
                "error": str(e)
            }
        finally:
            if progress_callback:
                self._progress_callbacks.remove(progress_callback)
    
    def get_download_progress(self, model_name: str) -> Optional[Dict]:
        """Get current download progress for a model."""
        return self._download_progress.get(model_name)
    
    def _update_model_cache(self, models: List[ModelInfo]):
        """Update the local model cache in database."""
        conn = self._get_conn()
        try:
            for model in models:
                conn.execute(
                    """INSERT INTO ollama_models 
                       (name, size, modified_at, digest, status)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(name) DO UPDATE SET
                       size = excluded.size,
                       modified_at = excluded.modified_at,
                       digest = excluded.digest,
                       status = excluded.status""",
                    (model.name, model.size, model.modified_at, 
                     model.digest, model.status)
                )
            conn.commit()
        finally:
            conn.close()
    
    def get_cached_models(self) -> List[ModelInfo]:
        """Get models from local cache."""
        conn = self._get_conn()
        res = []
        try:
            rows = conn.execute(
                "SELECT name, size, modified_at, digest, model_type, status FROM ollama_models"
            ).fetchall()
            res = [
                ModelInfo(
                    name=r[0], size=r[1], modified_at=r[2],
                    digest=r[3], model_type=r[4] or "chat", status=r[5]
                )
                for r in rows
            ]
        except Exception as e:
            logger.error("Failed to get cached models: %s", e)
        finally:
            conn.close()
        return res
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a model from Ollama."""
        try:
            url = f"{self._ollama_url}/api/delete"
            res = requests.delete(url, json={"name": model_name}, timeout=30)
            res.raise_for_status()
            
            # Update cache
            conn = self._get_conn()
            try:
                conn.execute("DELETE FROM ollama_models WHERE name = ?", (model_name,))
                conn.commit()
            finally:
                conn.close()
            
            logger.info("Deleted model: %s", model_name)
            return True
            
        except Exception as e:
            logger.error("Failed to delete model %s: %s", model_name, e)
            return False
    
    def get_model_details(self, model_name: str) -> Optional[Dict]:
        """Get detailed information about a model."""
        try:
            url = f"{self._ollama_url}/api/show"
            res = requests.post(url, json={"name": model_name}, timeout=10)
            res.raise_for_status()
            return res.json()
        except Exception as e:
            logger.error("Failed to get model details for %s: %s", model_name, e)
            return None

    def is_ollama_running(self) -> bool:
        """Check if Ollama API is responsive."""
        try:
            res = requests.get(f"{self._ollama_url}/api/tags", timeout=3)
            return res.status_code == 200
        except requests.exceptions.ConnectionError:
            return False
        except requests.exceptions.RequestException:
            return False
        except Exception:
            return False


# Lazy singleton — do NOT instantiate at import time.
# backend_entry.py sets CHROMA_PERSIST_DIR *after* Python imports are done.
# Instantiating here would lock in a wrong/temp path before the env is ready.
_model_manager_instance = None

def _get_model_manager() -> 'ModelManager':
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = ModelManager()
    return _model_manager_instance

# Provide a proxy object so that existing code using `model_manager.xxx` still works
class _ModelManagerProxy:
    """Transparent proxy that defers ModelManager creation until first attribute access."""
    def __getattr__(self, name):
        return getattr(_get_model_manager(), name)

model_manager = _ModelManagerProxy()
