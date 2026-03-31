import sqlite3
from pathlib import Path

from config import settings

db_path = Path(settings.CHROMA_PERSIST_DIR) / "vectors.db"
try:
    conn = sqlite3.connect(str(db_path))
    res = conn.execute("SELECT metadata FROM documents LIMIT 10")
    for row in res:
        print(row[0])
except Exception as e:
    print("Error:", e)
