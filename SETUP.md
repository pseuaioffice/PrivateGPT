# KnowledgeAI - Setup Guide

This document explains how to set up the KnowledgeAI RAG application on a new system.

---

## 1. Prerequisites

Ensure the following are installed on the system:
- **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
- **Git**: [Download Git](https://git-scm.com/downloads)
- **PostgreSQL (Optional)**: If you want to enable chat history persistence.

---

## 2. Clone the Repository

```bash
git clone https://github.com/pseuaioffice/PrivateGPT.git
cd PrivateGPT
```

---

## 3. Backend Setup

1. **Navigate to the backend folder:**
   ```bash
   cd backend
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the `backend/` directory by copying the example provided below.
   
   > [!IMPORTANT]
   > You **must** provide your own `HUGGINGFACE_TOKEN`. Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

   **Example `.env`:**
   ```env
   HUGGINGFACE_TOKEN="your_hf_token_here"
   HUGGINGFACE_BASE_URL=https://router.huggingface.co/hf-inference/v1
   CHAT_MODEL=Qwen/Qwen2.5-7B-Instruct
   EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
   DATABASE_URL=postgresql://user:password@localhost:5432/dbname  # Optional
   ```

5. **Start the Backend:**
   ```bash
   uvicorn main:app --reload
   ```

---

## 4. Frontend Setup

1. **Open a new terminal and navigate to the frontend folder:**
   ```bash
   cd PrivateGPT/frontend
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the Frontend:**
   ```bash
   python app.py
   ```

---

## 5. Usage

1. Open your browser to **http://127.0.0.1:5000**.
2. Select your persona (Admin or Regular User).
3. Upload documents via the sidebar to index them.
4. Start chatting!

---

## Troubleshooting

- **Error: Failed to decrypt saved password (pgAdmin):** If you hit this error in pgAdmin when setting up PostgreSQL, reset the pgAdmin configuration by renaming `%appdata%/pgadmin/pgadmin4.db` to `pgadmin4.db.bak`.
- **API Error code 410:** Ensure your `HUGGINGFACE_BASE_URL` is set to `https://router.huggingface.co/hf-inference/v1`.
