# 🚀 Quick Start Guide: Semantic Search API

> [!IMPORTANT]
> **BEFORE ANYTHING ELSE — verify your Python version:**
> ```powershell
> python --version
> # Must print Python 3.11.x
> # If it prints 3.14, run: deactivate then .\.venv\Scripts\activate
> ```

Follow these steps to quickly start and demo the project.

## 1. Prerequisites
Ensure you have **Python 3.11** active (via the `.venv` in the project root).
If you haven't created the venv yet, refer to the [walkthrough.md](file:///C:/Users/singh/.gemini/antigravity/brain/61fd5051-4ca2-4dcb-99d6-8ddbcc81a960/walkthrough.md).

## 2. `.venv` Creation (Fix for Python 3.14)
> [!WARNING]
> ALWAYS create venv with:
> ```powershell
> conda run -n base python -m venv .venv --clear
> ```
> NEVER with just: `python -m venv .venv`
> (system default is 3.14 which breaks chromadb)

## 3. Start the API Server
Run the following command in PowerShell from the `semantic-search` directory:

```powershell
# Activate the venv
.\.venv\Scripts\activate

# Start the server
uvicorn src.api:app --port 8000 --reload
```

## 4. Quick Demo (Health Check)
Open a new PowerShell window and run:

```powershell
Invoke-WebRequest -Uri http://localhost:8000/health -UseBasicParsing
```
**Expected Response:** `{"status":"ok", "index_built":true, ...}`

## 5. Documentation & Interaction
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)  
  *Use this to interact with all API endpoints directly from your browser.*
- **Redoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## 6. Key Endpoints for Demo
- `GET /health`: Check if models and index are loaded.
- `POST /search`: Perform a semantic search query.
- `GET /visualize`: (If implemented) View embedding clusters.

---
> [!TIP]
> If the server fails to start due to `chromadb` errors, ensure your terminal is NOT using Python 3.14. Check with `python --version`.
