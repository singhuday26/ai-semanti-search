import uvicorn
from src.api import app

# Justification:
# - Keeps application entrypoint separate from API logic
# - Matches Docker CMD invocation
# - Avoids circular imports

if __name__ == "__main__":
    uvicorn.run(
        "src.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
