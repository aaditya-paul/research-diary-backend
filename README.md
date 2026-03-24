# Research Diary - Backend

FastAPI-based REST API for the Research Diary application.

## Tech Stack

- **FastAPI** - Web framework
- **SQLAlchemy** - ORM
- **ChromaDB** - Vector database for semantic search
- **sentence-transformers** - Text embeddings

## Setup

1. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Install dependencies:
   ```bash   pip install -r requirements.txt
   ```

3. Run the server:
   ```bash
   uvicorn app.main:app --reload
   ```

The API runs at `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

## Database

- SQLite database: `research_diary.db`
- ChromaDB data: `chroma_data/`
