from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CORPUS_DIR = DATA_DIR / "corpus"
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
#PDF_URL = "https://www.cms.gov/files/document/2026-ncci-medicare-policy-manual-all-chapters.pdf"

def ensure_dirs():
    for path in (DATA_DIR, CHROMA_DIR, CORPUS_DIR):
        path.mkdir(parents=True, exist_ok=True)
