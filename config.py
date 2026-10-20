"""Central config. Import this FIRST in every entry point: it sets the telemetry
env vars before CrewAI / ChromaDB are imported (zero-network requirement)."""
import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")   # ChromaDB telemetry

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"             # default: no API keys, no network
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # free, local
CHROMA_DIR = "chroma_db"
FIXED_SIZE, FIXED_OVERLAP = 300, 60                       # characters
TOP_K = 4
# Set CHUNKING=fixed|sentence after reading demo_part1.py's precision/recall numbers.
DEPLOYED_STRATEGY = os.getenv("CHUNKING", "sentence")
MAX_REQUEST_TOKENS = 1500                                 # runtime-layer budget cap
LOG_PATH = "logs/requests.jsonl"
CALIBRATION_PATH = "calibration.json"
