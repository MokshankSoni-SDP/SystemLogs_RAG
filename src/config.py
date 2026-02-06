"""
Configuration module for RAG System.
Loads environment variables and defines system constants.
"""
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_KEY_2 = os.getenv("GROQ_API_KEY_2", "")
GROQ_API_KEY_3 = os.getenv("GROQ_API_KEY_3", "")

# List of available keys for rotation
GROQ_API_KEYS = [k for k in [GROQ_API_KEY, GROQ_API_KEY_2, GROQ_API_KEY_3] if k and k.strip()]

# Model Configuration
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
LLM_MODEL = "llama-3.3-70b-versatile"

# Preprocessing Configuration
IP_PATTERN = re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b')
PORT_PATTERN = re.compile(r':(\d{2,5})\b')
PID_PATTERN = re.compile(r'\[(\d+)\]|\(pid:?\s*(\d+)\)')
UUID_PATTERN = re.compile(r'\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b', re.IGNORECASE)
MEMORY_ADDR_PATTERN = re.compile(r'0x[0-9a-fA-F]{6,16}')

# Chunking Boundaries
CHUNK_SIZE_LINUX = 40  # max lines per chunk for Linux/macOS
CHUNK_SIZE_WINDOWS = 50  # max lines per chunk for Windows
TIME_GAP_THRESHOLD_SECONDS = 120  # max time gap before creating new chunk

# Legacy chunking (kept for backward compatibility during migration)
CHUNK_SIZE = 30  # number of log lines per chunk
OVERLAP = 5      # number of overlapping lines between chunks

# Query Classification
COMPLEX_QUERY_KEYWORDS = [
    # Aggregation / counting
    "how many", "count", "number of", "total",

    # Time & temporal reasoning
    "over time", "across time", "trend", "during", "between",
    "before", "after", "since", "until", "earlier", "later",

    # Frequency / repetition
    "repeated", "multiple", "frequent", "recurring",
    "sporadic", "intermittent", "burst", "spike",

    # Scope / coverage
    "all", "any", "every", "entire", "whole", "throughout",

    # Comparison / change
    "compare", "correlation", "difference", "change",
    "increase", "decrease", "variation", "shift",

    # Causality / relationship
    "related", "linked", "associated", "caused", "led to",
    "resulted in", "impact", "triggered",

    # Summarization / health
    "summary", "overview", "what happened", "overall",
    "anything unusual", "system behavior"
]


# Retrieval Configuration
TOP_K = 5              # Default (fallback)
TOP_K_SIMPLE = 4       # Range 3-5 -> 4
TOP_K_COMPLEX = 10     # Range 8-12 -> 10
SIMILARITY_THRESHOLD = 0.25  # Ignore chunks with score below this
MIN_KEYWORD_MATCHES = 1      # Minimum keyword matches required if query has significant words

# Qdrant Configuration
COLLECTION_NAME = "log_chunks"
VECTOR_SIZE = 768  # nomic-embed-text-v1.5 produces 768-dimensional vectors
BATCH_SIZE = 4     # Small batch size for GTX 1650 to prevent freezing
