---
title: Rag Syslogs
emoji: 🦀
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# RAG System for OS System Logs

A simple, robust RAG system for reliable question answering over OS system logs (Linux, Windows, macOS). Focuses on storing truth, retrieving context, and letting the LLM reason at query time.

## 🎯 Core Philosophy

**Not SIEM. Not incident detection. Not log parsing.**

This system:
- Stores truth (raw logs with minimal cleaning)
- Retrieves context (semantic search)
- Lets the LLM reason at query time

## 🚩 Features

- **OS-Agnostic Preprocessing**: Handles Linux, Windows, and macOS logs with unified normalization
- **Time-Aware Chunking**: Sequential chunking with intelligent boundaries (line count, time gaps, multi-line events)
- **Dual Query Paths**: 
  - Simple queries → Direct retrieval
  - Complex queries → LLM-assisted query planning
- **Evidence-Based Answering**: LLM answers only from retrieved evidence with explicit uncertainty handling
- **Clean UI**: Streamlit interface with query type display and enhanced metadata

## 🏗️ Architecture

```
Raw Logs
  ↓
Preprocessing (OS-agnostic normalization)
  ├─ Timestamp extraction
  ├─ Log level normalization  
  ├─ Message cleaning (IP/Port/PID → placeholders)
  └─ OS hint detection
  ↓
Time-Aware Chunking
  ├─ Line count boundary (Linux: 40, Windows: 25)
  ├─ Time gap boundary (120s threshold)
  └─ Structural boundary (multi-line events)
  ↓
Embedding (nomic-embed-text-v1.5, local)
  ↓
Vector DB (Qdrant)
  ↓
Query Classification
  ├─ Simple → Direct Retrieval → Answer
  └─ Complex → LLM Planning → Multi-Retrieval → Dedup → Answer
  ↓
Answer Generation (Groq LLaMA 3)
```

## 📋 Prerequisites

- Python 3.8+
- Groq API key ([Get one here](https://console.groq.com/))
- 4GB+ RAM (for embedding model)
- GPU optional (CPU auto-detected)

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd d:\PROJECTS\MLProjects\RAG_SysLogs2

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Edit `.env` file:
```bash
GROQ_API_KEY=your_actual_groq_api_key_here
```

### 3. Run the Application

```bash
streamlit run app.py
```

Open browser at `http://localhost:8501`

## 💻 Usage

### Ingest Logs

1. **Upload a file** or **paste logs** directly
2. Click **"Ingest Logs"**
3. System will:
   - Preprocess logs (normalize timestamps, levels, clean messages)
   - Chunk with time-aware boundaries
   - Generate embeddings
   - Store in vector database

### Ask Questions

1. Enter your question
2. System automatically classifies as:
   - **Simple**: "What errors occurred?" → Direct retrieval
   - **Complex**: "How many failed logins over time?" → LLM planning
3. View answer with:
   - Query type indicator
   - Sub-queries (for complex)
   - Retrieved chunks with timestamps
   - OS hints and metadata

## 📁 Project Structure

```
RAG_SysLogs2/
├── .env                    # API keys
├── requirements.txt        # Dependencies
├── README.md              # This file
├── app.py                 # Streamlit UI
├── src/
│   ├── config.py          # Configuration constants
│   ├── preprocessor.py    # OS-agnostic log parsing
│   ├── chunker.py         # Time-aware chunking
│   ├── embeddings.py      # Embedding model wrapper
│   ├── vector_db.py       # Qdrant integration
│   ├── query_classifier.py # Simple/complex classification
│   ├── query_planner.py   # LLM-assisted planning
│   ├── llm_client.py      # Groq LLM client
│   └── pipeline.py        # RAG orchestration
└── test_logs/             # Sample logs
    ├── linux_syslog.txt
    ├── windows_event.txt
    └── macos_system.txt
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

### Preprocessing
- `IP_PATTERN`, `PORT_PATTERN`, `PID_PATTERN`: Regex for message cleaning

### Chunking
- `CHUNK_SIZE_LINUX`: 40 lines (default)
- `CHUNK_SIZE_WINDOWS`: 25 lines (default)
- `TIME_GAP_THRESHOLD_SECONDS`: 120 seconds (default)

### Query Classification
- `COMPLEX_QUERY_KEYWORDS`: Keywords for complex query detection

### Models
- `EMBEDDING_MODEL`: nomic-ai/nomic-embed-text-v1.5
- `LLM_MODEL`: llama-3.3-70b-versatile

## 🔧 How It Works

### Preprocessing

Each raw log line becomes a canonical record:
```python
{
  "timestamp": "2024-01-15T10:23:45",  # ISO-8601 or fallback
  "host": "webserver",
  "source": "sshd",
  "level": "ERROR",  # INFO | WARN | ERROR | UNKNOWN
  "message": "Failed password for root from <IP> port <PORT>",
  "raw": "original log line",
  "os_hint": "linux"  # linux | windows | macos | unknown
}
```

### Time-Aware Chunking

Three boundary types:
1. **Line count**: 40 (Linux/macOS) or 25 (Windows)
2. **Time gap**: 120 seconds between consecutive records
3. **Structural**: Multi-line events (stack traces) stay together

No overlap. Each chunk includes `start_time`, `end_time`, `os_hint`, `line_count`.

### Query Processing

**Simple Query** (e.g., "What errors occurred?"):
1. Embed query
2. Retrieve top-K chunks
3. Generate answer

**Complex Query** (e.g., "How many failed logins over time?"):
1. LLM generates 3–6 sub-queries
2. Retrieve chunks for each sub-query
3. Deduplicate by chunk ID
4. Generate evidence-based answer

## 🎨 Example Questions

**Simple Queries**:
- "What errors occurred in the logs?"
- "Show me all failed SSH attempts"
- "What services restarted?"

**Complex Queries**:
- "How many errors occurred over time?"
- "What is the pattern of failed login attempts?"
- "Compare authentication failures across different sources"

## 🧪 Testing

Sample logs provided in `test_logs/`:
- `linux_syslog.txt`: Standard syslog with SSH, systemd, kernel events
- `windows_event.txt`: Event Viewer format with security logs
- `macos_system.txt`: macOS system logs with launchd, kernel messages

Upload these to test the system with different OS formats.

## ⚠️ Limitations

- **In-memory storage**: Qdrant data cleared on restart
- **No persistent state**: Re-ingest logs after restarting
- **API dependency**: Requires Groq API key
- **No authentication**: Single-user local application
- **No real-time ingestion**: Manual upload/paste only

## 🎓 What This Is NOT

❌ SIEM or incident detection
❌ Log parsing framework  
❌ Severity scoring system
❌ Alert classification
❌ Hardcoded service logic

✅ Simple RAG for log Q&A
✅ OS-agnostic log understanding
✅ Evidence-based reasoning

## 🤝 Contributing

This is a demonstration project showcasing clean RAG architecture for logs. Feel free to fork and enhance!

## 📝 License

Provided as-is for educational purposes.
