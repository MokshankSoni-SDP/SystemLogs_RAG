# 🤖 RAG System for System Logs

An intelligent **Retrieval-Augmented Generation (RAG)** system for querying, analyzing, and summarizing system logs (Linux, Windows, macOS) and container logs (Docker, Kubernetes) through a conversational AI interface.

---

## 📋 Table of Contents

- [Overview & Key Features](#-overview)
- [Complete Pipeline](#-pipeline)
- [Module Reference](#module-reference)
- [Error Taxonomy](#error-taxonomy)

- [Evolution of Project (from basic RAG to Deterministic Engine)](#hybrid-log-analytics-system)

---

## 🔎 Overview

This system allows you to upload raw system or container logs and ask natural-language questions about them. The pipeline automatically:

- **Parses** raw log lines from multiple OS formats into canonical records
- **Chunks** them intelligently using time-aware boundaries
- **Embeds** chunks using a local GPU-accelerated model
- **Routes** each query to the optimal execution path (Simple RAG, Complex RAG, or full Scan)
- **Answers** questions with evidence-based responses and cited sources

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-OS Support** | Parses Linux (syslog), Windows (Event Viewer), macOS, Docker, and Kubernetes logs |
| **3-Way Query Routing** | Automatically routes to Simple RAG, Complex RAG, or Scan-and-Summarize |
| **Time-Aware Chunking** | Splits logs on time gaps, OS-specific sizes, and structural boundaries |
| **Semantic Error Classification** | 18+ error categories (disk_full, OOM, connection_failed, etc.) |
| **Multi-Collection Knowledge Base** | Separate Qdrant collections per log source/project |

---

## 🔧 Pipeline

### 1. Ingestion Pipeline

When you upload logs, the following steps occur in order:

```
Raw Log Text
     │
     ▼
[Step 0] Archive Raw Logs
     │  → Detect OS hint (linux/windows/macos/unknown)
     │  → Save to raw_logs/<os_type>/<timestamp>_<source>.log
     │
     ▼
[Step 1] Preprocessor (preprocessor.py)
     │  → Split into lines
     │  → Extract timestamp (ISO-8601 normalized)
     │  → Extract log level (ERROR / WARN / INFO / UNKNOWN)
     │  → Detect OS type
     │  → Extract source/process name
     │  → Sanitize: replace IPs → <IP>, PIDs → <PID>, UUIDs → <UUID>,
     │               memory addresses → <ADDR>
     │  → Produces: List[canonical_record]
     │
     ▼
[Step 2] Chunker (chunker.py)
     │  → Detects dominant OS from records
     │  → Sets chunk size:
     │      - Linux/macOS: max 40 lines, time gap threshold 120s
     │      - Windows:     max 50 lines, time gap threshold 120s
     │      - Container:   max 30 lines, time gap threshold 60s
     │  → Splits on:
     │      Rule 1 — Max line count reached
     │      Rule 2 — Time gap > threshold between adjacent records
     │      Rule 3 — Container structural keywords (OOM, panic, exit code…)
     │  → Produces: List[chunk] with chunk_id, text, start_time, end_time,
     │              os_hint, log_type, source_file
     │
     ▼
[Step 3] Embedding Model (embeddings.py)
     │  → Model: nomic-ai/nomic-embed-text-v1.5 (768-dim vectors)
     │  → Falls back to CPU if CUDA unavailable
     │  → Produces: np.ndarray of shape (N, 768)
     │
     ▼
[Step 4] Vector DB (vector_db.py)
     │  → Qdrant (disk-backed at ./qdrant_data)
     │  → Stores chunk text + metadata as payload
     │  → Collection: "log_chunks" (or user-selected)
     │  → Cosine similarity index
     ▼
  Done ✅
```

---

### 2. Query Pipeline — 3-Way Routing

Every question passes through a **two-stage classifier** before being routed to one of three execution paths.

```
User Question
     │
     ▼
[Stage A] Hard Keyword Filter (query_classifier.py)
     │  → Checks for scan-mode keywords:
     │      counting ("how many", "total", "number of")
     │      global scope ("all errors", "across the logs")
     │      trends ("over time", "spike", "distribution")
     │      ranking ("most common", "top errors")
     │  → If matched → SCAN
     │
     ▼ (no match)
[Stage B] LLM Intent Classifier (Groq API)
     │  → Sends structured prompt to llama-3.3-8b
     │  → Model decides: RAG or SCAN
     │  → Temperature 0.1 (deterministic)
     │  → If SCAN → route to Scan path
     │
     ▼ (RAG)
[Stage C] Complexity Classifier
     │  → If query > 12 words → COMPLEX
     │  → If contains "why/how/explain/cause/compare…" → COMPLEX
     │  → Otherwise → SIMPLE
     │
     ├─── SIMPLE → Simple RAG Path
     ├─── COMPLEX → Complex RAG Path
     └─── SCAN → Scan-and-Summarize Path
```

---

#### 2.1 Simple RAG Path

For focused, specific questions (e.g., *"Why did sshd fail?"*, *"Show an example of a timeout error"*).

```
Query
  │
  ▼
[1] Embed query → 768-dim vector
  │
  ▼
[2] Qdrant vector search → top_k=4 chunks (cosine similarity)
  │
  ▼
[3] Filter chunks:
     - Remove chunks with similarity score < 0.25
     - If score > 0.55 → keep (strong semantic match, skip keyword check)
     - Otherwise → require keyword overlap with query terms
     - Fallback: keep top-2 chunks with score > 0.40 if all filtered out
  │
  ▼
[4] LLM Answer (llama-3.3-70b-versatile, temp=0.3)
     - Evidence-only prompt (no hallucination)
     - Cites chunk IDs and timestamps
     - Ends with confidence: High / Medium / Low
  │
  ▼
Answer + Source Evidence
```

---

#### 2.2 Complex RAG Path

For multi-faceted questions requiring analysis across several angles (e.g., *"How did the disk error impact network connectivity and what processes were affected?"*).

```
Query
  │
  ▼
[1] LLM Query Planner (query_planner.py, temp=0.1)
     - Decomposes the question into 2-4 focused sub-queries
     - Returns structured JSON list
  │
  ▼
[2] For each sub-query:
     - Embed sub-query → 768-dim vector
     - Qdrant search → top_k=10 chunks
     - Apply similarity + keyword filter
  │
  ▼
[3] Deduplicate all retrieved chunks by chunk_id
    (preserves order, keeps first occurrence)
  │
  ▼
[4] LLM Answer using all unique deduped chunks
     - Evidence-only prompt
     - Confidence rating
  │
  ▼
Answer + Source Evidence + Sub-queries Used
```

---

#### 2.3 Scan-and-Summarize Path

For queries requiring full-log aggregation, counting, trends, and rankings (e.g., *"How many disk errors occurred?"*, *"Which process caused the most errors?"*, *"What errors are present in the logs?"*).

This path is **deterministic** — the LLM never reads raw logs directly. It only plans operations and explains pre-computed structured statistics.

```
Query
  │
  ▼
[1] LLM Execution Planner (intent_planner.py, temp=0.1)
     - Converts question to a structured execution plan (JSON)
     - Plan has ordered steps with operations + parameters
     - Available operations:
         list_unique_errors, count_occurrences, filter_by_process,
         get_recent_events, get_before_after_context, bucket_by_time,
         sort_by_time, get_first_event, get_last_event,
         filter_by_time_range, find_peak, rank_by,
         filter_by_os, exclude_process
  │
  ▼
[2] Load archived raw logs from raw_logs/
     - Reads all .log files from disk
     - Re-preprocesses them (system or container mode)
     - Sorts by timestamp
  │
  ▼
[3] Execute plan deterministically (scan_operations.py)
     - Each step pipes its output into the next
     - Uses error_taxonomy.py for semantic error classification
       instead of relying on log-level fields
     - Steps can chain: bucket_by_time → find_peak → LLM summary
  │
  ▼
[4] Format result as structured stats
     (_format_scan_stats in pipeline.py)
     - Converts raw result to typed dict
       (error_count / frequency_distribution / time_series /
        peak_detection / ranked_items / log_list / count)
     - Prevents abstraction mixing (LLM never sees aggregate as log evidence)
  │
  ▼
[5] LLM Summarizer (temp=0.3)
     - Strict rules: explain ONLY what the stats say
     - NEVER infer, recalculate, or treat counts as error messages
  │
  ▼
Natural Language Summary
```

---

### Module Reference

| Module | Role |
|---|---|
| `config.py` | Loads `.env`, defines model names, chunk sizes, regex patterns, retrieval thresholds |
| `pipeline.py` | Top-level orchestrator for ingestion and query execution |
| `preprocessor.py` | Parses raw log lines into canonical records; handles Linux, Windows, macOS, Docker, K8s formats |
| `chunker.py` | Time-aware chunker; respects OS-specific line limits, time gaps, and multi-line events |
| `embeddings.py` | Wraps `nomic-embed-text-v1.5` via `sentence-transformers`; GPU-batched encoding |
| `vector_db.py` | Qdrant client wrapper; disk-backed storage; cosine similarity search; multi-collection support |
| `llm_client.py` | Groq API client; separate `plan_query()` and `answer_question()` methods; key rotation |
| `query_classifier.py` | Two-stage classifier: keyword filter + LLM intent → simple / complex / scan |
| `query_planner.py` | Generates sub-queries for Complex RAG path |
| `intent_planner.py` | Generates structured execution plans for Scan path |
| `scan_operations.py` | 14 deterministic Python operations on preprocessed log records |
| `error_taxonomy.py` | 18+ rule-based semantic error categories mapped from message patterns |
| `log_archiver.py` | Persists raw log text to disk organized by OS type and timestamp |

---

## Error Taxonomy

The system uses `error_taxonomy.py` to semantically classify errors from log message content (not just log level fields), covering:

| Category | Examples |
|---|---|
| `disk_full` | "no space left on device", "filesystem full" |
| `out_of_memory` | "OOM killer", "cannot allocate memory" |
| `permission_denied` | "access denied", "operation not permitted" |
| `authentication_failed` | "failed password", "login failed" |
| `connection_failed` | "connection refused", "broken pipe", "network unreachable" |
| `timeout` | "timed out", "request timeout" |
| `process_crash` | "segmentation fault", "core dumped", "abort" |
| `service_failure` | "failed to start", "unexpected exit", "exit code" |
| `kernel_error` | "kernel panic", "oops" |
| `hardware_error` | "machine check", "thermal shutdown", "fan failure" |
| `update_failure` | "update failed", "patch failed", "servicing stack error" |
| `configuration_error` | "syntax error", "invalid configuration" |
| ... and more | `dns_failure`, `memory_leak`, `cpu_exhaustion`, `resource_limit`, etc. |

Any error-like log that doesn't match a known category falls back to `unknown_error`.

---

## 📚 Multi-Collection Support

The system supports multiple **Qdrant collections** to logically separate different log sets:

- Each uploaded log set can be stored in a named collection (e.g., `server_a_logs`, `prod_logs`)
- Switch between collections from the UI sidebar without restarting
- The active collection is persisted to `.active_collection` and restored on startup
- Each collection uses the same 768-dim cosine similarity index

---
## 🚀 Setup & Installation

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional but recommended; GTX 1650 or better)
- [Groq API Key](https://console.groq.com/) (free tier available)

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd RAG_SysLogs2

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate       # Windows
# source venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create .env file
cp .env.example .env        # or create manually
```

---

## 🌐 Running the Application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

### Docker (Alternative)

```bash
docker build -t rag-syslogs .
docker run -p 8501:8501 --env-file .env rag-syslogs
```

---

## 💡 How to Use

### Step 1: Ingest Logs

1. Go to the **📤 Log Ingestion** tab
2. Upload a `.log`, `.txt`, or `.csv` file — or paste log text directly
3. Select log type: **System Logs** (Linux/Windows/macOS) or **Container Logs** (Docker/K8s)
4. Click **🚀 Ingest Logs**

### Step 2: Ask Questions

Switch to the **💬 Chat & Analysis** tab and ask questions like:

| Query Type | Example Questions |
|---|---|
| ⚡ Simple RAG | *"Why did the sshd process fail?"* |
| ⚡ Simple RAG | *"Show me an example of an authentication error"* |
| 🧠 Complex RAG | *"How did the disk error impact services and what was affected?"* |
| 🔍 Scan | *"How many errors occurred in total?"* |
| 🔍 Scan | *"Which process caused the most errors?"* |
| 🔍 Scan | *"What errors are present in the logs?"* |
| 🔍 Scan | *"When did error activity peak?"* |
| 🔍 Scan | *"Show the top 3 most frequent error types"* |

Multiple questions in one message are supported: separate them with `?`.

### Step 3: Manage Collections

Use the **📚 Knowledge Base** panel in the sidebar to:
- Switch between log collections
- Create new named collections
- View indexed chunk count

---


## 📦 Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.31.1 | Web UI |
| `qdrant-client` | 1.7.3 | Vector database |
| `sentence-transformers` | 2.3.1 | Embedding model wrapper |
| `transformers` | 4.37.2 | Model loading |
| `torch` | 2.5.1 | GPU computation |
| `groq` | 0.9.0 | LLM API client |
| `httpx` | 0.27.0 | HTTP client (proxy-safe) |
| `python-dotenv` | 1.0.1 | `.env` file loading |
| `numpy` | 1.26.4 | Numerical operations |
| `einops` | 0.8.2 | Tensor operations for nomic model |

---
---
---

# Hybrid Log Analytics System

## Evolution from Naïve RAG to Deterministic Scan & Summarize Engine

---

# 1. Introduction

## 1.1 Project Objective

The goal of this project was to build a system capable of answering arbitrary questions over raw operating system logs (Windows, Linux, macOS) with accuracy, honesty, and reproducibility.

Initially designed as a Retrieval-Augmented Generation (RAG) system, the architecture evolved into a hybrid analytics engine after discovering fundamental limitations of embeddings when applied to logs.

> Core realization: **Logs are not documents. They are time-series event streams.**

This insight drove the architectural transformation.

---

# 2. Phase 0 — Initial Assumption: Logs as Documents

## Original Hypothesis

System logs can be treated as textual documents:

```
Raw Logs
   ↓
Chunking
   ↓
Embedding
   ↓
Vector DB
   ↓
LLM Answer
```

This was inspired by successful RAG pipelines used for PDFs and knowledge bases.

---

# 3. Phase 1 — Naïve RAG Implementation

## 3.1 Pipeline

1. Read raw logs line-by-line.
2. Create fixed-size chunks.
3. Generate embeddings (`nomic-embed-text-v1.5`).
4. Store in Qdrant.
5. Retrieve top-k chunks.
6. LLM answers using retrieved context.

## 3.2 Early Success

Worked for semantic questions:

* “What does this error mean?”
* “Explain this failure.”
* “What happened in this log snippet?”

These are **local semantic questions**, which RAG handles well.

---

# 4. Phase 2 — The Counting Failure

When testing aggregation-style queries:

* “How many ERROR logs occurred?”
* “Which error happened first?”
* “Most frequent error?”
* “When did errors peak?”

The system failed.

### Observed Behavior

* Approximate answers
* Guessed counts
* Inconsistent numbers

### Root Cause

Embeddings retrieve **top-k similar chunks**, not the entire dataset.

> Retrieval ≠ Exhaustiveness
> Embeddings cannot count, sort, or perform time-series analysis.

This was the first major architectural turning point .

---

# 5. Phase 3 — Structural Improvements (Still RAG-Based)

## 5.1 OS-Agnostic Canonical Parsing

Logs were normalized into a canonical schema:

```json
{
  "timestamp": "...",
  "level": "ERROR | WARN | INFO | UNKNOWN",
  "process": "...",
  "message": "...",
  "os_hint": "windows | linux | macos"
}
```

Normalization included:

* IP → `<IP>`
* Port → `<PORT>`
* PID → `<PID>`
* Memory address → `<ADDR>`

Purpose:

* Improve embedding similarity
* Remove noise
* Enable structured filtering 

---

## 5.2 Time-Aware Chunking

Instead of fixed-size chunks:

* New chunk if time gap > 120 seconds
* Hard cap on max lines per chunk
* OS-specific line constraints
* No overlap

Metadata stored:

* `chunk_id`
* `start_time`
* `end_time`
* `os_hint`
* `line_count`

Improved semantic retrieval — but aggregation still failed.

---

# 6. Phase 4 — Fundamental Realization

We identified a conceptual mismatch:

| Documents       | Logs                    |
| --------------- | ----------------------- |
| Topic-based     | Time-series             |
| Contextual      | Sequential              |
| Local reasoning | Global reasoning        |
| Sampling works  | Exhaustiveness required |

Logs require:

* Counting
* Sorting
* Time bucketing
* Sequence reconstruction
* Negative reasoning

Embeddings cannot guarantee these operations .

---

# 7. Phase 5 — Introduction of Hybrid Architecture

## Core Idea

Separate semantic reasoning from computational reasoning.

Instead of forcing one technique to solve everything:

```
User Question
      ↓
Intent Classification
      ↓
  ┌───────────────┐
  │   RAG Path    │
  │   SCAN Path   │
  └───────────────┘
```

This became the defining architectural decision .

---

# 8. Phase 6 — Intent Classification Layer

Before execution, the system determines:

* Does this require semantic interpretation?
* Or deterministic computation?

Classification uses:

* Keyword heuristics
* LLM-based fallback routing

### Example Routing

| Question                    | Path |
| --------------------------- | ---- |
| What does this error mean?  | RAG  |
| How many errors occurred?   | SCAN |
| Which error occurred first? | SCAN |
| Explain this failure        | RAG  |

---

# 9. Path A — Retrieval-Augmented Generation (RAG)

Used for:

* Explanation
* Interpretation
* Contextual analysis

### Steps

1. Embed query.
2. Retrieve top-k chunks.
3. Construct evidence-only prompt.
4. LLM answers strictly from retrieved context.

Rules:

* Must cite evidence.
* Must not infer missing facts.
* Must state “Insufficient evidence” if applicable.

RAG now handles only what it is good at: explanation.

---

# 10. Path B — Deterministic Scan & Summarize (Breakthrough)

This is the major architectural innovation .

## 10.1 No-Vector Truth Layer

Logs stored in structured JSON format for full traversal.

## 10.2 LLM as Semantic Compiler

Instead of computing, LLM translates natural language into execution plans.

Example:

```json
{
  "operation": "count_occurrences",
  "parameters": {
    "level": "ERROR"
  }
}
```

## 10.3 Deterministic Execution

Python performs:

* Counting
* Grouping
* Sorting
* Time bucketing
* Signature detection
* Before/after context reconstruction

Example output:

```json
{
  "total_errors": 927,
  "by_error_type": {
    "disk_full": 130,
    "out_of_memory": 22
  }
}
```

## 10.4 LLM as Explainer (Not Calculator)

LLM receives structured results and converts them into natural language.

Key Rule:

> LLMs explain results — they do not compute them. 

---

# 11. Solved Failure Modes

The hybrid design directly solves failures documented in the PDF :

| Question Type     | Why RAG Fails            | How Hybrid Solves       |
| ----------------- | ------------------------ | ----------------------- |
| Counting          | Partial retrieval        | Full dataset scan       |
| Ranking           | Semantic bias            | Frequency aggregation   |
| First/Last        | Unordered chunks         | Timestamp sorting       |
| Trend Detection   | No time-series awareness | Time bucketing          |
| Before/After      | Broken continuity        | Sequence reconstruction |
| Negative Evidence | Cannot detect absence    | Exhaustive scan         |
| Correlation       | Requires computation     | Deterministic logic     |

---

# 12. Final End-to-End Pipeline

```
1. User Upload
2. Canonical Preprocessing
3. Time-Aware Chunking (RAG only)
4. Embedding (RAG path only)
5. Intent Classification
6A. RAG Execution
6B. SCAN Execution
7. LLM Explanation
8. Confidence Scoring
```

---

# 13. Confidence Model

Confidence is rule-based:

* High → Deterministic complete scan
* Medium → Partial grouping
* Low → Insufficient evidence

No confidence is guessed by the LLM.

---

# 14. Architectural Principles Established

1. Logs are event streams, not documents.
2. Embeddings are probabilistic, not exhaustive.
3. Retrieval is sampling, not analytics.
4. Counting must be deterministic.
5. LLMs are translators, not calculators.
6. Separation of abstraction layers prevents hallucination.
7. Negative reasoning requires full scanning.
8. Correlation requires computation, not generation.

---

# 15. Final System State

The project evolved from:

> “Let’s build a RAG over logs.”

to

> “Let’s build a log analytics engine that intelligently uses RAG where appropriate.”

It is now a:

## Hybrid Log Analytics Engine

With:

* OS-agnostic ingestion
* Canonical structured storage
* Time-aware chunking
* Vector retrieval for semantics
* Deterministic scan for computation
* LLM-based explanation layer
* Strict honesty enforcement
* Reproducible outputs

---

# 16. Maturity Statement

The system now answers only what the data truly supports — and explicitly refuses when it cannot.

Every answer is:

* Reproducible
* Verifiable
* Deterministic (when required)
* Grounded in evidence 

---

If you want next, I can:

* Add an Abstract + Keywords section
* Add a Mermaid architecture diagram
* Add a comparison table vs pure RAG
* Convert this into a polished research-report style version
* Or generate a 5–7 minute mentor presentation script

Just tell me which format you need next.
