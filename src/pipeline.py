"""
RAG Pipeline module.
Orchestrates the entire workflow: preprocessing, chunking, embedding, retrieval, and answering.
Supports both simple and complex query paths.
"""
from typing import List, Dict, Tuple, Any

from .preprocessor import preprocess_logs
from .chunker import chunk_preprocessed_logs
from .embeddings import EmbeddingModel
from .vector_db import VectorDB
from .llm_client import LLMClient
from .query_classifier import classify_query
from .query_planner import generate_sub_queries
from .config import TOP_K, TOP_K_SIMPLE, TOP_K_COMPLEX, SIMILARITY_THRESHOLD
from .log_archiver import persist_raw_logs
from .preprocessor import detect_os_hint, preprocess_logs
from .intent_planner import generate_execution_plan
from .scan_operations import execute_plan


class RAGPipeline:
    """
    Main pipeline that orchestrates the RAG workflow.
    Supports preprocessing, time-aware chunking, and dual query paths.
    """
    
    def __init__(self):
        """Initialize all components of the pipeline."""
        print("Initializing RAG Pipeline...")
        
        # Initialize components
        self.embedding_model = EmbeddingModel()
        self.vector_db = VectorDB()
        self.llm_client = None  # Lazy initialization to avoid API key check on startup
        
        # Health check for Groq client (will fail fast if proxies are misconfigured)
        from .llm_client import groq_health_check
        health_status = groq_health_check()
        print(f"Groq Health Check: {health_status}")
        
        print("RAG Pipeline initialized successfully")
    
    def _get_llm_client(self) -> LLMClient:
        """Lazy initialization of LLM client."""
        if self.llm_client is None:
            self.llm_client = LLMClient()
        return self.llm_client
    
    def run(self, question: str) -> Dict:
        """
        Run the full RAG pipeline (e2e).
        
        Args:
            question: User's question
            
        Returns:
            Dictionary containing answer, sources, and metadata
        """
        llm_client = self._get_llm_client()
        # Reset token tracking
        llm_client.reset_usage()

        print(f"=== Starting pipeline for query: '{question}' ===")
        
        answer, retrieved_chunks, metadata = self.query(question)
        
        # Get usage stats
        usage_stats = llm_client.get_usage_stats()
        
        print(f"=== Pipeline finished for query: '{question}' ===")
        
        return {
            "answer": answer,
            "sources": retrieved_chunks,
            "metadata": metadata,
            "usage_stats": usage_stats
        }
    
    def ingest_logs(self, log_text: str, source_name: str = "uploaded_logs", log_type: str = "system") -> Dict:
        """
        Ingest logs with preprocessing: preprocess -> chunk -> embed -> store.
        
        Args:
            log_text: Raw log content
            source_name: Name of the source (file name or identifier)
            log_type: Type of logs ("system" or "container")
            
        Returns:
            Dictionary with ingestion statistics
        """
        print(f"\n=== Starting log ingestion for: {source_name} (Type: {log_type}) ===")
        
        # Step 0: Archive raw logs (New Requirement)
        try:
            # Quick OS detection on first few lines
            os_hint = "unknown"
            for line in log_text.splitlines()[:20]:
                if line.strip():
                    hint = detect_os_hint(line)
                    if hint != "unknown":
                        os_hint = hint
                        break
            
            # Persist to disk
            saved_path = persist_raw_logs(log_text, source_name, os_hint, log_type)
            print(f"✅ Raw logs archived at: {saved_path}")
        except Exception as e:
            print(f"⚠️ Warning: Checkpoint archival failed: {e}")

        # Step 1: Preprocess the logs
        print(f"Step 1: Preprocessing logs ({log_type} mode)...")
        preprocessed_records = preprocess_logs(log_text, log_type=log_type)
        print(f"Preprocessed {len(preprocessed_records)} log records")
        
        if not preprocessed_records:
            return {
                "success": False,
                "message": "No log records parsed. Log text may be empty or invalid.",
                "num_chunks": 0
            }
        
        # Step 2: Chunk the preprocessed records (time-aware)
        print("Step 2: Chunking logs (time-aware, OS/Container specific)...")
        chunks = chunk_preprocessed_logs(preprocessed_records, source_name)
        print(f"Created {len(chunks)} chunks")
        
        if not chunks:
            return {
                "success": False,
                "message": "No chunks created from preprocessed records.",
                "num_chunks": 0
            }
        
        # Step 3: Generate embeddings
        print("Step 3: Generating embeddings...")
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_model.encode(chunk_texts)
        print(f"Generated {len(embeddings)} embeddings")
        
        # Step 4: Store in vector database
        print("Step 4: Storing in vector database...")
        self.vector_db.store_chunks(chunks, embeddings, source_name)
        
        total_vectors = self.vector_db.count()
        
        print(f"=== Ingestion complete ===\n")
        
        return {
            "success": True,
            "message": f"Successfully ingested {len(chunks)} chunks from {source_name}",
            "num_chunks": len(chunks),
            "total_vectors": total_vectors
        }
    
    def query(self, question: str, top_k: int = None) -> Tuple[str, List[Dict], Dict]:
        """
        Answer a question using three-way routing (simple/complex RAG, or scan mode).
        
        Args:
            question: User's question
            top_k: Optional hardcoded top_k, otherwise adaptive
            
        Returns:
            Tuple of (answer, retrieved_chunks, metadata)
            metadata includes: query_type, sub_queries (if complex), execution_plan (if scan)
        """
        print(f"\n=== Processing query: {question} ===")
        
        # Check if we have any data
        if self.vector_db.count() == 0:
            return (
                "No logs have been ingested yet. Please upload and ingest logs first.",
                [],
                {"query_type": "none"}
            )
        
        # Two-stage classification (keywords + LLM)
        llm_client = self._get_llm_client()
        query_type = classify_query(question, llm_client)
        print(f"Query classified as: {query_type.upper()}")
        
        metadata = {"query_type": query_type}
        
        # Route to appropriate query path
        if query_type == "scan":
            # Scan-and-Summarize path (deterministic)
            answer, scan_metadata = self._scan_query_path(question)
            metadata.update(scan_metadata)
            retrieved_chunks = []  # Scan mode doesn't use chunks
        elif query_type == "simple":
            # Simple RAG path
            effective_top_k = top_k or TOP_K_SIMPLE
            metadata["top_k"] = effective_top_k
            print(f"Adaptive top_k set to: {effective_top_k}")
            answer, retrieved_chunks = self._simple_query_path(question, effective_top_k)
        else:
            # Complex RAG path
            effective_top_k = top_k or TOP_K_COMPLEX
            metadata["top_k"] = effective_top_k
            print(f"Adaptive top_k set to: {effective_top_k}")
            answer, retrieved_chunks, sub_queries = self._complex_query_path(question, effective_top_k)
            metadata["sub_queries"] = sub_queries
        
        print("=== Query processing complete ===\n")
        
        return answer, retrieved_chunks, metadata
    
    def _scan_query_path(self, question: str) -> Tuple[str, Dict]:
        """
        Scan-and-Summarize path: LLM planning -> Deterministic execution -> LLM summary.
        
        Args:
            question: User's question requiring global visibility or aggregation
            
        Returns:
            Tuple of (answer, metadata_dict)
        """
        print("-> Using SCAN-AND-SUMMARIZE path")
        
        # Step 1: Generate execution plan using LLM
        print("Step 1: Generating execution plan with LLM...")
        try:
            llm_client = self._get_llm_client()
            plan = generate_execution_plan(question, llm_client)
            print(f"Generated plan with {len(plan['steps'])} step(s)")
            for i, step in enumerate(plan['steps'], 1):
                print(f"  Step {i}: {step['operation']}")
        except ValueError as e:
            return f"Failed to generate execution plan: {e}", {"error": str(e)}
        except Exception as e:
            return f"Error in planning: {e}", {"error": str(e)}
        
        # Step 2: Load preprocessed logs from raw_logs directory
        print("Step 2: Loading archived raw logs...")
        logs = self._load_preprocessed_logs()
        
        if not logs:
            return "No archived logs found. Please ingest logs first.", {"error": "no_logs"}
        
        print(f"Loaded {len(logs)} preprocessed log records")
        
        # Step 3: Execute plan deterministically
        print("Step 3: Executing plan...")
        try:
            result = execute_plan(logs, plan)
            print(f"Execution complete. Result type: {type(result).__name__}")
        except Exception as e:
            return f"Execution failed: {e}", {"error": str(e), "plan": plan}
        
        # Step 4: Summarize results with LLM
        print("Step 4: Summarizing results with LLM...")
        try:
            summary = self._summarize_scan_results(result, question, plan, llm_client)
        except Exception as e:
            return f"Summarization failed: {e}", {"error": str(e), "raw_result": str(result)[:500]}
        
        metadata = {
            "execution_plan": plan,
            "log_count": len(logs),
            "result_type": type(result).__name__
        }
        
        return summary, metadata
    
    def _load_preprocessed_logs(self) -> List[Dict]:
        """
        Load and preprocess logs from raw_logs directory.
        
        Returns:
            List of preprocessed log records
        """
        from pathlib import Path
        
        all_logs = []
        raw_logs_dir = Path("raw_logs")
        
        if not raw_logs_dir.exists():
            print("Warning: raw_logs directory does not exist")
            return []
        
        # Find all .log files
        log_files = list(raw_logs_dir.rglob("*.log"))
        
        if not log_files:
            print("Warning: No .log files found in raw_logs/")
            return []
        
        print(f"Found {len(log_files)} archived log file(s)")
        
        for log_file in log_files:
            try:
                raw_text = log_file.read_text(encoding="utf-8")
                
                # Detect log type from path (system or container)
                log_type = "container" if "container" in str(log_file) else "system"
                
                # Preprocess
                records = preprocess_logs(raw_text, log_type=log_type)
                all_logs.extend(records)
            except Exception as e:
                print(f"Warning: Failed to load {log_file}: {e}")
        
        # Sort by timestamp
        all_logs.sort(key=lambda x: x.get("timestamp", ""))
        
        return all_logs
    
    def _summarize_scan_results(self, result: Any, question: str, plan: Dict, llm_client) -> str:
        """
        Use LLM to summarize scan operation results in natural language.
        
        Args:
            result: Result from scan operations (list, dict, etc.)
            question: Original user question
            plan: Execution plan
            llm_client: LLM client instance
            
        Returns:
            Natural language summary
        """
        operation = plan['steps'][0]['operation'] if plan['steps'] else 'unknown'
        
        # Format result for LLM
        if isinstance(result, list):
            if len(result) == 0:
                formatted_result = "No results found."
            elif len(result) <= 20:
                formatted_result = "\\n".join(str(item) for item in result)
            else:
                # Truncate long lists
                formatted_result = "\\n".join(str(item) for item in result[:20])
                formatted_result += f"\\n... and {len(result) - 20} more items"
        elif isinstance(result, dict):
            formatted_result = "\\n".join(f"{k}: {v}" for k, v in list(result.items())[:20])
        else:
            formatted_result = str(result)[:1000]
        
        # Different prompts based on operation type
        if operation in ["list_unique_errors", "get_recent_events"]:
            prompt = f"""You are given a chronological list of log events.
Summarize them clearly for the user.

User question: {question}

Results:
{formatted_result}

Provide a clear, concise answer. Do not invent information."""
        
        elif operation == "count_occurrences":
            prompt = f"""You are given counts of log events.
Summarize the most important patterns.

User question: {question}

Results (format: item, count):
{formatted_result}

Provide a clear, concise answer focusing on patterns."""
        
        elif operation == "bucket_by_time":
            prompt = f"""You are given event counts over time.
Describe trends qualitatively.

User question: {question}

Results (format: timestamp: count):
{formatted_result}

Provide a clear, concise answer about trends. Do not calculate statistics."""
        
        else:
            prompt = f"""Summarize these log analysis results for the user.

User question: {question}

Results:
{formatted_result}

Provide a clear, concise answer."""
        
        try:
            # Create a fake chunk with the results as context
            result_chunk = {
                "chunk_id": "scan_results",
                "text": formatted_result,
                "source": "scan_operation",
                "score": 1.0
            }
            
            # Use the appropriate prompt based on operation
            modified_question = f"{question}\\n\\nContext: {prompt.split('Results:')[0]}"
            
            return llm_client.answer_question([result_chunk], modified_question)
        except:
            # Fallback: simple formatting
            return f"Here are the results:\\n\\n{formatted_result}"
    
    def _simple_query_path(self, question: str, top_k: int) -> Tuple[str, List[Dict]]:
        """
        Simple query path: Direct embed -> retrieve -> answer.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (answer, retrieved_chunks)
        """
        print("-> Using SIMPLE query path")
        
        # Step 1: Embed the query
        print("Step 1: Embedding query...")
        query_embedding = self.embedding_model.encode([question])[0]
        
        # Step 2: Retrieve relevant chunks
        print(f"Step 2: Retrieving top {top_k} relevant chunks...")
        raw_chunks = self.vector_db.search(query_embedding, top_k=top_k)
        
        # Step 2.5: Filter chunks (Similarity & Keyword)
        retrieved_chunks = self._filter_chunks(question, raw_chunks)
        print(f"Retrieved {len(raw_chunks)} chunks, filtered down to {len(retrieved_chunks)}")
        
        if not retrieved_chunks:
            # Fallback: If filtering was too strict, take top 2 high-similarity chunks regardless of keywords
            # This prevents the "No relevant log chunks" error when semantic match is good but keywords differ
            fallback_chunks = [c for c in raw_chunks if c.get('score', 0) > 0.40][:2]
            if fallback_chunks:
                print(f"Applying fallback: Using {len(fallback_chunks)} high-similarity chunks despite keyword mismatch.")
                retrieved_chunks = fallback_chunks
            else:
                return "No relevant log chunks found after filtering. Try rephrasing or ingestion more logs.", []
        
        # Step 3: Generate answer using LLM
        print("Step 3: Generating answer with LLM...")
        try:
            llm_client = self._get_llm_client()
            answer = llm_client.answer_question(retrieved_chunks, question)
        except ValueError as e:
            answer = str(e)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return answer, retrieved_chunks
    
    def _complex_query_path(self, question: str, top_k: int) -> Tuple[str, List[Dict], List[str]]:
        """
        Complex query path: LLM planning -> multi-retrieve -> deduplicate -> answer.
        
        Args:
            question: User's complex question
            top_k: Number of chunks to retrieve per sub-query
            
        Returns:
            Tuple of (answer, retrieved_chunks, sub_queries)
        """
        print("-> Using COMPLEX query path")
        
        # Step 1: Generate sub-queries using LLM planner
        print("Step 1: Planning query with LLM...")
        try:
            llm_client = self._get_llm_client()
            sub_queries = generate_sub_queries(question, llm_client)
            print(f"Generated {len(sub_queries)} sub-queries:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"  {i}. {sq}")
        except ValueError as e:
            # API key error
            return str(e), [], []
        except Exception as e:
            print(f"Query planning failed: {e}. Falling back to simple path.")
            answer, retrieved_chunks = self._simple_query_path(question, top_k)
            return answer, retrieved_chunks, [question]
        
        # Step 2: Retrieve chunks for each sub-query
        print(f"Step 2: Retrieving chunks for {len(sub_queries)} sub-queries...")
        all_chunks = []
        
        for i, sub_query in enumerate(sub_queries, 1):
            print(f"  Retrieving for sub-query {i}: {sub_query[:50]}...")
            query_embedding = self.embedding_model.encode([sub_query])[0]
            chunks = self.vector_db.search(query_embedding, top_k=top_k)
            
            # Filter each sub-query's results
            filtered_sq_chunks = self._filter_chunks(sub_query, chunks)
            all_chunks.extend(filtered_sq_chunks)
        
        # Step 3: Deduplicate chunks
        print("Step 3: Deduplicating retrieved chunks...")
        unique_chunks = self._deduplicate_chunks(all_chunks)
        print(f"Retrieved {len(all_chunks)} total chunks, {len(unique_chunks)} unique")
        
        if not unique_chunks:
            # Fallback for complex path
            fallback_chunks = [c for c in all_chunks if c.get('score', 0) > 0.45][:3]
            if fallback_chunks:
                print(f"Applying complex fallback: Using {len(fallback_chunks)} chunks.")
                unique_chunks = self._deduplicate_chunks(fallback_chunks)
            else:
                return "No relevant log chunks found for the planned sub-queries.", [], sub_queries
        
        # Step 4: Generate evidence-based answer
        print("Step 4: Generating evidence-based answer with LLM...")
        try:
            answer = llm_client.answer_question(unique_chunks, question)
        except Exception as e:
            answer = f"Error generating answer: {str(e)}"
        
        return answer, unique_chunks, sub_queries
    
    def _filter_chunks(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """
        Filter chunks based on similarity score and keyword matching.
        
        Args:
            query: User's query/sub-query
            chunks: List of retrieved chunks
            
        Returns:
            Filtered list of chunks
        """
        filtered = []
        
        # Extract keywords from query (simple stopword removal)
        stopwords = {'what', 'how', 'when', 'where', 'why', 'who', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'any', 'all', 'many', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'show', 'tell', 'me', 'about', 'find', 'get'}
        query_words = [w.lower().strip('?!.,') for w in query.split() if w.lower().strip('?!.,') not in stopwords and len(w) > 2]
        
        # Stemming-lite: Handle common plurals/suffixes to avoid literal mismatch
        stemmed_keywords = []
        for kw in query_words:
            stemmed_keywords.append(kw)
            if kw.endswith('s') and len(kw) > 3: stemmed_keywords.append(kw[:-1]) # errors -> error
            if kw.endswith('es') and len(kw) > 4: stemmed_keywords.append(kw[:-2]) # processes -> process
            if kw.endswith('ing') and len(kw) > 5: stemmed_keywords.append(kw[:-3]) # starting -> start

        print(f"Filter keywords: {stemmed_keywords}")

        for chunk in chunks:
            score = chunk.get('score', 1.0)
            
            # Check 1: Similarity Threshold
            if score < SIMILARITY_THRESHOLD:
                continue
            
            # Check 2: Strong Semantic Match
            # If similarity is very high, skip keyword check (AI knows best)
            if score > 0.55:
                filtered.append(chunk)
                continue

            # Check 3: Keyword overlap (if query has significant words)
            if stemmed_keywords:
                chunk_text_lower = chunk['text'].lower()
                has_keyword = any(kw in chunk_text_lower for kw in stemmed_keywords)
                if not has_keyword:
                    continue
            
            filtered.append(chunk)
            
        return filtered

    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Deduplicate chunks based on chunk_id.
        Preserves order and keeps the first occurrence.
        
        Args:
            chunks: List of chunks (may contain duplicates)
            
        Returns:
            List of unique chunks
        """
        seen_ids = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id', chunk.get('id', None))
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_chunks.append(chunk)
            elif not chunk_id:
                # No ID, keep it anyway
                unique_chunks.append(chunk)
        
        return unique_chunks
    
    def list_collections(self) -> List[str]:
        """List all available collections."""
        return self.vector_db.list_collections()
        
    def switch_collection(self, collection_name: str):
        """Switch the active collection."""
        self.vector_db.switch_collection(collection_name)

    def clear_database(self):
        """Clear all data from the vector database."""
        self.vector_db.clear()
        print("Vector database cleared")
    
    def get_stats(self) -> Dict:
        """Get current pipeline statistics."""
        return {
            "total_vectors": self.vector_db.count(),
            "embedding_model": self.embedding_model.model.get_sentence_embedding_dimension(),
            "device": self.embedding_model.device
        }

    def get_usage_stats(self) -> Dict:
        """Get LLM token usage statistics for the last run."""
        return self.llm_client.usage_stats
