"""
Vector database module using Qdrant.
Handles storage and retrieval of log chunk embeddings.
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from typing import List, Dict
import numpy as np

from .config import COLLECTION_NAME, VECTOR_SIZE


class VectorDB:
    """
    Wrapper for Qdrant vector database operations.
    Stores chunk embeddings and enables semantic search.
    """
    
    def __init__(self):
        """Initialize Qdrant client in disk-backed mode."""
        print("Initializing Qdrant vector database (disk-backed: ./qdrant_data)...")
        # Fix 4: Use disk storage instead of :memory: to prevent RAM explosion
        try:
            self.client = QdrantClient(path="./qdrant_data")
        except Exception as e:
            err_msg = str(e).lower()
            # Case 1: Database is strictly corrupt (Pydantic validation errors)
            if "validation" in err_msg or "valueerror" in err_msg:
                print(f"WARNING: Qdrant database corruption detected: {str(e)}")
                print("Attempting to recover by wiping corrupt data...")
                import shutil
                import os
                import time
                
                # Close any lingering connections if possible (rarely works if obj creation failed but good hygiene)
                if hasattr(self, 'client') and self.client:
                    try: self.client.close()
                    except: pass
                
                # Retry deletion to handle transient locks
                max_retries = 3
                for i in range(max_retries):
                    try:
                        if os.path.exists("./qdrant_data"):
                            shutil.rmtree("./qdrant_data")
                        break
                    except PermissionError:
                        if i < max_retries - 1:
                            time.sleep(1)
                        else:
                            raise RuntimeError("Could not wipe corrupt database due to file lock. Please close all Python processes.")
                            
                print("Corrupt data cleared. Re-initializing...")
                self.client = QdrantClient(path="./qdrant_data")
            
            # Case 2: Database is locked by another process (e.g. another Streamlit tab)
            elif "process cannot access" in err_msg or "lock" in err_msg or "permissionerror" in err_msg:
                print("CRITICAL: Database is locked by another process.")
                raise RuntimeError(
                    "Database is locked! "
                    "This usually means another Streamlit instance or terminal is running. "
                    "Please KILL ALL terminal instances and restart."
                ) from e
            
            # Case 3: Other errors
            else:
                raise e
            
        # Default initialization
        self.collection_name = COLLECTION_NAME
        
        # Restore last active collection from persistence file
        try:
            if os.path.exists(".active_collection"):
                with open(".active_collection", "r") as f:
                    saved_name = f.read().strip()
                    if saved_name:
                        self.collection_name = saved_name
                        print(f"Persisted active collection restored: {self.collection_name}")
        except Exception:
            pass

        self._collection_initialized = False
    
    def initialize(self):
        """Create the collection if it doesn't exist."""
        if self._collection_initialized:
            return
        
        # Check if collection exists
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Create collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection: {self.collection_name}")
        else:
            print(f"Collection {self.collection_name} already exists")
        
        self._collection_initialized = True

    def list_collections(self) -> List[str]:
        """List all available collections in Qdrant."""
        try:
            collections = self.client.get_collections().collections
            return [c.name for c in collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []

    def switch_collection(self, new_collection_name: str):
        """Switch the active collection."""
        if not new_collection_name:
            raise ValueError("Collection name cannot be empty")
        self.collection_name = new_collection_name
        self._collection_initialized = False # Will trigger checks on next operation
        
        # Persist selection to file
        try:
            with open(".active_collection", "w") as f:
                f.write(new_collection_name)
        except Exception as e:
            print(f"Warning: Could not persist active collection: {e}")
            
        print(f"Switched to collection: {self.collection_name}")

    def get_collection_info(self):
        """Get current collection status."""
        # Don't auto-initialize here, just check status
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "exists": True, 
                "count": info.points_count,
                "status": info.status,
                "name": self.collection_name
            }
        except Exception as e:
            return {
                "exists": False, 
                "count": 0, 
                "error": str(e),
                "name": self.collection_name
            }
    
    def store_chunks(self, chunks: List[Dict], embeddings: np.ndarray, source_name: str):
        """
        Store chunk embeddings with metadata in Qdrant.
        
        Args:
            chunks: List of chunk dictionaries with metadata
            embeddings: NumPy array of embeddings
            source_name: Name of the source file/input
        """
        self.initialize()
        
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        # Prepare points for insertion
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            # Build payload with enhanced metadata
            payload = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": source_name,
            }
            
            # Add enhanced metadata if available (new format)
            if "start_time" in chunk:
                payload["start_time"] = chunk.get("start_time")
                payload["end_time"] = chunk.get("end_time")
                payload["os_hint"] = chunk.get("os_hint", "unknown")
                payload["line_count"] = chunk.get("line_count", 0)
                payload["source_file"] = chunk.get("source_file", source_name)
            
            # Backward compatibility: old format (start_line, end_line)
            if "start_line" in chunk:
                payload["start_line"] = chunk["start_line"]
                payload["end_line"] = chunk["end_line"]
            
            point = PointStruct(
                id=idx,
                vector=embedding.tolist(),
                payload=payload
            )
            points.append(point)
        
        # Insert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        print(f"Stored {len(chunks)} chunks in vector database")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Search for similar chunks given a query embedding.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            
        Returns:
            List of dictionaries containing chunk metadata and similarity scores
        """
        self.initialize()
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k
        )
        
        # Format results
        retrieved_chunks = []
        for result in results:
            chunk_data = {
                "chunk_id": result.payload["chunk_id"],
                "text": result.payload["text"],
                "source": result.payload["source"],
                "score": result.score
            }
            
            # Add enhanced metadata if available
            if "start_time" in result.payload:
                chunk_data["start_time"] = result.payload["start_time"]
                chunk_data["end_time"] = result.payload["end_time"]
                chunk_data["os_hint"] = result.payload.get("os_hint", "unknown")
                chunk_data["line_count"] = result.payload.get("line_count", 0)
                chunk_data["source_file"] = result.payload.get("source_file", "unknown")
            
            # Backward compatibility: old format
            if "start_line" in result.payload:
                chunk_data["start_line"] = result.payload["start_line"]
                chunk_data["end_line"] = result.payload["end_line"]
            
            retrieved_chunks.append(chunk_data)
        
        return retrieved_chunks
    
    def clear(self):
        """Clear all data from the collection."""
        if self._collection_initialized:
            self.client.delete_collection(collection_name=self.collection_name)
            self._collection_initialized = False
            print(f"Cleared collection: {self.collection_name}")
    
    def count(self) -> int:
        """Get the number of vectors in the collection."""
        self.initialize()
        
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)
            return collection_info.points_count
        except Exception:
            return 0
