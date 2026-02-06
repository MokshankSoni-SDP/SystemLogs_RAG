import os
# Fix 3: Disable tokenizer parallelism to reduce CPU/RAM usage on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import gc
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from .config import EMBEDDING_MODEL

class EmbeddingModel:
    """
    Wrapper for embedding model that handles text-to-vector conversion.
    Uses nomic-ai/nomic-embed-text-v1.5 via sentence-transformers.
    """
    
    def __init__(self):
        """Initialize the embedding model with strict GPU support."""
        # Sanity check for GPU
        print("DEBUG: Checking CUDA availability...")
        print(f"DEBUG: torch.cuda.is_available() = {torch.cuda.is_available()}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.device == "cuda":
            print(f"DEBUG: Found GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**2)
            print(f"DEBUG: VRAM: {vram:.2f} MB")
        else:
            print("WARNING: CUDA NOT AVAILABLE. Running on CPU (Expect slowness/freezes).")
            # Try to force it if user thinks it should be there? 
            # No, if is_available() is False, forcing 'cuda' will crash. 
            # We rely on the diagnostic prints to debug why it's missing.

        print(f"Loading embedding model on {self.device}...")
        
        # Load the model
        self.model = SentenceTransformer(
            EMBEDDING_MODEL, 
            device=self.device, 
            trust_remote_code=True
        )
        print(f"Embedding model loaded successfully on {self.device}")
    
    def encode(self, texts: List[str], batch_size: int = 4) -> np.ndarray:
        """
        Generate embeddings for a list of texts with explicit batching.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 4 for GTX 1650 safety)
            
        Returns:
            NumPy array of embeddings
        """
        if not texts:
            return np.array([])
            
        # Fix 2: Explicit batching loop to manage memory
        all_embeddings = []
        total_texts = len(texts)
        
        print(f"Starting embedding generation for {total_texts} chunks (Batch size: {batch_size})")
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i : i + batch_size]
            
            # Encode batch
            # We disable internal progress bar since we might loop manually, 
            # or we rely on the parent pipeline to show status.
            batch_embeddings = self.model.encode(
                batch,
                batch_size=batch_size, # internal batching matches our explicit loop
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            all_embeddings.append(batch_embeddings)
            
            # Optional: Aggressive GC to keep RAM clean
            # gc.collect() 
            
        # Concatenate all batches
        if all_embeddings:
            final_embeddings = np.vstack(all_embeddings)
            print(f"Completed embedding generation. Shape: {final_embeddings.shape}")
            return final_embeddings
        else:
            return np.array([])
