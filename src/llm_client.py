"""
LLM client module using Groq API.
Handles query planning and question answering with proper separation of concerns.
"""
# CRITICAL FIX: Remove ALL proxy environment variables BEFORE any imports
# Windows can inject these at system level, causing httpx/groq to fail
import os

_PROXY_VARS = [
    "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
    "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy"
]

print(f"[LLM] Clearing proxy environment variables: {_PROXY_VARS}")
for proxy_var in _PROXY_VARS:
    if proxy_var in os.environ:
        print(f"[LLM] Removed {proxy_var}={os.environ[proxy_var]}")
        os.environ.pop(proxy_var)

from groq import Groq, RateLimitError
from typing import List, Dict
import time

from .config import GROQ_API_KEYS, LLM_MODEL


def groq_health_check() -> str:
    """
    Minimal Groq client health check.
    Tests if the client initializes and can make a basic API call.
    Uses the first available key.
    """
    if not GROQ_API_KEYS:
        return "✗ FAILED: No API keys found"
        
    try:
        # Try up to all keys
        for key in GROQ_API_KEYS:
            try:
                client = Groq(api_key=key)
                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=5
                )
                return f"✓ Groq client healthy: {resp.choices[0].message.content}"
            except Exception:
                continue
                
        return "✗ Groq client FAILED: All keys failed health check"
    except Exception as e:
        return f"✗ Groq client FAILED: {str(e)}"


class LLMClient:
    """
    Wrapper for Groq API with separate methods for planning and answering.
    Uses llama-3.3-70b-versatile model with optimized temperatures.
    Supports automatic key rotation on RateLimitError.
    """
    
    def __init__(self):
        """Initialize Groq client."""
        if not GROQ_API_KEYS:
            raise ValueError(
                "Please set at least one valid GROQ_API_KEY in .env file. "
                "Get your API key from https://console.groq.com/"
            )
        
        print(f"[LLM] Initializing Groq client with {len(GROQ_API_KEYS)} keys available...")
        self.current_key_idx = 0
        self.model = LLM_MODEL
        self._init_current_client()
        
        # Token usage tracking
        self.usage_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_input_tokens": 0,
            "max_output_tokens": 0,
            "total_calls": 0
        }

    def reset_usage(self):
        """Reset usage statistics for a new query."""
        self.usage_stats = {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "max_input_tokens": 0,
            "max_output_tokens": 0,
            "total_calls": 0
        }

    def get_usage_stats(self) -> Dict:
        """Get current usage statistics."""
        return self.usage_stats

    def _update_usage(self, usage):
        """Update usage stats from an API response usage object."""
        if not usage:
            return
            
        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        
        self.usage_stats["total_input_tokens"] += input_tokens
        self.usage_stats["total_output_tokens"] += output_tokens
        self.usage_stats["max_input_tokens"] = max(self.usage_stats["max_input_tokens"], input_tokens)
        self.usage_stats["max_output_tokens"] = max(self.usage_stats["max_output_tokens"], output_tokens)
        self.usage_stats["total_calls"] += 1
        
    def _init_current_client(self):
        """Initialize or re-initialize client with current key."""
        current_key = GROQ_API_KEYS[self.current_key_idx]
        self.client = Groq(api_key=current_key)
        # Mask key for privacy
        masked_key = f"{current_key[:4]}...{current_key[-4:]}" if len(current_key) > 8 else "***"
        print(f"[LLM] ✓ Client initialized with key {self.current_key_idx + 1}/{len(GROQ_API_KEYS)} ({masked_key})")
        
    def _rotate_key(self) -> bool:
        """
        Switch to the next available API key.
        Returns True if successful, False if all keys exhausted (cycle complete).
        """
        if len(GROQ_API_KEYS) <= 1:
            print("[LLM] Only one key available. Cannot rotate.")
            return False
            
        next_idx = (self.current_key_idx + 1) % len(GROQ_API_KEYS)
        
        # Prevent infinite loops: check if we've cycled back to start (in a single request context)
        # Note: Ideally we'd track this per-request, but for simple rotation this suffices
        # If next is 0, we've looped.
        
        self.current_key_idx = next_idx
        print(f"[LLM] ⚠️ Rate limit or error detected. Rotating to key {self.current_key_idx + 1}...")
        self._init_current_client()
        return True
    
    def plan_query(self, planner_prompt: str) -> str:
        """
        Generate sub-queries for complex questions using LLM as a query planner.
        
        Uses:
        - Temperature: 0.1 (very deterministic, structured output)
        - Max tokens: 512 (sufficient for JSON sub-queries)
        
        Args:
            planner_prompt: Prompt for query planning with JSON format instructions
            
        Returns:
            JSON string with sub-queries or error message
        """
    def plan_query(self, planner_prompt: str) -> str:
        """
        Generate sub-queries for complex questions using LLM as a query planner.
        Retries with key rotation if rate limited.
        """
        max_attempts = len(GROQ_API_KEYS)
        
        for attempt in range(max_attempts):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": "You are a strict JSON-only query planner."},
                        {"role": "user", "content": planner_prompt}
                    ],
                    model=self.model,
                    temperature=0.1,  # Very low for structured output
                    max_tokens=512
                )
                
                # Track usage
                if hasattr(chat_completion, 'usage'):
                    self._update_usage(chat_completion.usage)
                    
                return chat_completion.choices[0].message.content
                
            except RateLimitError as e:
                print(f"[LLM] Rate limit hit during planning (Attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1 and self._rotate_key():
                    time.sleep(1) # Brief pause before retry
                    continue
                else:
                    return f"Error planning query: Rate limit exceeded on all keys."
            except Exception as e:
                # For non-rate-limit errors, we might not want to rotate, or maybe we do?
                # Let's rotate on any API-level error just to be safe/robust
                print(f"[LLM] Error during planning (Attempt {attempt + 1}/{max_attempts}): {e}")
                if "401" in str(e) or "429" in str(e): # Auth or Rate Limit
                    if attempt < max_attempts - 1 and self._rotate_key():
                         time.sleep(1)
                         continue
                return f"Error planning query: {str(e)}"
                
        return "Error planning query: All attempts failed."
    
    def answer_question(self, context_chunks: List[Dict], question: str) -> str:
        """
        Generate an evidence-based answer to a question using retrieved log chunks.
        Retries with key rotation if rate limited.
        """
        # (Content limitation logic omitted for brevity as it's unchanged)
        # Limit chunks to avoid context overflow
        context_chunks = context_chunks[:8]
        
        # Format context from chunks with enhanced metadata
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            # Handle both old and new chunk formats
            start_time = chunk.get('start_time', 'N/A')
            end_time = chunk.get('end_time', 'N/A')
            os_hint = chunk.get('os_hint', 'unknown')
            chunk_id = chunk.get('chunk_id', f'chunk_{i}')
            
            # Format metadata
            metadata = f"Chunk ID: {chunk_id}"
            if start_time != 'N/A':
                metadata += f" | Time: {start_time} to {end_time}"
            if os_hint != 'unknown':
                metadata += f" | OS: {os_hint}"
            
            context_parts.append(
                f"--- Log Chunk {i} ---\n"
                f"{metadata}\n"
                f"{chunk['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Determine context description based on log types
        log_types = set()
        for chunk in context_chunks:
            l_type = chunk.get('log_type', 'system') # Default to system for old chunks
            log_types.add(l_type)
        
        if 'container' in log_types and 'system' in log_types:
            context_desc = "mixed operating system and container logs"
        elif 'container' in log_types:
            context_desc = "container logs (Docker/Kubernetes)"
        else:
            context_desc = "operating system logs (Windows/Linux/macOS)"

        # Build evidence-based answering prompt with confidence requirement
        system_message = (
            f"You are a log analysis assistant analyzing {context_desc}. "
            "Your task is to answer questions based ONLY on the provided log evidence. "
            "CRITICAL RULES:\n"
            "1. Answer ONLY from the evidence provided\n"
            "2. If evidence is insufficient, explicitly say 'Insufficient evidence to answer'\n"
            "3. Cite specific log chunks or timestamps when relevant\n"
            "4. Be explicit about any uncertainty or limitations in the evidence\n"
            "5. End your answer with a confidence level: High / Medium / Low"
        )
        
        user_message = f"""Question: {question}

Log Evidence:
{context}

Instructions:
- Answer the question based solely on the log evidence above
- If the evidence is insufficient, say so clearly
- Include relevant timestamps or chunk references in your answer
- End with your confidence level (High/Medium/Low)

Answer:"""
        
        # Call Groq API with retry logic
        max_attempts = len(GROQ_API_KEYS)
        
        for attempt in range(max_attempts):
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model,
                    temperature=0.3,  
                    max_tokens=4000   
                )
                
                # Track usage
                if hasattr(chat_completion, 'usage'):
                    self._update_usage(chat_completion.usage)
                
                answer = chat_completion.choices[0].message.content
                return answer
                
            except RateLimitError as e:
                print(f"[LLM] Rate limit hit during answering (Attempt {attempt + 1}/{max_attempts}): {e}")
                if attempt < max_attempts - 1 and self._rotate_key():
                    time.sleep(1) # Brief pause before retry
                    continue
                else:
                     return "Error generating answer: Rate limit exceeded on all keys."
            
            except Exception as e:
                print(f"[LLM] Error during answering (Attempt {attempt + 1}/{max_attempts}): {e}")
                if "401" in str(e) or "429" in str(e): # Auth or Rate Limit
                    if attempt < max_attempts - 1 and self._rotate_key():
                         time.sleep(1)
                         continue
                return f"Error generating answer: {str(e)}"
                
        return "Error generating answer: All attempts failed."
