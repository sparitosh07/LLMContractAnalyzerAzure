"""Token counting utilities similar to AzureML RAG."""

import os
import tempfile
import tiktoken
from typing import Callable, Optional
from contextlib import contextmanager


@contextmanager
def tiktoken_cache_dir():
    """Create a temporary cache directory for tiktoken."""
    with tempfile.TemporaryDirectory() as temp_dir:
        old_cache_dir = os.environ.get("TIKTOKEN_CACHE_DIR")
        os.environ["TIKTOKEN_CACHE_DIR"] = temp_dir
        try:
            yield temp_dir
        finally:
            if old_cache_dir is not None:
                os.environ["TIKTOKEN_CACHE_DIR"] = old_cache_dir
            elif "TIKTOKEN_CACHE_DIR" in os.environ:
                del os.environ["TIKTOKEN_CACHE_DIR"]


def token_length_function(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
    """
    Create a token length function using tiktoken.
    
    Args:
        encoding_name: The encoding to use for token counting
        
    Returns:
        Function that takes text and returns token count
    """
    def _token_length(text: str) -> int:
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            return len(encoding.encode(text, disallowed_special=()))
        except Exception:
            # Fallback to character-based estimation if tiktoken fails
            return len(text) // 4  # Rough approximation: 4 chars per token
    
    return _token_length


def estimate_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Text to count tokens for
        encoding_name: Encoding to use
        
    Returns:
        Estimated token count
    """
    token_fn = token_length_function(encoding_name)
    return token_fn(text)


def adjust_chunk_size_for_prefix(
    chunk_size: int, 
    prefix: str, 
    encoding_name: str = "cl100k_base",
    min_chunk_size: int = 100
) -> int:
    """
    Adjust chunk size accounting for prefix tokens.
    
    Args:
        chunk_size: Original chunk size in tokens
        prefix: Prefix text that will be added to chunks
        encoding_name: Token encoding to use
        min_chunk_size: Minimum allowed chunk size
        
    Returns:
        Adjusted chunk size
    """
    if not prefix.strip():
        return chunk_size
    
    prefix_tokens = estimate_tokens(prefix, encoding_name)
    
    if prefix_tokens > chunk_size // 2:
        # If prefix is too large, truncate it and use minimal adjustment
        adjusted_size = max(min_chunk_size, chunk_size - (chunk_size // 4))
    else:
        adjusted_size = max(min_chunk_size, chunk_size - prefix_tokens)
    
    return adjusted_size