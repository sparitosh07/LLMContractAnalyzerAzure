"""
LangChain-based text splitters matching AzureML RAG implementation.
Uses actual LangChain text splitters instead of custom implementations.
"""

import copy
import re
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass

# Import LangChain text splitters - same ones used by AzureML RAG
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter, 
    MarkdownTextSplitter,
    NLTKTextSplitter,
    Language
)
from langchain_core.documents import Document as LangChainDocument

from .utils.tokens import token_length_function, tiktoken_cache_dir


@dataclass  
class Document:
    """Document compatible with our existing code."""
    page_content: str
    metadata: Dict[str, Any]
    document_id: Optional[str] = None
    
    @classmethod
    def from_langchain(cls, lc_doc: LangChainDocument, document_id: Optional[str] = None):
        """Create Document from LangChain Document."""
        return cls(
            page_content=lc_doc.page_content,
            metadata=lc_doc.metadata,
            document_id=document_id
        )
    
    def to_langchain(self) -> LangChainDocument:
        """Convert to LangChain Document."""
        return LangChainDocument(
            page_content=self.page_content,
            metadata=self.metadata
        )


class MarkdownHeaderTextSplitter:
    """
    Custom Markdown splitter that preserves header hierarchy.
    This matches the custom implementation in AzureML RAG.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        length_function: Optional[Callable[[str], int]] = None,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        **kwargs
    ):
        """Initialize markdown header splitter."""
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function or len
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images
        
        # Create sub-splitter for large sections using LangChain
        with tiktoken_cache_dir():
            self._sub_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name="cl100k_base",
                **kwargs
            )
    
    @classmethod
    def from_tiktoken_encoder(
        cls,
        encoding_name: str = "cl100k_base",
        **kwargs
    ):
        """Create splitter with tiktoken encoder."""
        with tiktoken_cache_dir():
            return cls(
                length_function=token_length_function(encoding_name),
                **kwargs
            )
    
    def split_text(self, text: str) -> List[str]:
        """Split markdown text by headers."""
        blocks = self._parse_markdown_blocks(text)
        chunks = []
        
        for block in blocks:
            nested_headers = self._get_nested_headers(block)
            content = f"{nested_headers}\n{block['content']}" if nested_headers else block['content']
            
            if self._length_function(content) > self._chunk_size:
                # Use LangChain's TokenTextSplitter for large sections
                sub_chunks = self._sub_splitter.split_text(block['content'])
                for sub_chunk in sub_chunks:
                    full_chunk = f"{nested_headers}\n{sub_chunk}" if nested_headers else sub_chunk
                    chunks.append(full_chunk)
            else:
                chunks.append(content)
                
        return chunks
    
    def create_documents(
        self, 
        texts: List[str], 
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create documents from texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        
        for i, text in enumerate(texts):
            chunks = self.split_text(text)
            for chunk in chunks:
                doc = Document(
                    page_content=chunk,
                    metadata=copy.deepcopy(_metadatas[i])
                )
                documents.append(doc)
                
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        return self.create_documents(texts, metadatas)
    
    def _parse_markdown_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Parse markdown into blocks with header hierarchy."""
        # Remove hyperlinks and images if requested
        if self._remove_hyperlinks:
            text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        if self._remove_images:
            text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', '', text)
            
        # Split by headers
        blocks = re.split(r'^(#+\s.*?)$', text, flags=re.MULTILINE)
        blocks = [b.strip() for b in blocks if b.strip()]
        
        parsed_blocks = []
        header_stack = []
        
        if not blocks[0].startswith('#'):
            # Content before first header
            parsed_blocks.append({
                'header': None,
                'content': blocks[0],
                'level': 0,
                'parents': []
            })
            blocks = blocks[1:]
        
        for i in range(0, len(blocks), 2):
            if i >= len(blocks):
                break
                
            header = blocks[i]
            content = blocks[i + 1] if i + 1 < len(blocks) else ""
            
            level = header.count('#')
            
            # Update header stack
            while header_stack and header_stack[-1]['level'] >= level:
                header_stack.pop()
                
            parents = [h['header'] for h in header_stack]
            header_stack.append({'header': header, 'level': level})
            
            parsed_blocks.append({
                'header': header,
                'content': content,
                'level': level,
                'parents': parents
            })
            
        return parsed_blocks
    
    def _get_nested_headers(self, block: Dict[str, Any]) -> str:
        """Get nested header string for a block."""
        headers = []
        if block['parents']:
            headers.extend(block['parents'])
        if block['header']:
            headers.append(block['header'])
            
        return '\n'.join(headers) if headers else ""


def get_langchain_splitter(
    file_extension: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    use_rcts: bool = True,
    use_nltk: bool = False,
    **kwargs
) -> Any:
    """
    Get appropriate LangChain text splitter for file extension.
    This matches the exact logic from AzureML RAG chunking.py
    """
    
    # Handle Python code - use LangChain RecursiveCharacterTextSplitter
    if file_extension == ".py":
        with tiktoken_cache_dir():
            return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name="cl100k_base",
                separators=RecursiveCharacterTextSplitter.get_separators_for_language(Language.PYTHON),
                is_separator_regex=True,
                disallowed_special=(),
                **kwargs
            )
    
    # Handle NLTK sentence splitting - use LangChain NLTKTextSplitter
    if use_nltk:
        return NLTKTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=token_length_function(),
            **kwargs
        )
    
    # Handle text formats - use LangChain TokenTextSplitter
    text_formats = [".txt", ".pdf", ".ppt", ".pptx", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".json"]
    if file_extension == ".txt" or file_extension in text_formats:
        with tiktoken_cache_dir():
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name="cl100k_base",
                length_function=token_length_function(),
                disallowed_special=(),
                **kwargs
            )
    
    # Handle HTML - use LangChain TokenTextSplitter  
    elif file_extension in [".html", ".htm"]:
        with tiktoken_cache_dir():
            return TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                encoding_name="cl100k_base",
                length_function=token_length_function(),
                disallowed_special=(),
                **kwargs
            )
    
    # Handle Markdown - use LangChain MarkdownTextSplitter or custom header splitter
    elif file_extension == ".md":
        if use_rcts:
            # Use LangChain's MarkdownTextSplitter
            with tiktoken_cache_dir():
                return MarkdownTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    encoding_name="cl100k_base",
                    disallowed_special=(),
                    **kwargs
                )
        else:
            # Use custom header-aware splitter (matches AzureML RAG)
            with tiktoken_cache_dir():
                return MarkdownHeaderTextSplitter.from_tiktoken_encoder(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    encoding_name="cl100k_base",
                    remove_hyperlinks=True,
                    remove_images=True,
                    disallowed_special=(),
                    **kwargs
                )
    
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def split_documents_with_langchain(
    documents: List[Document],
    splitter_args: Dict[str, Any],
    activity_logger=None
) -> List[Document]:
    """
    Split documents using LangChain text splitters.
    This matches the split_documents function from AzureML RAG.
    """
    if not documents:
        return []
    
    # Get file extension from first document
    first_doc = documents[0]
    filename = first_doc.metadata.get("source", {}).get("filename", "document.txt")
    file_extension = "." + filename.split(".")[-1].lower() if "." in filename else ".txt"
    
    # Get appropriate LangChain splitter
    splitter = get_langchain_splitter(file_extension, **splitter_args)
    
    all_chunks = []
    
    for doc in documents:
        # Handle chunk prefix (from AzureML RAG logic)
        chunk_prefix = doc.metadata.get("chunk_prefix", "")
        
        # Convert to LangChain document and split
        lc_doc = doc.to_langchain()
        split_docs = splitter.split_documents([lc_doc])
        
        # Convert back and add prefixes
        for i, lc_chunk in enumerate(split_docs):
            chunk = Document.from_langchain(lc_chunk, f"{filename}#{i}")
            
            # Add chunk prefix if available
            if chunk_prefix:
                chunk.page_content = chunk_prefix.strip() + "\n\n" + chunk.page_content
            
            # Normalize line endings
            chunk.page_content = chunk.page_content.replace('\r\n', '\n').replace('\r', '\n')
            
            # Clean up metadata
            if "chunk_prefix" in chunk.metadata:
                del chunk.metadata["chunk_prefix"]
            
            all_chunks.append(chunk)
    
    if activity_logger:
        activity_logger.set_activity_info("langchain_chunks", len(all_chunks))
    
    return all_chunks


# File extension to splitter mapping (matches AzureML RAG)
FILE_EXTENSION_SPLITTERS = {
    # Plain text
    ".txt": lambda **kwargs: get_langchain_splitter(".txt", **kwargs),
    ".md": lambda **kwargs: get_langchain_splitter(".md", **kwargs),
    ".html": lambda **kwargs: get_langchain_splitter(".html", **kwargs),
    ".htm": lambda **kwargs: get_langchain_splitter(".htm", **kwargs),
    ".csv": lambda **kwargs: get_langchain_splitter(".csv", **kwargs),
    ".json": lambda **kwargs: get_langchain_splitter(".json", **kwargs),
    
    # Encoded text (treated as text after parsing)
    ".pdf": lambda **kwargs: get_langchain_splitter(".pdf", **kwargs),
    ".ppt": lambda **kwargs: get_langchain_splitter(".ppt", **kwargs),
    ".pptx": lambda **kwargs: get_langchain_splitter(".pptx", **kwargs),
    ".doc": lambda **kwargs: get_langchain_splitter(".doc", **kwargs),
    ".docx": lambda **kwargs: get_langchain_splitter(".docx", **kwargs),
    ".xls": lambda **kwargs: get_langchain_splitter(".xls", **kwargs),
    ".xlsx": lambda **kwargs: get_langchain_splitter(".xlsx", **kwargs),
    
    # Code
    ".py": lambda **kwargs: get_langchain_splitter(".py", **kwargs),
}