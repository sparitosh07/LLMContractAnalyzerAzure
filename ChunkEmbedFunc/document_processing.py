"""Document processing classes similar to AzureML RAG."""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Iterator
from abc import ABC, abstractmethod

from langchain_text_splitters import Document, split_documents_with_langchain
from utils.tokens import adjust_chunk_size_for_prefix, estimate_tokens


@dataclass
class DocumentSource:
    """Document source information."""
    path: Union[str, Path]
    filename: str
    url: str
    mtime: float
    content_type: Optional[str] = None
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata dictionary."""
        return {
            "filename": self.filename,
            "url": self.url,
            "mtime": self.mtime,
            "content_type": self.content_type,
            "file_extension": Path(self.filename).suffix.lower()
        }


@dataclass 
class ChunkedDocument:
    """Document with chunks and metadata."""
    chunks: List[Document]
    source: DocumentSource
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Post initialization processing."""
        # Merge source metadata
        source_metadata = {"source": self.source.get_metadata()}
        self.metadata = {**self.metadata, **source_metadata}
    
    @property
    def page_content(self) -> str:
        """Get combined page content."""
        return "\n\n".join([chunk.page_content for chunk in self.chunks])
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get document metadata."""
        return self.metadata
    
    def flatten(self) -> List[Document]:
        """Flatten chunks with chunk IDs."""
        flattened = []
        for i, chunk in enumerate(self.chunks):
            chunk.metadata = {**chunk.metadata, **self.metadata}
            chunk.metadata["source"]["chunk_id"] = str(i)
            flattened.append(chunk)
        return flattened


class BaseDocumentLoader(ABC):
    """Base class for document loaders."""
    
    def __init__(self, content: str, document_source: DocumentSource, metadata: Dict[str, Any]):
        """Initialize loader."""
        self.content = content
        self.document_source = document_source
        self.metadata = metadata
    
    def load_chunked_document(self) -> ChunkedDocument:
        """Load content into ChunkedDocument."""
        # Ensure source title is set
        if "source" in self.metadata:
            if "title" not in self.metadata["source"]:
                self.metadata["source"]["title"] = Path(self.document_source.filename).name
        else:
            self.metadata["source"] = {"title": Path(self.document_source.filename).name}
        
        documents = self.load()
        return ChunkedDocument(chunks=documents, source=self.document_source, metadata=self.metadata)
    
    @abstractmethod
    def load(self) -> List[Document]:
        """Load content into Document(s)."""
        pass
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return supported file extensions."""
        return [".txt"]


class TextFileLoader(BaseDocumentLoader):
    """Load text files with intelligent title extraction."""
    
    def load(self) -> List[Document]:
        """Load text content into Document."""
        title, clean_title = self._extract_title(self.content, self.document_source.filename)
        
        # Update metadata with extracted title
        self.metadata = {**self.metadata, "source": {"title": clean_title}}
        
        # Create chunk prefix for better context
        chunk_prefix = title + "\n\n" if title != clean_title else ""
        self.metadata["chunk_prefix"] = chunk_prefix
        
        return [Document(
            page_content=self.content,
            metadata=self.metadata,
            document_id=self.document_source.filename
        )]
    
    def _extract_title(self, text: str, filename: str) -> tuple[str, str]:
        """Extract title from text content."""
        file_extension = Path(filename).suffix.lower()
        
        if file_extension == ".md":
            return self._extract_markdown_title(text, filename)
        elif file_extension == ".py":
            return self._extract_python_title(text, filename)
        else:
            return self._extract_generic_title(text, filename)
    
    def _extract_markdown_title(self, text: str, filename: str) -> tuple[str, str]:
        """Extract title from markdown content."""
        import re
        
        # Look for H1 header
        heading_match = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        if heading_match:
            title = heading_match.group(0).strip()
            clean_title = heading_match.group(1).strip()
            return title, clean_title
        
        # Look for YAML front matter title
        yaml_match = re.search(r"^---\s*\n.*?^title:\s*(.+)$.*?\n---", text, re.MULTILINE | re.DOTALL)
        if yaml_match:
            clean_title = yaml_match.group(1).strip().strip('"').strip("'")
            return f"# {clean_title}", clean_title
        
        return f"Title: {filename}", filename
    
    def _extract_python_title(self, text: str, filename: str) -> tuple[str, str]:
        """Extract title from Python file."""
        import ast
        
        try:
            tree = ast.parse(text)
            docstring = ast.get_docstring(tree)
            if docstring:
                # Use first line of docstring as title
                first_line = docstring.split('\n')[0].strip()
                title = f"{filename}: {first_line}"
                return f"Title: {title}", title
        except Exception:
            pass
        
        return f"Title: {filename}", filename
    
    def _extract_generic_title(self, text: str, filename: str) -> tuple[str, str]:
        """Extract title from generic text."""
        # Look for explicit title line
        for line in text.splitlines()[:10]:  # Check first 10 lines
            if line.lower().startswith("title:"):
                title = line[6:].strip()
                return f"Title: {title}", title
        
        # Use first non-empty line if it looks like a title
        for line in text.splitlines()[:5]:
            line = line.strip()
            if line and len(line) < 100 and not line.endswith('.'):
                return f"Title: {line}", line
        
        return f"Title: {filename}", filename
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return supported extensions."""
        return [".txt", ".md", ".py"]


class MarkdownFileLoader(TextFileLoader):
    """Specialized loader for Markdown files."""
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        return [".md"]


class PythonFileLoader(TextFileLoader):
    """Specialized loader for Python files."""
    
    @classmethod 
    def supported_extensions(cls) -> List[str]:
        return [".py"]


class DocumentIntelligencePDFLoader(BaseDocumentLoader):
    """Load PDF using existing Document Intelligence results with page metadata."""
    
    def __init__(self, content: str, document_source: DocumentSource, metadata: Dict[str, Any], doc_intel_data: dict):
        """Initialize with Document Intelligence results."""
        super().__init__(content, document_source, metadata)
        self.doc_intel_data = doc_intel_data
    
    def load(self) -> List[Document]:
        """Load using Document Intelligence page data."""
        if not self.doc_intel_data:
            raise ValueError("Document Intelligence results (doc_intel_data) are required for PDF processing")
        
        documents = []
        full_content = self.doc_intel_data.get('content', '')
        pages_data = self.doc_intel_data.get('pages', [])
        
        if not pages_data:
            raise ValueError("No pages found in Document Intelligence results")
        
        for page_info in pages_data:
            # Extract text for this specific page using spans
            if 'spans' not in page_info or not page_info['spans']:
                continue
                
            page_span = page_info['spans'][0]  # Usually one span per page
            start_offset = page_span['offset']
            length = page_span['length']
            page_text = full_content[start_offset:start_offset + length]
            
            # Create metadata with page number
            page_metadata = {**self.metadata}
            if "source" not in page_metadata:
                page_metadata["source"] = {}
            
            page_metadata["source"] = {
                **page_metadata["source"],
                "page_number": page_info['page_number'],
                "title": Path(self.document_source.filename).name
            }
            
            # Create chunk prefix for better context
            chunk_prefix = f"Title: {Path(self.document_source.filename).name} (Page {page_info['page_number']})\n\n"
            page_metadata["chunk_prefix"] = chunk_prefix
            
            documents.append(Document(
                page_content=page_text.strip(),
                metadata=page_metadata,
                document_id=f"{self.document_source.filename}#page{page_info['page_number']}"
            ))
        
        return documents
    
    @classmethod
    def supported_extensions(cls) -> List[str]:
        """Return supported file extensions."""
        return [".pdf"]


class DocumentProcessor:
    """Process documents through crack and chunk pipeline."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_rcts: bool = True,
        encoding_name: str = "cl100k_base"
    ):
        """Initialize document processor."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_rcts = use_rcts
        self.encoding_name = encoding_name
        
        # Initialize document loaders
        self.loaders = {
            ".txt": TextFileLoader,
            ".md": MarkdownFileLoader,
            ".py": PythonFileLoader,
            ".pdf": DocumentIntelligencePDFLoader,
        }
    
    def crack_document(
        self, 
        content: str, 
        filename: str, 
        content_type: Optional[str] = None,
        doc_intel_data: Optional[dict] = None
    ) -> ChunkedDocument:
        """Crack document content into structured format."""
        # Create document source
        source = DocumentSource(
            path=filename,
            filename=filename,
            url=filename,
            mtime=0.0,  # Not available for in-memory content
            content_type=content_type
        )
        
        # Get appropriate loader
        file_extension = Path(filename).suffix.lower()
        loader_class = self.loaders.get(file_extension, TextFileLoader)
        
        # Load document - pass doc_intel_data for PDF files
        if file_extension == ".pdf" and loader_class == DocumentIntelligencePDFLoader:
            if doc_intel_data is None:
                raise ValueError("doc_intel_data is required for PDF processing")
            loader = loader_class(content, source, {}, doc_intel_data)
        else:
            loader = loader_class(content, source, {})
            
        return loader.load_chunked_document()
    
    def chunk_document(self, chunked_document: ChunkedDocument, activity_logger=None) -> ChunkedDocument:
        """Chunk document using LangChain text splitters (matches AzureML RAG approach)."""
        if not chunked_document.chunks:
            return chunked_document
        
        # Get chunk prefix and adjust chunk size (matches AzureML RAG logic)
        document_metadata = chunked_document.get_metadata()
        chunk_prefix = document_metadata.get("chunk_prefix", "")
        
        adjusted_chunk_size = self.chunk_size
        if chunk_prefix:
            adjusted_chunk_size = adjust_chunk_size_for_prefix(
                self.chunk_size, chunk_prefix, self.encoding_name
            )
        
        # Prepare splitter arguments (matches AzureML RAG)
        splitter_args = {
            "chunk_size": adjusted_chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "use_rcts": self.use_rcts,
            "use_nltk": False,  # Can be configured via environment
        }
        
        # Use LangChain-based document splitting
        all_chunks = split_documents_with_langchain(
            chunked_document.chunks, 
            splitter_args,
            activity_logger=activity_logger
        )
        
        # Update metadata for all chunks
        for i, chunk in enumerate(all_chunks):
            chunk.metadata = {**chunk.metadata, **document_metadata}
            if "chunk_prefix" in chunk.metadata:
                del chunk.metadata["chunk_prefix"]
            
            # Set document ID  
            chunk.document_id = f"{chunked_document.source.filename}#{i}"
        
        # Update chunked document
        chunked_document.chunks = all_chunks
        if "chunk_prefix" in chunked_document.metadata:
            del chunked_document.metadata["chunk_prefix"]
            
        return chunked_document
    
    def process_document(
        self, 
        content: str, 
        filename: str, 
        content_type: Optional[str] = None,
        activity_logger=None,
        doc_intel_data: Optional[dict] = None
    ) -> ChunkedDocument:
        """Full processing pipeline: crack and chunk (matches AzureML RAG workflow)."""
        # Step 1: Crack document (extract and structure content)
        chunked_doc = self.crack_document(content, filename, content_type, doc_intel_data)
        
        # Step 2: Chunk document using LangChain text splitters
        chunked_doc = self.chunk_document(chunked_doc, activity_logger)
        
        return chunked_doc
    
    def get_processing_stats(self, chunked_document: ChunkedDocument) -> Dict[str, Any]:
        """Get processing statistics."""
        total_tokens = sum(
            estimate_tokens(chunk.page_content, self.encoding_name) 
            for chunk in chunked_document.chunks
        )
        
        return {
            "total_chunks": len(chunked_document.chunks),
            "total_tokens": total_tokens,
            "avg_tokens_per_chunk": total_tokens / len(chunked_document.chunks) if chunked_document.chunks else 0,
            "source_file": chunked_document.source.filename,
            "file_extension": Path(chunked_document.source.filename).suffix.lower(),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }