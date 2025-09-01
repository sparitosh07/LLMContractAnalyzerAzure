"""PDF processing using Azure Document Intelligence API with layout analysis."""

import io
import time
import base64
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError, ServiceRequestError

from .config import DocumentIntelligenceConfig
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentPage:
    """Represents a single page from document analysis."""
    page_number: int
    content: str
    width: float
    height: float
    angle: float
    unit: str
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get page metadata."""
        return {
            "page_number": self.page_number,
            "dimensions": {
                "width": self.width,
                "height": self.height,
                "unit": self.unit
            },
            "angle": self.angle
        }


@dataclass
class DocumentAnalysisResult:
    """Complete document analysis result."""
    content: str
    pages: List[DocumentPage]
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]
    
    def get_full_text(self) -> str:
        """Get complete document text."""
        return self.content
    
    def get_page_content(self, page_number: int) -> Optional[str]:
        """Get content for specific page."""
        for page in self.pages:
            if page.page_number == page_number:
                return page.content
        return None


class PDFProcessor:
    """Modular PDF processor using Azure Document Intelligence with layout analysis."""
    
    def __init__(self, config: DocumentIntelligenceConfig):
        """Initialize PDF processor with Document Intelligence config."""
        logger.info("Initializing PDF processor with Document Intelligence API")
        
        self.config = config
        self.client = DocumentAnalysisClient(
            endpoint=config.endpoint,
            credential=AzureKeyCredential(config.api_key)
        )
        
        logger.info(f"PDF processor initialized with endpoint: {config.endpoint}")
    
    def process_pdf_bytes(
        self, 
        pdf_bytes: bytes, 
        filename: str = "document.pdf",
        use_layout_model: bool = True
    ) -> DocumentAnalysisResult:
        """
        Process PDF bytes using Document Intelligence API.
        
        Args:
            pdf_bytes: PDF file as bytes
            filename: Original filename for metadata
            use_layout_model: Whether to use layout model for better structure extraction
            
        Returns:
            DocumentAnalysisResult with extracted text and metadata
        """
        logger.info(f"Starting PDF processing for {filename} ({len(pdf_bytes)} bytes)")
        
        model_id = "prebuilt-layout" if use_layout_model else "prebuilt-read"
        logger.info(f"Using Document Intelligence model: {model_id}")
        
        start_time = time.time()
        
        try:
            # Start document analysis
            logger.info("Submitting document for analysis...")
            poller = self.client.begin_analyze_document(
                model_id=model_id,
                document=io.BytesIO(pdf_bytes)
            )
            
            logger.info("Waiting for analysis to complete...")
            result = poller.result()
            
            processing_time = time.time() - start_time
            logger.info(f"Document analysis completed in {processing_time:.2f} seconds")
            
            # Extract content and metadata
            return self._extract_content_and_metadata(result, filename, processing_time, use_layout_model)
            
        except HttpResponseError as e:
            logger.error(f"HTTP error during PDF processing: {e.status_code} - {e.message}")
            raise Exception(f"Document Intelligence API error: {e.message}")
        except ServiceRequestError as e:
            logger.error(f"Service request error during PDF processing: {str(e)}")
            raise Exception(f"Document Intelligence service error: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error during PDF processing: {str(e)}")
            raise Exception(f"PDF processing failed: {str(e)}")
    
    def process_pdf_base64(
        self, 
        pdf_base64: str, 
        filename: str = "document.pdf",
        use_layout_model: bool = True
    ) -> DocumentAnalysisResult:
        """
        Process base64-encoded PDF using Document Intelligence API.
        
        Args:
            pdf_base64: PDF file as base64 string
            filename: Original filename for metadata
            use_layout_model: Whether to use layout model
            
        Returns:
            DocumentAnalysisResult with extracted text and metadata
        """
        logger.info(f"Converting base64 PDF to bytes for processing: {filename}")
        
        try:
            pdf_bytes = base64.b64decode(pdf_base64)
            logger.info(f"Successfully decoded base64 PDF ({len(pdf_bytes)} bytes)")
            return self.process_pdf_bytes(pdf_bytes, filename, use_layout_model)
        except Exception as e:
            logger.error(f"Failed to decode base64 PDF: {str(e)}")
            raise Exception(f"Base64 decoding failed: {str(e)}")
    
    def _extract_content_and_metadata(
        self, 
        analysis_result, 
        filename: str,
        processing_time: float,
        use_layout_model: bool
    ) -> DocumentAnalysisResult:
        """Extract content and metadata from Document Intelligence result."""
        logger.info("Extracting content and metadata from analysis result")
        
        # Extract full content
        full_content = analysis_result.content
        logger.info(f"Extracted {len(full_content)} characters of text content")
        
        # Extract page information
        pages = []
        for page in analysis_result.pages:
            logger.debug(f"Processing page {page.page_number}")
            
            # Extract page content (lines and words)
            page_content_parts = []
            
            # Extract lines if available (layout model)
            if hasattr(page, 'lines') and page.lines:
                logger.debug(f"Page {page.page_number}: Found {len(page.lines)} lines")
                for line in page.lines:
                    page_content_parts.append(line.content)
            # Fall back to words if lines not available
            elif hasattr(page, 'words') and page.words:
                logger.debug(f"Page {page.page_number}: Found {len(page.words)} words")
                for word in page.words:
                    page_content_parts.append(word.content)
            
            page_content = '\n'.join(page_content_parts)
            
            doc_page = DocumentPage(
                page_number=page.page_number,
                content=page_content,
                width=page.width,
                height=page.height,
                angle=page.angle,
                unit=page.unit
            )
            pages.append(doc_page)
            
            logger.debug(f"Page {page.page_number}: Extracted {len(page_content)} characters")
        
        logger.info(f"Successfully processed {len(pages)} pages")
        
        # Compile metadata
        metadata = {
            "filename": filename,
            "file_extension": ".pdf",
            "total_pages": len(pages),
            "model_used": "prebuilt-layout" if use_layout_model else "prebuilt-read",
            "api_version": self.config.api_version,
            "extraction_method": "azure_document_intelligence"
        }
        
        # Compile processing stats
        processing_stats = {
            "processing_time_seconds": processing_time,
            "total_pages": len(pages),
            "total_characters": len(full_content),
            "model_used": metadata["model_used"],
            "extraction_successful": True
        }
        
        logger.info(f"PDF processing completed: {processing_stats}")
        
        return DocumentAnalysisResult(
            content=full_content,
            pages=pages,
            metadata=metadata,
            processing_stats=processing_stats
        )
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported document formats."""
        return [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
    
    def validate_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Validate document before processing."""
        logger.info(f"Validating document: {filename}")
        
        file_extension = Path(filename).suffix.lower()
        
        # Check file extension
        if file_extension not in self.get_supported_formats():
            logger.warning(f"Unsupported file extension: {file_extension}")
            return {
                "valid": False,
                "error": f"Unsupported file format: {file_extension}",
                "supported_formats": self.get_supported_formats()
            }
        
        # Check file size (Document Intelligence has limits)
        max_size_mb = 500  # Document Intelligence limit
        file_size_mb = len(file_bytes) / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            logger.warning(f"File too large: {file_size_mb:.2f}MB (max: {max_size_mb}MB)")
            return {
                "valid": False,
                "error": f"File too large: {file_size_mb:.2f}MB (maximum: {max_size_mb}MB)",
                "file_size_mb": file_size_mb,
                "max_size_mb": max_size_mb
            }
        
        logger.info(f"Document validation passed: {filename} ({file_size_mb:.2f}MB)")
        return {
            "valid": True,
            "file_size_mb": file_size_mb,
            "file_extension": file_extension
        }


def create_pdf_processor(config: DocumentIntelligenceConfig) -> PDFProcessor:
    """Factory function to create PDF processor instance."""
    logger.info("Creating PDF processor instance")
    return PDFProcessor(config)