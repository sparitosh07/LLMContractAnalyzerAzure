import logging
import json
import os
import uuid
import base64
import asyncio
import aiohttp
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import azure.functions as func
from openai import AzureOpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, SearchField, SearchFieldDataType, SimpleField, SearchableField, VectorSearch,
    VectorSearchProfile, HnswAlgorithmConfiguration, VectorSearchAlgorithmKind
)
from azure.core.credentials import AzureKeyCredential

from .config import get_app_config, AppConfig
from .document_processing import DocumentProcessor
from .pdf_processor import PDFProcessor, create_pdf_processor
from .local_storage import LocalFileSaver, create_local_file_saver
from .contract_extractor import ContractExtractor, create_contract_extractor, ContractExtractionConfig
from .utils.logging_utils import (
    track_activity, monitor_performance, 
    LoggingConfig, exception_handler, log_processing_stats
)


# Setup logging
LoggingConfig.setup_logging(level=logging.INFO)
logger = LoggingConfig.get_function_logger("process_document")

# Global config (loaded once)
APP_CONFIG: Optional[AppConfig] = None


async def call_contract_extraction_async(
    text_content: str, 
    filename: str, 
    document_id: str,
    extraction_chunk_size: int = 2000,
    extraction_chunk_overlap: int = 400
) -> Optional[Dict[str, Any]]:
    """Call contract extraction function asynchronously"""
    try:
        # Get function URL (assumes running locally on same host)
        function_url = "http://localhost:7071/api/ContractExtractorFunc"
        
        payload = {
            "text_content": text_content,
            "filename": filename,
            "document_id": document_id,
            "extraction_chunk_size": extraction_chunk_size,
            "extraction_chunk_overlap": extraction_chunk_overlap,
            "save_to_local": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(function_url, json=payload, timeout=300) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Contract extraction completed successfully")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Contract extraction failed with status {response.status}: {error_text}")
                    return None
                    
    except Exception as e:
        logger.error(f"Failed to call contract extraction function: {str(e)}")
        return None


def get_app_config_cached() -> AppConfig:
    """Get cached app configuration."""
    global APP_CONFIG
    if APP_CONFIG is None:
        APP_CONFIG = get_app_config()
    return APP_CONFIG


@monitor_performance(logger, "embedding_generation")
def generate_embeddings_batch(
    chunks: List[Dict[str, Any]], 
    openai_client: AzureOpenAI, 
    config: AppConfig
) -> List[Dict[str, Any]]:
    """
    Generate embeddings for text chunks using Azure OpenAI with batching and retry logic.
    """
    embedded_chunks = []
    batch_size = min(config.processing.batch_size, 100)  # OpenAI limit
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        
        for retry in range(config.processing.max_retries):
            try:
                # Generate embeddings for batch
                texts = [chunk["content"] for chunk in batch]
                
                response = openai_client.embeddings.create(
                    model=config.openai.embedding_model,
                    input=texts
                )
                
                # Add embeddings to chunks
                for j, embedding_data in enumerate(response.data):
                    batch[j]["embedding"] = embedding_data.embedding
                    embedded_chunks.append(batch[j])
                
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} chunks)")
                break
                
            except Exception as e:
                logger.warning(f"Embedding batch {i//batch_size + 1} failed (attempt {retry + 1}): {str(e)}")
                if retry == config.processing.max_retries - 1:
                    logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1} after {config.processing.max_retries} attempts")
                    # Continue with next batch instead of failing completely
                    break
    
    logger.info(f"Successfully generated embeddings for {len(embedded_chunks)}/{len(chunks)} chunks")
    return embedded_chunks


async def generate_embeddings_async(
    chunks: List[Dict[str, Any]], 
    openai_client: AzureOpenAI, 
    config: AppConfig
) -> List[Dict[str, Any]]:
    """
    Async version of embedding generation for parallel processing.
    """
    logger.info("Starting async embedding generation")
    # Use sync function in async context (OpenAI client handles this)
    return generate_embeddings_batch(chunks, openai_client, config)


@monitor_performance(logger, "search_index_creation")
def create_search_index_if_not_exists(config: AppConfig) -> bool:
    """
    Create search index if it doesn't exist.
    Returns True if index exists or was created successfully.
    """
    try:
        index_client = SearchIndexClient(
            endpoint=config.search.endpoint,
            credential=AzureKeyCredential(config.search.api_key)
        )
        
        # Check if index exists
        try:
            existing_index = index_client.get_index(config.search.index_name)
            logger.info(f"Search index '{config.search.index_name}' already exists")
            return True
        except Exception:
            logger.info(f"Search index '{config.search.index_name}' not found, creating...")
        
        # Define index schema
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SimpleField(name="document_id", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding", 
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True, 
                vector_search_dimensions=3072,  # text-embedding-3-large dimension
                vector_search_profile_name="default-profile"
            ),
            SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="metadata", type=SearchFieldDataType.String),
            SimpleField(name="tokens", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="file_extension", type=SearchFieldDataType.String, filterable=True),
            SearchableField(name="title", type=SearchFieldDataType.String)
        ]
        
        # Configure vector search
        vector_search = VectorSearch(
            profiles=[
                VectorSearchProfile(
                    name="default-profile",
                    algorithm_configuration_name="default-algorithm"
                )
            ],
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="default-algorithm",
                    kind=VectorSearchAlgorithmKind.HNSW
                )
            ]
        )
        
        # Create index
        index = SearchIndex(
            name=config.search.index_name,
            fields=fields,
            vector_search=vector_search
        )
        
        result = index_client.create_index(index)
        logger.info(f"Successfully created search index: {result.name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create search index: {str(e)}")
        return False


@monitor_performance(logger, "search_index_update")
def update_search_index_batch(
    chunks: List[Dict[str, Any]], 
    search_client: SearchClient, 
    document_id: str,
    config: AppConfig
) -> Dict[str, Any]:
    """
    Update Azure Cognitive Search index with embedded chunks using batch operations.
    """
    documents = []
    
    for chunk in chunks:
        document = {
            "id": chunk["id"],
            "document_id": document_id,
            "content": chunk["content"],
            "embedding": chunk.get("embedding", []),
            "chunk_index": chunk["chunk_index"],
            "metadata": json.dumps(chunk.get("metadata", {})),
            "tokens": chunk.get("token_count", 0),
            "file_extension": chunk.get("file_extension", ""),
            "title": chunk.get("title", "")
        }
        documents.append(document)
    
    # Batch upload with retry logic
    batch_size = min(config.processing.batch_size, 1000)  # ACS limit
    total_uploaded = 0
    failed_uploads = 0
    
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        
        for retry in range(config.processing.max_retries):
            try:
                result = search_client.upload_documents(batch)
                
                # Count successful uploads
                successful = sum(1 for r in result if r.succeeded)
                total_uploaded += successful
                failed_uploads += len(batch) - successful
                
                logger.info(f"Uploaded batch {i//batch_size + 1}: {successful}/{len(batch)} documents succeeded")
                break
                
            except Exception as e:
                logger.warning(f"Search index batch {i//batch_size + 1} failed (attempt {retry + 1}): {str(e)}")
                if retry == config.processing.max_retries - 1:
                    logger.error(f"Failed to upload batch {i//batch_size + 1} after {config.processing.max_retries} attempts")
                    failed_uploads += len(batch)
    
    success = failed_uploads == 0
    return {
        "success": success,
        "uploaded_count": total_uploaded,
        "failed_count": failed_uploads,
        "total_documents": len(documents)
    }


@exception_handler(logger)
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function to process documents: PDF extraction via Document Intelligence, chunking, and embedding.
    
    Expected request body:
    {
        "text": "your text content here (for text files)",
        "pdf_base64": "base64 encoded PDF content (for PDF files)", 
        "filename": "document.pdf",  
        "document_id": "optional_document_id",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "content_type": "application/pdf",
        "use_layout_model": true,
        "extract_contract": false,
        "extraction_chunk_size": 2000,
        "extraction_chunk_overlap": 400
    }
    """
    with track_activity(logger, "process_document") as activity_logger:
        # Start total timing
        total_start_time = time.time()
        timing_stats = {"start_time": total_start_time}
        
        try:
            logger.info("=== Starting document processing request ===")
            
            # Get configuration
            logger.info("Loading application configuration...")
            config = get_app_config_cached()
            logger.info("Configuration loaded successfully")
            
            # Parse and validate request
            logger.info("Parsing request body...")
            req_body = req.get_json()
            if not req_body:
                logger.error("No request body provided")
                return func.HttpResponse(
                    json.dumps({"error": "Request body is required"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            logger.info("Request body parsed successfully")
            
            # Extract parameters with defaults
            filename = req_body.get('filename', 'document.txt')
            document_id = req_body.get('document_id', str(uuid.uuid4()))
            chunk_size = req_body.get('chunk_size', config.processing.chunk_size)
            chunk_overlap = req_body.get('chunk_overlap', config.processing.chunk_overlap)
            content_type = req_body.get('content_type', 'text/plain')
            use_layout_model = req_body.get('use_layout_model', True)
            
            logger.info(f"Processing parameters - Document ID: {document_id}, Filename: {filename}, Content Type: {content_type}")
            
            activity_logger.set_activity_info("document_id", document_id)
            activity_logger.set_activity_info("filename", filename)
            activity_logger.set_activity_info("content_type", content_type)
            
            # Determine processing path based on content type
            text_content = None
            pdf_analysis_result = None
            
            if content_type == "application/pdf" or filename.lower().endswith('.pdf'):
                pdf_start = time.time()
                logger.info("=== Processing PDF document ===")
                
                pdf_base64 = req_body.get('pdf_base64', '').strip()
                if not pdf_base64:
                    logger.error("PDF base64 content is required for PDF files")
                    return func.HttpResponse(
                        json.dumps({"error": "pdf_base64 content is required for PDF files"}),
                        status_code=400,
                        mimetype="application/json"
                    )
                
                # Initialize PDF processor
                logger.info("Initializing PDF processor with Document Intelligence...")
                pdf_processor = create_pdf_processor(config.document_intelligence)
                
                # Process PDF
                logger.info("Starting PDF extraction with Document Intelligence API...")
                pdf_analysis_result = pdf_processor.process_pdf_base64(pdf_base64, filename, use_layout_model)
                text_content = pdf_analysis_result.get_full_text()
                
                timing_stats["pdf_processing_time"] = time.time() - pdf_start
                logger.info(f"PDF extraction completed in {timing_stats['pdf_processing_time']:.2f}s - Extracted {len(text_content)} characters from {pdf_analysis_result.metadata['total_pages']} pages")
                activity_logger.set_activity_info("pdf_pages", pdf_analysis_result.metadata['total_pages'])
                activity_logger.set_activity_info("pdf_processing_time", pdf_analysis_result.processing_stats['processing_time_seconds'])
                
            else:
                logger.info("=== Processing text document ===")
                text_content = req_body.get('text', '').strip()
                if not text_content:
                    logger.error("Text content is required for non-PDF files")
                    return func.HttpResponse(
                        json.dumps({"error": "Text content is required and cannot be empty"}),
                        status_code=400,
                        mimetype="application/json"
                    )
                logger.info(f"Text content loaded - {len(text_content)} characters")
            
            activity_logger.set_activity_info("text_length", len(text_content))
            
            # Initialize OpenAI client
            logger.info("Initializing OpenAI client...")
            openai_client = AzureOpenAI(
                api_key=config.openai.api_key,
                api_version=config.openai.api_version,
                azure_endpoint=config.openai.endpoint
            )
            logger.info("OpenAI client initialized successfully")
            
            # Create search index if it doesn't exist
            logger.info("Checking/creating search index...")
            index_created = create_search_index_if_not_exists(config)
            if not index_created:
                logger.error("Failed to create or verify search index")
                return func.HttpResponse(
                    json.dumps({"error": "Failed to create or verify search index"}),
                    status_code=500,
                    mimetype="application/json"
                )
            
            search_client = SearchClient(
                endpoint=config.search.endpoint,
                index_name=config.search.index_name,
                credential=AzureKeyCredential(config.search.api_key)
            )
            
            # Step 1: Process document with sophisticated chunking
            logger.info("=== Starting document chunking ===")
            logger.info(f"Processing document: {filename} ({len(text_content)} chars)")
            
            processor = DocumentProcessor(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                use_rcts=config.processing.use_rcts,
                encoding_name=config.processing.encoding_name
            )
            logger.info(f"Document processor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            
            chunked_document = processor.process_document(text_content, filename, content_type, activity_logger)
            logger.info(f"Document chunking completed - Generated {len(chunked_document.chunks)} chunks")
            
            if not chunked_document.chunks:
                logger.error("No chunks were generated from the document")
                return func.HttpResponse(
                    json.dumps({"error": "No chunks generated from the text"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            # Get processing stats
            logger.info("=== Generating processing statistics ===")
            processing_stats = processor.get_processing_stats(chunked_document)
            log_processing_stats(logger, processing_stats, activity_logger)
            logger.info(f"Processing stats: {processing_stats}")
            
            # Convert to format for embeddings
            logger.info("=== Preparing chunks for embedding generation ===")
            chunks_for_embedding = []
            for i, chunk in enumerate(chunked_document.chunks):
                chunk_data = {
                    "id": str(uuid.uuid4()),
                    "content": chunk.page_content,
                    "chunk_index": i,
                    "metadata": chunk.metadata,
                    "file_extension": Path(filename).suffix.lower(),
                    "title": chunk.metadata.get("source", {}).get("title", filename),
                    "token_count": len(chunk.page_content.split())  # Rough estimate
                }
                chunks_for_embedding.append(chunk_data)
                logger.debug(f"Prepared chunk {i+1}/{len(chunked_document.chunks)} for embedding (ID: {chunk_data['id']})")
            
            logger.info(f"Prepared {len(chunks_for_embedding)} chunks for embedding generation")
            
            # Start parallel processing: embeddings and contract extraction
            parallel_start = time.time()
            logger.info("=== Starting Parallel Processing ===")
            
            # Check if contract extraction is requested
            extract_contract = req_body.get('extract_contract', False)
            
            if extract_contract:
                logger.info("Starting parallel processing: embeddings + contract extraction")
                
                # Run embedding generation and contract extraction in parallel
                async def run_parallel_processing():
                    # Create tasks for parallel execution
                    embedding_task = asyncio.create_task(
                        generate_embeddings_async(chunks_for_embedding, openai_client, config)
                    )
                    
                    extraction_task = asyncio.create_task(
                        call_contract_extraction_async(
                            text_content=text_content,
                            filename=filename,
                            document_id=document_id,
                            extraction_chunk_size=req_body.get('extraction_chunk_size', 2000),
                            extraction_chunk_overlap=req_body.get('extraction_chunk_overlap', 400)
                        )
                    )
                    
                    # Wait for both to complete
                    embedded_chunks, contract_extraction_result = await asyncio.gather(
                        embedding_task, extraction_task, return_exceptions=True
                    )
                    
                    return embedded_chunks, contract_extraction_result
                
                # Run parallel processing
                embedded_chunks, contract_extraction_result = asyncio.run(run_parallel_processing())
                
                # Handle exceptions from parallel tasks
                if isinstance(embedded_chunks, Exception):
                    logger.error(f"Embedding generation failed: {str(embedded_chunks)}")
                    embedded_chunks = []
                if isinstance(contract_extraction_result, Exception):
                    logger.error(f"Contract extraction failed: {str(contract_extraction_result)}")
                    contract_extraction_result = None
                    
                logger.info("Parallel processing completed")
                
            else:
                logger.info("Contract extraction not requested, running embeddings only")
                # Step 2: Generate embeddings with batching
                logger.info("=== Starting embedding generation ===")
                logger.info(f"Generating embeddings for {len(chunks_for_embedding)} chunks using model: {config.openai.embedding_model}")
                embedded_chunks = generate_embeddings_batch(chunks_for_embedding, openai_client, config)
                contract_extraction_result = None
            
            timing_stats["parallel_processing_time"] = time.time() - parallel_start
            logger.info(f"Embedding generation completed - {len(embedded_chunks)} chunks successfully embedded in {timing_stats['parallel_processing_time']:.2f}s")
            
            if not embedded_chunks:
                logger.error("No embeddings were generated")
                return func.HttpResponse(
                    json.dumps({"error": "No embeddings generated"}),
                    status_code=500,
                    mimetype="application/json"
                )
            
            # Step 3: Save files locally (if enabled)
            logger.info("=== Local file saving ===")
            local_file_saver = create_local_file_saver(config.local_storage)
            saved_files = {}
            
            if config.local_storage.enabled:
                logger.info("Local storage enabled - saving processing results to files")
                # Prepare metadata for saving
                save_metadata = {"content_type": content_type}
                if pdf_analysis_result:
                    save_metadata.update(pdf_analysis_result.metadata)
                
                saved_files = local_file_saver.save_processing_results(
                    document_id=document_id,
                    filename=filename,
                    original_text=text_content,
                    chunks=chunks_for_embedding,
                    embeddings=embedded_chunks,
                    metadata=save_metadata,
                    processing_stats=processing_stats,
                    timing_stats=timing_stats
                )
                logger.info(f"Local files saved: {list(saved_files.keys())}")
            else:
                logger.info("Local storage disabled - skipping file save")
            
            # Step 4: Update search index with batching
            search_start = time.time()
            logger.info("=== Search index update ===")
            logger.info(f"Updating search index with {len(embedded_chunks)} chunks")
            index_result = update_search_index_batch(embedded_chunks, search_client, document_id, config)
            timing_stats["search_index_time"] = time.time() - search_start
            
            # Calculate total processing time
            timing_stats["total_processing_time"] = time.time() - total_start_time
            
            # Compile comprehensive response
            logger.info("=== Compiling response ===")
            response = {
                "document_id": document_id,
                "filename": filename,
                "processing_stats": processing_stats,
                "embedding_stats": {
                    "total_chunks": len(chunks_for_embedding),
                    "embedded_chunks": len(embedded_chunks),
                    "embedding_success_rate": len(embedded_chunks) / len(chunks_for_embedding) if chunks_for_embedding else 0
                },
                "index_stats": index_result,
                "timing_stats": timing_stats,
                "metadata": {
                    "content_type": content_type,
                    "processing_config": {
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "use_rcts": config.processing.use_rcts,
                        "encoding_name": config.processing.encoding_name
                    }
                }
            }
            
            # Add PDF-specific metadata if applicable
            if pdf_analysis_result:
                logger.info("Adding PDF analysis metadata to response")
                response["pdf_analysis"] = {
                    "pages_processed": pdf_analysis_result.metadata['total_pages'],
                    "model_used": pdf_analysis_result.metadata['model_used'],
                    "processing_time_seconds": pdf_analysis_result.processing_stats['processing_time_seconds'],
                    "extraction_method": pdf_analysis_result.metadata['extraction_method']
                }
            
            # Add local storage info if files were saved
            if saved_files:
                logger.info("Adding local storage info to response")
                response["local_storage"] = {
                    "enabled": config.local_storage.enabled,
                    "files_saved": saved_files,
                    "storage_stats": local_file_saver.get_storage_stats()
                }
            
            # Add contract extraction results if performed
            if extract_contract and contract_extraction_result:
                logger.info("Adding contract extraction results to response")
                response["contract_extraction"] = contract_extraction_result
                activity_logger.set_activity_info("contract_extraction_success", 
                    contract_extraction_result.get("extraction_result", {}).get("extraction_metadata", {}).get("success", False))
            
            activity_logger.set_activity_info("total_chunks", len(chunks_for_embedding))
            activity_logger.set_activity_info("embedded_chunks", len(embedded_chunks))
            activity_logger.set_activity_info("uploaded_chunks", index_result.get("uploaded_count", 0))
            
            status_code = 200 if index_result["success"] else 207  # 207 for partial success
            
            logger.info(f"=== Request completed successfully ===")
            logger.info(f"Final stats - Chunks: {len(chunks_for_embedding)}, Embedded: {len(embedded_chunks)}, Status: {status_code}")
            logger.info(f"Total processing time: {timing_stats['total_processing_time']:.2f}s")
            
            return func.HttpResponse(
                json.dumps(response, indent=2),
                status_code=status_code,
                mimetype="application/json"
            )
            
        except ValueError as e:
            logger.error(f"=== Validation error occurred ===")
            logger.error(f"Validation error: {str(e)}")
            activity_logger.error(f"Validation error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Configuration or validation error: {str(e)}"}),
                status_code=400,
                mimetype="application/json"
            )
        except Exception as e:
            logger.error(f"=== Unexpected error occurred ===")
            logger.error(f"Processing error: {str(e)}")
            activity_logger.error(f"Processing error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Processing failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )

