"""
Azure Function for chunking and embedding documents
Handles document processing, chunking, embeddings generation, and search index updates
"""

import logging
import json
import uuid
import time
from typing import Dict, Any, List
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

from .config import get_app_config
from .document_processing import DocumentProcessor
from .local_storage import LocalFileSaver
from .utils.logging_utils import track_activity


def generate_embeddings_batch(chunks: List[Dict], openai_client: AzureOpenAI, config) -> List[Dict]:
    """Generate embeddings for chunks in batches"""
    logger = logging.getLogger('chunk_embed_func.embeddings')
    embeddings = []
    batch_size = config.processing.batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_texts = [chunk["content"] for chunk in batch]
        
        try:
            response = openai_client.embeddings.create(
                input=batch_texts,
                model=config.openai.embedding_model
            )
            
            for j, embedding_data in enumerate(response.data):
                chunk_index = i + j
                embeddings.append({
                    **batch[j],
                    "embedding": embedding_data.embedding
                })
            
            logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} chunks)")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch {i//batch_size + 1}: {str(e)}")
            continue
    
    return embeddings


def upload_to_search_index(chunks: List[Dict], embeddings: List[Dict], search_client: SearchClient, document_id: str, filename: str) -> Dict[str, Any]:
    """Upload embeddings to Azure Cognitive Search"""
    logger = logging.getLogger('chunk_embed_func.search')
    
    try:
        documents = []
        for chunk, embedding in zip(chunks, embeddings):
            doc = {
                "id": chunk["id"],
                "document_id": document_id,
                "content": chunk["content"],
                "embedding": embedding["embedding"],
                "chunk_index": chunk["chunk_index"],
                "metadata": json.dumps(chunk["metadata"]),
                "tokens": chunk.get("token_count", 0),
                "file_extension": chunk["file_extension"],
                "title": chunk["title"]
            }
            documents.append(doc)
        
        result = search_client.upload_documents(documents)
        
        uploaded_count = sum(1 for r in result if r.succeeded)
        failed_count = len(result) - uploaded_count
        
        logger.info(f"Uploaded batch 1: {uploaded_count}/{len(documents)} documents succeeded")
        
        return {
            "success": uploaded_count > 0,
            "uploaded_count": uploaded_count,
            "failed_count": failed_count,
            "total_documents": len(documents)
        }
        
    except Exception as e:
        logger.error(f"Failed to upload to search index: {str(e)}")
        return {
            "success": False,
            "uploaded_count": 0,
            "failed_count": len(chunks),
            "total_documents": len(chunks),
            "error": str(e)
        }


def create_search_index_if_not_exists(config) -> bool:
    """Create search index if it doesn't exist"""
    logger = logging.getLogger('chunk_embed_func.search_index')
    
    try:
        search_index_client = SearchIndexClient(
            endpoint=config.search.endpoint,
            credential=AzureKeyCredential(config.search.api_key)
        )
        
        # Check if index exists
        try:
            search_index_client.get_index(config.search.index_name)
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
        
        search_index_client.create_index(index)
        logger.info(f"Search index '{config.search.index_name}' created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create search index: {str(e)}")
        return False


def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function V1 for document chunking and embedding.
    
    Expected request body:
    {
        "text_content": "document text...",
        "filename": "document.pdf",
        "document_id": "optional-uuid"
    }
    """
    
    activity_name = "chunk_embed_process"
    start_time = time.time()
    
    with track_activity(logging.getLogger('azure_function.chunk_embed_process'), activity_name) as activity_logger:
        try:
            logger = activity_logger.logger
            logger.info("=== Starting chunking and embedding request ===")
            
            # Load configuration
            logger.info("Loading application configuration...")
            config = get_app_config()
            logger.info("Configuration loaded successfully")
            
            # Parse request body
            logger.info("Parsing request body...")
            try:
                req_body = req.get_json()
                if not req_body:
                    return func.HttpResponse(
                        json.dumps({"error": "Request body is required"}),
                        status_code=400,
                        mimetype="application/json"
                    )
            except Exception as e:
                logger.error(f"Failed to parse request body: {str(e)}")
                return func.HttpResponse(
                    json.dumps({"error": f"Invalid JSON in request body: {str(e)}"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            logger.info("Request body parsed successfully")
            
            # Extract parameters
            text_content = req_body.get("text_content")
            filename = req_body.get("filename", "document.txt")
            document_id = req_body.get("document_id") or str(uuid.uuid4())
            
            if not text_content:
                return func.HttpResponse(
                    json.dumps({"error": "text_content is required"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            logger.info(f"Processing parameters - Document ID: {document_id}, Filename: {filename}")
            logger.info(f"Text content length: {len(text_content)} characters")
            
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
            index_creation_start = time.time()
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
            
            index_creation_time = time.time() - index_creation_start
            logger.info(f"Metric - search_index_creation_duration: {index_creation_time}")
            logger.info(f"Metric - search_index_creation_success: 1")
            
            # Process document for chunking and embeddings
            logger.info("=== Starting document chunking ===")
            logger.info(f"Processing document: {filename} ({len(text_content)} chars)")
            
            processor = DocumentProcessor(
                chunk_size=config.processing.chunk_size,
                chunk_overlap=config.processing.chunk_overlap,
                use_rcts=config.processing.use_rcts,
                encoding_name=config.processing.encoding_name
            )
            logger.info(f"Document processor initialized with chunk_size={config.processing.chunk_size}, chunk_overlap={config.processing.chunk_overlap}")
            
            # Process document
            chunked_document = processor.process_document(text_content, filename, "text/plain", activity_logger)
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
            for key, value in processing_stats.items():
                logger.info(f"  {key}: {value}")
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
                    "token_count": len(chunk.page_content.split())
                }
                chunks_for_embedding.append(chunk_data)
                logger.debug(f"Prepared chunk {i+1}/{len(chunked_document.chunks)} for embedding (ID: {chunk_data['id']})")
            
            logger.info(f"Prepared {len(chunks_for_embedding)} chunks for embedding generation")
            
            # Generate embeddings
            logger.info("=== Starting embedding generation ===")
            embedding_start = time.time()
            
            logger.info(f"Generating embeddings for {len(chunks_for_embedding)} chunks using model: {config.openai.embedding_model}")
            
            embedded_chunks = generate_embeddings_batch(chunks_for_embedding, openai_client, config)
            embedding_time = time.time() - embedding_start
            
            logger.info(f"Metric - embedding_generation_duration: {embedding_time}")
            logger.info(f"Metric - embedding_generation_success: 1")
            logger.info(f"Embedding generation completed - {len(embedded_chunks)} chunks successfully embedded in {embedding_time:.2f}s")
            
            # Save files locally if enabled
            saved_files = {}
            if config.local_storage.enabled:
                logger.info("=== Local file saving ===")
                local_saver = LocalFileSaver(config.local_storage)
                logger.info("Local storage enabled - saving processing results to files")
                
                saved_files = local_saver.save_processing_results(
                    document_id=document_id,
                    filename=filename,
                    original_text=text_content,
                    chunks=chunks_for_embedding,
                    embeddings=embedded_chunks,
                    metadata={"content_type": "text/plain"},
                    processing_stats=processing_stats,
                    timing_stats={
                        "start_time": start_time,
                        "embedding_generation_time": embedding_time,
                        "search_index_creation_time": index_creation_time
                    }
                )
                logger.info(f"Local files saved: {list(saved_files.keys())}")
            
            # Update search index
            logger.info("=== Search index update ===")
            search_start = time.time()
            
            logger.info(f"Updating search index with {len(embedded_chunks)} chunks")
            
            upload_results = upload_to_search_index(
                chunks=chunks_for_embedding,
                embeddings=embedded_chunks,
                search_client=search_client,
                document_id=document_id,
                filename=filename
            )
            
            search_time = time.time() - search_start
            logger.info(f"Metric - search_index_update_duration: {search_time}")
            logger.info(f"Metric - search_index_update_success: 1")
            
            # Compile response
            total_time = time.time() - start_time
            logger.info("=== Compiling response ===")
            
            response_data = {
                "document_id": document_id,
                "filename": filename,
                "processing_stats": processing_stats,
                "embedding_stats": {
                    "total_chunks": len(chunks_for_embedding),
                    "embedded_chunks": len(embedded_chunks),
                    "embedding_success_rate": len(embedded_chunks) / len(chunks_for_embedding) if chunks_for_embedding else 0
                },
                "index_stats": {
                    "success": upload_results["success"],
                    "uploaded_count": upload_results["uploaded_count"],
                    "failed_count": upload_results["failed_count"],
                    "total_documents": upload_results["total_documents"]
                },
                "timing_stats": {
                    "start_time": start_time,
                    "embedding_generation_time": embedding_time,
                    "search_index_creation_time": index_creation_time,
                    "search_index_update_time": search_time,
                    "total_processing_time": total_time
                },
                "metadata": {
                    "content_type": "text/plain",
                    "processing_config": {
                        "chunk_size": config.processing.chunk_size,
                        "chunk_overlap": config.processing.chunk_overlap,
                        "use_rcts": config.processing.use_rcts,
                        "encoding_name": config.processing.encoding_name
                    }
                }
            }
            
            # Add local storage info if enabled
            if config.local_storage.enabled:
                logger.info("Adding local storage info to response")
                local_storage_stats = local_saver.get_storage_stats()
                response_data["local_storage"] = {
                    "enabled": True,
                    "files_saved": saved_files,
                    "storage_stats": local_storage_stats
                }
            
            logger.info("=== Request completed successfully ===")
            logger.info(f"Final stats - Chunks: {processing_stats['total_chunks']}, Embedded: {len(embedded_chunks)}, Status: 200")
            logger.info(f"Total processing time: {total_time:.2f}s")
            
            return func.HttpResponse(
                json.dumps(response_data),
                status_code=200,
                mimetype="application/json"
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            duration = time.time() - start_time
            activity_logger.logger.info(f"[{activity_name}] Activity failed in {duration:.2f} seconds")
            return func.HttpResponse(
                json.dumps({"error": f"Processing failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )