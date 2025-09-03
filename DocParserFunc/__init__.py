import logging
import json
import uuid
import asyncio
import aiohttp
import time
import os
from typing import Dict, Any, Optional

import azure.functions as func

from .config import get_app_config, AppConfig, get_contract_extraction_prompts
from .pdf_processor import create_pdf_processor
from .utils.logging_utils import track_activity, LoggingConfig, exception_handler


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


async def call_cosmos_writer_async(
    document_id: str,
    filename: str,
    results: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Call CosmosWriterFunc asynchronously"""
    try:
        function_url = "http://localhost:7071/api/CosmosWriterFunc"
        
        payload = {
            "document_id": document_id,
            "filename": filename,
            "results": results
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(function_url, json=payload, timeout=120) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Cosmos DB write completed successfully")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Cosmos DB write failed with status {response.status}: {error_text}")
                    return None
                    
    except Exception as e:
        logger.error(f"Failed to call CosmosWriterFunc: {str(e)}")
        return None


def get_app_config_cached() -> AppConfig:
    """Get cached app configuration."""
    global APP_CONFIG
    if APP_CONFIG is None:
        APP_CONFIG = get_app_config()
    return APP_CONFIG




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
            content_type = req_body.get('content_type', 'text/plain')
            use_layout_model = req_body.get('use_layout_model', True)
            extract_contract = req_body.get('extract_contract', False)
            
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
            
            
            # DocParserFunc now acts as an orchestrator for parallel processing
            logger.info("=== Starting Document Processing Orchestration ===")
            logger.info(f"Processing document: {filename} ({len(text_content)} chars)")
            
            # Start parallel processing
            parallel_start = time.time()
            logger.info("=== Starting Parallel Processing ===")
            
            if extract_contract:
                logger.info("Starting parallel processing: embeddings + contract extraction")
                
                # Run chunking/embedding and contract extraction in parallel
                async def run_parallel_processing():
                    # Create tasks for parallel execution
                    chunk_embed_task = asyncio.create_task(
                        call_chunk_embed_func_async(
                            text_content=text_content,
                            filename=filename,
                            document_id=document_id
                        )
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
                    chunk_embed_result, contract_extraction_result = await asyncio.gather(
                        chunk_embed_task, extraction_task, return_exceptions=True
                    )
                    
                    return chunk_embed_result, contract_extraction_result
                
                # Run parallel processing
                chunk_embed_result, contract_extraction_result = asyncio.run(run_parallel_processing())
                
                # Handle exceptions from parallel tasks
                if isinstance(chunk_embed_result, Exception):
                    logger.error(f"Chunk embedding function failed: {str(chunk_embed_result)}")
                    chunk_embed_result = {"error": str(chunk_embed_result)}
                if isinstance(contract_extraction_result, Exception):
                    logger.error(f"Contract extraction failed: {str(contract_extraction_result)}")
                    contract_extraction_result = {"error": str(contract_extraction_result)}
                    
                logger.info("Parallel processing completed")
                
                # ChunkEmbedFunc and ContractExtractorFunc handle their own processing
                logger.info("Both functions completed processing")
                
            else:
                logger.info("Contract extraction not requested, running chunking and embedding only")
                # Call ChunkEmbedFunc for chunking and embedding
                logger.info("=== Calling ChunkEmbedFunc ===")
                chunk_embed_result = asyncio.run(call_chunk_embed_func_async(
                    text_content=text_content,
                    filename=filename,
                    document_id=document_id
                ))
                contract_extraction_result = None
                
                # ChunkEmbedFunc handles its own processing
                logger.info("ChunkEmbedFunc processing completed")
            
            timing_stats["parallel_processing_time"] = time.time() - parallel_start
            
            # Calculate total processing time
            timing_stats["total_processing_time"] = time.time() - total_start_time
            
            # Compile comprehensive response
            logger.info("=== Compiling response ===")
            response = {
                "document_id": document_id,
                "filename": filename,
                "timing_stats": timing_stats,
                "metadata": {
                    "content_type": content_type,
                    "orchestrator": "DocParserFunc",
                    "parallel_functions": ["ChunkEmbedFunc", "ContractExtractorFunc"] if extract_contract else ["ChunkEmbedFunc"]
                }
            }
            
            # Add results from ChunkEmbedFunc (if parallel processing)
            if extract_contract:
                if not isinstance(chunk_embed_result, Exception) and "error" not in chunk_embed_result:
                    response["chunk_embed_result"] = chunk_embed_result
                    logger.info("ChunkEmbedFunc results added to response")
                else:
                    response["chunk_embed_error"] = str(chunk_embed_result) if isinstance(chunk_embed_result, Exception) else chunk_embed_result.get("error")
                    logger.error("ChunkEmbedFunc failed")
            else:
                # For non-parallel processing, ChunkEmbedFunc result is the main result
                if not isinstance(chunk_embed_result, Exception) and "error" not in chunk_embed_result:
                    response.update(chunk_embed_result)
                    logger.info("ChunkEmbedFunc results merged into response")
                else:
                    response["error"] = str(chunk_embed_result) if isinstance(chunk_embed_result, Exception) else chunk_embed_result.get("error")
                    logger.error("ChunkEmbedFunc failed")
            
            # Add PDF-specific metadata if applicable
            if pdf_analysis_result:
                logger.info("Adding PDF analysis metadata to response")
                response["pdf_analysis"] = {
                    "pages_processed": pdf_analysis_result.metadata['total_pages'],
                    "model_used": pdf_analysis_result.metadata['model_used'],
                    "processing_time_seconds": pdf_analysis_result.processing_stats['processing_time_seconds'],
                    "extraction_method": pdf_analysis_result.metadata['extraction_method']
                }
            
            # Add contract extraction results if performed
            if extract_contract and contract_extraction_result and not isinstance(contract_extraction_result, Exception) and "error" not in contract_extraction_result:
                response["contract_extraction"] = contract_extraction_result
                logger.info("Contract extraction results added to response")
            elif extract_contract:
                if isinstance(contract_extraction_result, Exception):
                    response["contract_extraction_error"] = str(contract_extraction_result)
                elif contract_extraction_result is None:
                    response["contract_extraction_error"] = "Contract extraction function returned no result"
                else:
                    response["contract_extraction_error"] = contract_extraction_result.get("error", "Unknown error")
                logger.error("Contract extraction failed")
            
            # Determine status code
            if extract_contract:
                # Both functions must succeed for 200
                chunk_success = not isinstance(chunk_embed_result, Exception) and chunk_embed_result is not None and "error" not in chunk_embed_result
                extract_success = not isinstance(contract_extraction_result, Exception) and contract_extraction_result is not None and "error" not in contract_extraction_result
                status_code = 200 if (chunk_success and extract_success) else 207
            else:
                # Only ChunkEmbedFunc needs to succeed
                status_code = 200 if (not isinstance(chunk_embed_result, Exception) and "error" not in chunk_embed_result) else 500
            
            # Write to ADLS and local storage
            logger.info("=== Writing Results to Storage ===")
            try:
                import sys
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'CosmosWriterFunc'))
                from adls_integration import write_results_to_adls_and_local
                
                storage_info = write_results_to_adls_and_local(
                    config, 
                    document_id, 
                    filename, 
                    response
                )
                
                # Add storage info to response
                if storage_info["local_storage"]["enabled"]:
                    response["local_storage"] = storage_info["local_storage"]
                if storage_info["adls_storage"]["enabled"]:
                    response["adls_storage"] = storage_info["adls_storage"]
                    
                logger.info("Storage operations completed")
                
            except Exception as e:
                logger.warning(f"Storage operations failed: {str(e)}")
            
            # Write to Cosmos DB (if successful processing)
            if status_code == 200:
                try:
                    logger.info("=== Writing to Cosmos DB ===\"")
                    cosmos_result = asyncio.run(call_cosmos_writer_async(
                        document_id=document_id,
                        filename=filename,
                        results=response
                    ))
                    
                    if cosmos_result:
                        response["cosmos_db_write"] = {
                            "success": True,
                            "message": "Document successfully written to Cosmos DB"
                        }
                        logger.info("Cosmos DB write completed successfully")
                    else:
                        response["cosmos_db_write"] = {
                            "success": False,
                            "message": "Failed to write to Cosmos DB"
                        }
                        logger.warning("Cosmos DB write failed")
                        
                except Exception as e:
                    logger.warning(f"Cosmos DB write failed: {str(e)}")
                    response["cosmos_db_write"] = {
                        "success": False,
                        "error": str(e)
                    }
            
            logger.info(f"=== Request completed successfully ===")
            logger.info(f"Status: {status_code}")
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


async def call_chunk_embed_func_async(text_content: str, filename: str, document_id: str) -> Dict[str, Any]:
    """Call ChunkEmbedFunc asynchronously"""
    try:
        async with aiohttp.ClientSession() as session:
            payload = {
                "text_content": text_content,
                "filename": filename,
                "document_id": document_id
            }
            
            async with session.post(
                "http://localhost:7071/api/ChunkEmbedFunc",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=300
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("ChunkEmbedFunc completed successfully")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"ChunkEmbedFunc failed with status {response.status}: {error_text}")
                    return {"error": f"ChunkEmbedFunc failed with status {response.status}: {error_text}"}
                    
    except Exception as e:
        logger.error(f"Failed to call ChunkEmbedFunc: {str(e)}")
        return {"error": str(e)}


