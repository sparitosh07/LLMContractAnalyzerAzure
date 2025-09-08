"""
Orchestrator function for document processing.
Contains the main processing logic from DocParserFunc.
"""

import logging
import json
import time
import asyncio
import aiohttp
from typing import Dict, Any, Optional
import azure.durable_functions as df

logger = logging.getLogger(__name__)


def orchestrator_function(context: df.DurableOrchestrationContext) -> Dict[str, Any]:
    """
    Main orchestrator for document processing.
    Replaces the main logic from DocParserFunc.
    """
    input_data = context.get_input()
    
    try:
        logger.info("=== Starting Document Processing Orchestration ===")
        
        # Start total timing
        total_start_time = time.time()
        timing_stats = {"start_time": total_start_time}
        
        # Extract parameters
        document_id = input_data["document_id"]
        filename = input_data["filename"]
        content_type = input_data["content_type"]
        use_layout_model = input_data.get("use_layout_model", True)
        extract_contract = input_data.get("extract_contract", False)
        
        logger.info(f"Processing parameters - Document ID: {document_id}, Filename: {filename}, Content Type: {content_type}")
        
        # Step 1: Process document content (PDF or text)
        text_content = None
        doc_intel_data = None
        pdf_analysis_result = None
        
        if content_type == "application/pdf" or filename.lower().endswith('.pdf'):
            # Process PDF using activity function
            pdf_input = {
                "pdf_base64": input_data.get("pdf_base64"),
                "filename": filename,
                "use_layout_model": use_layout_model
            }
            
            pdf_result = yield context.call_activity("pdf_processing_activity", pdf_input)
            
            if "error" in pdf_result:
                return {"error": f"PDF processing failed: {pdf_result['error']}"}
            
            text_content = pdf_result["text_content"]
            doc_intel_data = pdf_result["doc_intel_data"]
            pdf_analysis_result = pdf_result.get("analysis_metadata")
            timing_stats["pdf_processing_time"] = pdf_result.get("processing_time", 0)
            
        else:
            # Process text document
            text_content = input_data.get('text', '').strip()
            if not text_content:
                return {"error": "Text content is required and cannot be empty"}
            doc_intel_data = None
        
        if not text_content:
            return {"error": "No text content extracted from document"}
        
        logger.info(f"Text content extracted - {len(text_content)} characters")
        
        # Step 2: Prepare function calls
        parallel_start = time.time()
        
        if extract_contract:
            logger.info("Starting parallel processing: chunking/embedding + contract extraction")
            
            # Prepare inputs for parallel execution
            chunk_embed_input = {
                "text_content": text_content,
                "filename": filename,
                "document_id": document_id,
                "doc_intel_data": doc_intel_data
            }
            
            contract_extraction_input = {
                "text_content": text_content,
                "filename": filename,
                "document_id": document_id,
                "extraction_chunk_size": input_data.get("extraction_chunk_size", 2000),
                "extraction_chunk_overlap": input_data.get("extraction_chunk_overlap", 400),
                "save_to_local": True
            }
            
            # Execute HTTP calls in parallel using activities
            chunk_embed_task = context.call_activity("call_chunk_embed_activity", chunk_embed_input)
            contract_extraction_task = context.call_activity("call_contract_extraction_activity", contract_extraction_input)
            
            # Wait for both to complete
            chunk_embed_result, contract_extraction_result = yield context.task_all([
                chunk_embed_task, 
                contract_extraction_task
            ])
            
        else:
            logger.info("Contract extraction not requested, running chunking and embedding only")
            
            chunk_embed_input = {
                "text_content": text_content,
                "filename": filename,
                "document_id": document_id,
                "doc_intel_data": doc_intel_data
            }
            
            chunk_embed_result = yield context.call_activity("call_chunk_embed_activity", chunk_embed_input)
            contract_extraction_result = None
        
        timing_stats["parallel_processing_time"] = time.time() - parallel_start
        timing_stats["total_processing_time"] = time.time() - total_start_time
        
        # Step 3: Compile response
        logger.info("=== Compiling response ===")
        response = {
            "document_id": document_id,
            "filename": filename,
            "timing_stats": timing_stats,
            "metadata": {
                "content_type": content_type,
                "orchestrator": "DocParserDurableFunc",
                "parallel_functions": ["ChunkEmbedFunc", "ContractExtractorFunc"] if extract_contract else ["ChunkEmbedFunc"]
            }
        }
        
        # Add results from ChunkEmbedFunc
        if extract_contract:
            if chunk_embed_result and "error" not in chunk_embed_result:
                response["chunk_embed_result"] = chunk_embed_result
                logger.info("ChunkEmbedFunc results added to response")
            else:
                response["chunk_embed_error"] = chunk_embed_result.get("error", "Unknown error")
                logger.error("ChunkEmbedFunc failed")
        else:
            if chunk_embed_result and "error" not in chunk_embed_result:
                response.update(chunk_embed_result)
                logger.info("ChunkEmbedFunc results merged into response")
            else:
                response["error"] = chunk_embed_result.get("error", "Unknown error")
                logger.error("ChunkEmbedFunc failed")
        
        # Add PDF-specific metadata if applicable
        if pdf_analysis_result:
            logger.info("Adding PDF analysis metadata to response")
            response["pdf_analysis"] = pdf_analysis_result
        
        # Add contract extraction results if performed
        if extract_contract and contract_extraction_result and "error" not in contract_extraction_result:
            response["contract_extraction"] = contract_extraction_result
            logger.info("Contract extraction results added to response")
        elif extract_contract:
            response["contract_extraction_error"] = contract_extraction_result.get("error", "Unknown error") if contract_extraction_result else "Contract extraction function returned no result"
            logger.error("Contract extraction failed")
        
        # Determine status
        if extract_contract:
            chunk_success = chunk_embed_result and "error" not in chunk_embed_result
            extract_success = contract_extraction_result and "error" not in contract_extraction_result
            status_code = 200 if (chunk_success and extract_success) else 207
        else:
            status_code = 200 if (chunk_embed_result and "error" not in chunk_embed_result) else 500
        
        response["status_code"] = status_code
        
        # Step 4: Write to Cosmos DB if successful
        if status_code == 200:
            try:
                logger.info("=== Writing to Cosmos DB ===")
                cosmos_input = {
                    "document_id": document_id,
                    "filename": filename,
                    "results": response
                }
                
                cosmos_result = yield context.call_activity("call_cosmos_writer_activity", cosmos_input)
                
                if cosmos_result and cosmos_result.get("cosmos_write_success"):
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
        
        logger.info(f"=== Orchestration completed successfully ===")
        logger.info(f"Status: {status_code}")
        logger.info(f"Total processing time: {timing_stats['total_processing_time']:.2f}s")
        
        return response
        
    except Exception as e:
        logger.error(f"Orchestration failed: {str(e)}")
        return {
            "error": f"Processing failed: {str(e)}",
            "document_id": input_data.get("document_id"),
            "filename": input_data.get("filename")
        }


# Register the orchestrator
main = df.Orchestrator.create(orchestrator_function)