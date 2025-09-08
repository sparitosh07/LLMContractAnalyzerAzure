"""
Activity functions for document processing orchestration.
Each activity handles a specific part of the processing pipeline.
"""

import logging
import json
import time
import asyncio
import aiohttp
import os
import sys
from typing import Dict, Any, Optional
import azure.durable_functions as df

# Import from DocParserFunc for PDF processing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DocParserFunc'))
from config import get_app_config
from pdf_processor import create_pdf_processor

logger = logging.getLogger(__name__)


def pdf_processing_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Activity function for PDF processing using Document Intelligence.
    """
    try:
        logger.info("=== Starting PDF Processing Activity ===")
        
        pdf_base64 = input_data.get("pdf_base64", "").strip()
        filename = input_data.get("filename")
        use_layout_model = input_data.get("use_layout_model", True)
        
        if not pdf_base64:
            return {"error": "PDF base64 content is required for PDF files"}
        
        # Get configuration and initialize processor
        logger.info("Initializing PDF processor with Document Intelligence...")
        config = get_app_config()
        pdf_processor = create_pdf_processor(config.document_intelligence)
        
        # Process PDF
        start_time = time.time()
        logger.info("Starting PDF extraction with Document Intelligence API...")
        pdf_analysis_result = pdf_processor.process_pdf_base64(pdf_base64, filename, use_layout_model)
        text_content = pdf_analysis_result.get_full_text()
        processing_time = time.time() - start_time
        
        logger.info(f"PDF extraction completed in {processing_time:.2f}s - Extracted {len(text_content)} characters from {pdf_analysis_result.metadata['total_pages']} pages")
        
        # Use the user's approach with Document Intelligence results
        # Note: analyze_result would be the raw Document Intelligence result object
        # For now, we'll construct doc_intel_data from the processed result
        doc_intel_data = {
            'content': text_content,
            'pages': [
                {
                    'page_number': page.page_number,
                    'spans': [{'offset': 0, 'length': len(page.content)}]  # Simplified span info
                }
                for page in pdf_analysis_result.pages
            ]
        }
        
        return {
            "text_content": text_content,
            "doc_intel_data": doc_intel_data,
            "processing_time": processing_time,
            "analysis_metadata": {
                "pages_processed": pdf_analysis_result.metadata['total_pages'],
                "model_used": pdf_analysis_result.metadata['model_used'],
                "processing_time_seconds": pdf_analysis_result.processing_stats['processing_time_seconds'],
                "extraction_method": pdf_analysis_result.metadata['extraction_method']
            }
        }
        
    except Exception as e:
        logger.error(f"PDF processing activity failed: {str(e)}")
        return {"error": str(e)}


async def call_chunk_embed_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Activity function to call ChunkEmbedFunc HTTP endpoint.
    """
    try:
        logger.info("=== Starting Chunk Embed Activity ===")
        
        payload = {
            "text_content": input_data["text_content"],
            "filename": input_data["filename"],
            "document_id": input_data["document_id"],
            "doc_intel_data": input_data.get("doc_intel_data")  # Pass Document Intelligence data
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:7071/api/ChunkEmbedFunc",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)
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


async def call_contract_extraction_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Activity function to call ContractExtractorFunc HTTP endpoint.
    """
    try:
        logger.info("=== Starting Contract Extraction Activity ===")
        
        payload = {
            "text_content": input_data["text_content"],
            "filename": input_data["filename"],
            "document_id": input_data["document_id"],
            "extraction_chunk_size": input_data.get("extraction_chunk_size", 2000),
            "extraction_chunk_overlap": input_data.get("extraction_chunk_overlap", 400),
            "save_to_local": input_data.get("save_to_local", True)
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:7071/api/ContractExtractorFunc",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Contract extraction completed successfully")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Contract extraction failed with status {response.status}: {error_text}")
                    return {"error": f"Contract extraction failed with status {response.status}: {error_text}"}
                    
    except Exception as e:
        logger.error(f"Failed to call ContractExtractorFunc: {str(e)}")
        return {"error": str(e)}


async def call_cosmos_writer_activity(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Activity function to call CosmosWriterFunc HTTP endpoint.
    """
    try:
        logger.info("=== Starting Cosmos Writer Activity ===")
        
        payload = {
            "document_id": input_data["document_id"],
            "filename": input_data["filename"],
            "results": input_data["results"]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "http://localhost:7071/api/CosmosWriterFunc",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info("Cosmos DB write completed successfully")
                    return result
                else:
                    error_text = await response.text()
                    logger.error(f"Cosmos DB write failed with status {response.status}: {error_text}")
                    return {"error": f"Cosmos DB write failed with status {response.status}: {error_text}"}
                    
    except Exception as e:
        logger.error(f"Failed to call CosmosWriterFunc: {str(e)}")
        return {"error": str(e)}


# Register activity functions
pdf_processing = df.Activity.create(pdf_processing_activity)
call_chunk_embed = df.Activity.create(call_chunk_embed_activity)
call_contract_extraction = df.Activity.create(call_contract_extraction_activity)
call_cosmos_writer = df.Activity.create(call_cosmos_writer_activity)