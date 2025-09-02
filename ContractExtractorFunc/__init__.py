"""
Azure Function for contract extraction using LangGraph map-reduce methodology.
Runs in parallel with chunking/embedding for optimal performance.
"""

import logging
import json
import os
import uuid
from typing import Dict, Any, Optional
from pathlib import Path
import azure.functions as func

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DocParserFunc'))
from config import get_app_config, AppConfig, get_contract_extraction_prompts
from .contract_extractor import ContractExtractor, create_contract_extractor
from .local_storage import LocalFileSaver, create_local_file_saver
from .utils.logging_utils import (
    track_activity, monitor_performance, 
    LoggingConfig, exception_handler
)

# Setup logging
LoggingConfig.setup_logging(level=logging.INFO)
logger = LoggingConfig.get_function_logger("extract_contract")

# Global config (loaded once)
APP_CONFIG: Optional[AppConfig] = None


def get_app_config_cached() -> AppConfig:
    """Get cached app configuration."""
    global APP_CONFIG
    if APP_CONFIG is None:
        APP_CONFIG = get_app_config()
    return APP_CONFIG


@exception_handler(logger)
def main(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function for contract extraction using LangGraph.
    
    Expected request body:
    {
        "text_content": "extracted text from PDF/document",
        "filename": "document.pdf",
        "document_id": "unique_document_id",
        "extraction_chunk_size": 2000,
        "extraction_chunk_overlap": 400,
        "save_to_local": true
    }
    """
    with track_activity(logger, "extract_contract") as activity_logger:
        try:
            logger.info("=== Starting contract extraction request ===")
            
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
            
            # Extract parameters
            text_content = req_body.get('text_content', '').strip()
            filename = req_body.get('filename', 'document.txt')
            document_id = req_body.get('document_id', str(uuid.uuid4()))
            extraction_chunk_size = req_body.get('extraction_chunk_size', 2000)
            extraction_chunk_overlap = req_body.get('extraction_chunk_overlap', 400)
            save_to_local = req_body.get('save_to_local', True)
            
            if not text_content:
                logger.error("Text content is required")
                return func.HttpResponse(
                    json.dumps({"error": "text_content is required and cannot be empty"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            logger.info(f"Processing parameters - Document ID: {document_id}, Filename: {filename}")
            logger.info(f"Text content length: {len(text_content)} characters")
            
            activity_logger.set_activity_info("document_id", document_id)
            activity_logger.set_activity_info("filename", filename)
            activity_logger.set_activity_info("text_length", len(text_content))
            
            # Initialize contract extractor with LangGraph
            logger.info("=== Initializing LangGraph Contract Extractor ===")
            # Use config from main config file, but allow override from request
            extraction_config = config.contract_extraction
            if extraction_chunk_size != 2000:  # Non-default value provided
                extraction_config.chunk_size = extraction_chunk_size
            if extraction_chunk_overlap != 400:  # Non-default value provided  
                extraction_config.chunk_overlap = extraction_chunk_overlap
                
            contract_extractor = create_contract_extractor(
                config.openai, 
                extraction_config, 
                prompt_generation_func=get_contract_extraction_prompts
            )
            logger.info(f"Contract extractor initialized with chunk_size={extraction_chunk_size}, overlap={extraction_chunk_overlap}")
            
            # Extract contract data using LangGraph map-reduce
            logger.info("=== Starting LangGraph Contract Extraction ===")
            contract_extraction_result = contract_extractor.extract_contract(
                text_content=text_content,
                filename=filename,
                document_id=document_id
            )
            
            extraction_success = contract_extraction_result.get("extraction_metadata", {}).get("success", True)
            logger.info(f"Contract extraction completed - Success: {extraction_success}")
            
            activity_logger.set_activity_info("extraction_success", extraction_success)
            activity_logger.set_activity_info("chunks_processed", 
                contract_extraction_result.get("extraction_metadata", {}).get("chunks_processed", 0))
            
            # Save to local storage if enabled
            saved_files = {}
            if save_to_local and config.local_storage.enabled:
                logger.info("=== Saving Contract Extraction Results ===")
                local_file_saver = create_local_file_saver(config.local_storage)
                saved_files = save_contract_extraction_locally(
                    local_file_saver, 
                    document_id, 
                    filename, 
                    contract_extraction_result
                )
                logger.info(f"Contract extraction files saved: {list(saved_files.keys())}")
            
            # Compile response
            logger.info("=== Compiling Response ===")
            response = {
                "document_id": document_id,
                "filename": filename,
                "extraction_result": contract_extraction_result,
                "processing_stats": {
                    "text_length": len(text_content),
                    "extraction_method": "langgraph_map_reduce",
                    "extraction_success": extraction_success
                }
            }
            
            # Add local storage info if files were saved
            if saved_files:
                response["local_storage"] = {
                    "enabled": config.local_storage.enabled,
                    "files_saved": saved_files
                }
            
            status_code = 200 if extraction_success else 207  # 207 for partial success
            
            logger.info(f"=== Contract extraction request completed - Status: {status_code} ===")
            
            return func.HttpResponse(
                json.dumps(response, indent=2),
                status_code=status_code,
                mimetype="application/json"
            )
            
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            activity_logger.error(f"Validation error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Configuration or validation error: {str(e)}"}),
                status_code=400,
                mimetype="application/json"
            )
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            activity_logger.error(f"Processing error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Contract extraction failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )


def save_contract_extraction_locally(
    local_file_saver: LocalFileSaver, 
    document_id: str, 
    filename: str, 
    extraction_result: Dict[str, Any]
) -> Dict[str, str]:
    """Save contract extraction results to local files"""
    
    if not local_file_saver.config.enabled:
        return {}
    
    logger.info(f"Saving contract extraction results for document: {document_id}")
    
    # Create document-specific folder (reuse existing structure)
    doc_folder = local_file_saver._create_document_folder(document_id, filename)
    saved_files = {}
    
    try:
        # Save extracted contract as JSON
        contract_file = doc_folder / "extracted_contract.json"
        with open(contract_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_result["extracted_contract"], f, indent=2, ensure_ascii=False)
        saved_files["extracted_contract"] = str(contract_file)
        logger.info(f"Saved extracted contract: {contract_file}")
        
        # Save extraction metadata
        metadata_file = doc_folder / "extraction_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_result["extraction_metadata"], f, indent=2, ensure_ascii=False)
        saved_files["extraction_metadata"] = str(metadata_file)
        logger.info(f"Saved extraction metadata: {metadata_file}")
        
        # Save human-readable summary
        summary_file = doc_folder / "contract_summary.txt"
        summary_content = generate_contract_summary(extraction_result["extracted_contract"], document_id, filename)
        summary_file.write_text(summary_content, encoding='utf-8')
        saved_files["contract_summary"] = str(summary_file)
        logger.info(f"Saved contract summary: {summary_file}")
        
        logger.info(f"Contract extraction files saved successfully: {len(saved_files)} files")
        return saved_files
        
    except Exception as e:
        logger.error(f"Failed to save contract extraction files: {str(e)}")
        return saved_files


def generate_contract_summary(contract_data: Dict[str, Any], document_id: str, filename: str) -> str:
    """Generate human-readable contract summary"""
    
    summary = f"""Contract Extraction Summary
=============================

Document ID: {document_id}
Filename: {filename}
Extraction Method: LangGraph Map-Reduce

Contract Details:
"""
    
    # Add unique market reference info
    umr = contract_data.get("unique_market_reference", {})
    if any(umr.values()):
        summary += "\nUnique Market Reference:\n"
        for key, value in umr.items():
            if value:
                summary += f"- {key.replace('_', ' ').title()}: {value}\n"
    
    # Add limits summary
    limits = contract_data.get("limits", [])
    if limits:
        summary += f"\nLimits ({len(limits)} found):\n"
        for limit in limits[:5]:  # Show first 5
            amount = limit.get("amount", {})
            summary += f"- {limit.get('description', 'N/A')}: {amount.get('formatted_text', 'N/A')}\n"
    
    # Add premiums summary
    premiums = contract_data.get("premiums", [])
    if premiums:
        summary += f"\nPremiums ({len(premiums)} found):\n"
        for premium in premiums[:5]:  # Show first 5
            amount = premium.get("amount", {})
            summary += f"- {premium.get('premium_type', 'N/A')}: {amount.get('formatted_text', 'N/A')}\n"
    
    # Add coverages summary
    coverages = contract_data.get("coverages", [])
    if coverages:
        summary += f"\nCoverages ({len(coverages)} found):\n"
        for coverage in coverages[:5]:  # Show first 5
            summary += f"- {coverage.get('coverage_name', 'N/A')}: {coverage.get('coverage_type', 'N/A')}\n"
    
    # Add exclusions summary
    exclusions = contract_data.get("exclusions", [])
    if exclusions:
        summary += f"\nExclusions ({len(exclusions)} found):\n"
        for exclusion in exclusions[:5]:  # Show first 5
            summary += f"- {exclusion.get('title', 'N/A')}: {exclusion.get('exclusion_type', 'N/A')}\n"
    
    return summary