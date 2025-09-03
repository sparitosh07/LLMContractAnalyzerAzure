"""
Integration module to add ADLS storage capability to existing functions.
"""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from .utils.logging_utils import get_logger
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DocParserFunc'))
    from utils.logging_utils import get_logger

from .adls_storage import write_document_results_to_adls

logger = get_logger(__name__)


def write_results_to_adls_and_local(
    config, 
    document_id: str, 
    filename: str, 
    results: Dict[str, Any],
    local_file_saver=None
) -> Dict[str, Any]:
    """
    Write processing results to both ADLS and local storage.
    
    Returns:
        Dictionary with storage information
    """
    storage_info = {
        "local_storage": {"enabled": False},
        "adls_storage": {"enabled": False}
    }
    
    # Write to local storage (existing functionality)
    if local_file_saver and config.local_storage.enabled:
        try:
            logger.info("=== Writing to Local Storage ===")
            local_files = save_combined_results_locally(
                local_file_saver, 
                document_id, 
                filename, 
                results
            )
            storage_info["local_storage"] = {
                "enabled": True,
                "files_saved": local_files,
                "storage_stats": local_file_saver.get_storage_stats()
            }
            logger.info(f"Local storage completed: {len(local_files)} files")
        except Exception as e:
            logger.error(f"Local storage failed: {str(e)}")
            storage_info["local_storage"] = {"enabled": True, "error": str(e)}
    
    # Write to ADLS (new functionality)
    if config.adls:
        try:
            logger.info("=== Writing to ADLS ===")
            adls_path = write_document_results_to_adls(config.adls, document_id, filename, results)
            
            if adls_path:
                storage_info["adls_storage"] = {
                    "enabled": True,
                    "file_path": adls_path,
                    "account_name": config.adls.account_name,
                    "container_name": config.adls.container_name
                }
                logger.info(f"ADLS storage completed: {adls_path}")
            else:
                storage_info["adls_storage"] = {"enabled": True, "error": "Failed to write to ADLS"}
        except Exception as e:
            logger.error(f"ADLS storage failed: {str(e)}")
            storage_info["adls_storage"] = {"enabled": True, "error": str(e)}
    else:
        logger.info("ADLS not configured, skipping ADLS storage")
    
    return storage_info


def save_combined_results_locally(
    local_file_saver, 
    document_id: str, 
    filename: str, 
    results: Dict[str, Any]
) -> Dict[str, str]:
    """Save combined processing results to local files."""
    if not local_file_saver.config.enabled:
        return {}
    
    logger.info(f"Saving combined results for document: {document_id}")
    
    # Create document-specific folder
    doc_folder = local_file_saver._create_document_folder(document_id, filename)
    saved_files = {}
    
    try:
        # Save complete results as JSON
        combined_file = doc_folder / "combined_results.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        saved_files["combined_results"] = str(combined_file)
        logger.info(f"Saved combined results: {combined_file}")
        
        # Save individual components for easy access
        if "chunk_embed_result" in results:
            chunk_embed_file = doc_folder / "chunk_embed_result.json"
            with open(chunk_embed_file, 'w', encoding='utf-8') as f:
                json.dump(results["chunk_embed_result"], f, indent=2, ensure_ascii=False)
            saved_files["chunk_embed_result"] = str(chunk_embed_file)
        
        if "contract_extraction" in results:
            contract_file = doc_folder / "contract_extraction_result.json"
            with open(contract_file, 'w', encoding='utf-8') as f:
                json.dump(results["contract_extraction"], f, indent=2, ensure_ascii=False)
            saved_files["contract_extraction_result"] = str(contract_file)
        
        # Save processing summary
        summary_file = doc_folder / "processing_summary.txt"
        summary_content = generate_processing_summary(results, document_id, filename)
        summary_file.write_text(summary_content, encoding='utf-8')
        saved_files["processing_summary"] = str(summary_file)
        
        logger.info(f"Combined results saved successfully: {len(saved_files)} files")
        return saved_files
        
    except Exception as e:
        logger.error(f"Failed to save combined results: {str(e)}")
        return saved_files


def generate_processing_summary(results: Dict[str, Any], document_id: str, filename: str) -> str:
    """Generate human-readable processing summary."""
    
    summary = f"""Document Processing Summary
============================

Document ID: {document_id}
Filename: {filename}
Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Orchestrator: {results.get('metadata', {}).get('orchestrator', 'Unknown')}

"""
    
    # Add timing information
    timing = results.get("timing_stats", {})
    if timing:
        summary += f"""Processing Performance:
- Total processing time: {timing.get('total_processing_time', 0):.2f}s
- Parallel processing time: {timing.get('parallel_processing_time', 0):.2f}s

"""
    
    # Add chunking/embedding results
    chunk_result = results.get("chunk_embed_result", {})
    if chunk_result:
        processing_stats = chunk_result.get("processing_stats", {})
        embedding_stats = chunk_result.get("embedding_stats", {})
        
        summary += f"""Chunking & Embedding Results:
- Chunks generated: {processing_stats.get('total_chunks', 0)}
- Embeddings created: {embedding_stats.get('embedded_chunks', 0)}
- Success rate: {embedding_stats.get('embedding_success_rate', 0) * 100:.1f}%

"""
    
    # Add contract extraction results
    contract_result = results.get("contract_extraction", {})
    if contract_result:
        extraction_result = contract_result.get("extraction_result", {})
        extracted_contract = extraction_result.get("extracted_contract", {})
        
        summary += f"""Contract Extraction Results:
- Policy Number: {extracted_contract.get('unique_market_reference', {}).get('policy_number', 'N/A')}
- Limits found: {len(extracted_contract.get('limits', []))}
- Premiums found: {len(extracted_contract.get('premiums', []))}
- Coverages found: {len(extracted_contract.get('coverages', []))}
- Exclusions found: {len(extracted_contract.get('exclusions', []))}

"""
    
    return summary


def create_adls_integration(config):
    """Create ADLS integration utilities."""
    if not config.adls:
        logger.info("ADLS not configured, integration disabled")
        return None
    
    logger.info("Creating ADLS integration")
    
    def write_results(document_id: str, filename: str, results: Dict[str, Any]) -> Optional[str]:
        return write_document_results_to_adls(config.adls, document_id, filename, results)
    
    return write_results