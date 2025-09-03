"""
Azure Data Lake Storage utilities for writing processed document data.
"""

import json
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from .utils.logging_utils import get_logger
except ImportError:
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DocParserFunc'))
    from utils.logging_utils import get_logger

logger = get_logger(__name__)


def write_document_results_to_adls(
    adls_config, 
    document_id: str, 
    filename: str, 
    results: Dict[str, Any]
) -> Optional[str]:
    """Write complete document processing results to ADLS using DataLakeServiceClient directly."""
    try:
        from azure.storage.filedatalake import DataLakeServiceClient
        
        # Initialize client
        account_url = f"https://{adls_config.account_name}.dfs.core.windows.net"
        service_client = DataLakeServiceClient(
            account_url=account_url,
            credential=adls_config.account_key
        )
        
        # Get file system client
        file_system_client = service_client.get_file_system_client(
            file_system=adls_config.container_name
        )
        
        # Ensure container exists
        try:
            file_system_client.create_file_system()
            logger.info(f"Created ADLS container: {adls_config.container_name}")
        except Exception:
            logger.info(f"ADLS container already exists: {adls_config.container_name}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create folder path: processed-documents/{timestamp}_{document_id}_{filename}/
        safe_filename = _make_safe_filename(filename)
        folder_name = f"{timestamp}_{document_id}_{safe_filename}"
        folder_path = f"{adls_config.base_path}/{folder_name}"
        
        # Write combined results file
        combined_file_path = f"{folder_path}/combined_results.json"
        _write_json_to_adls(file_system_client, combined_file_path, results)
        logger.info(f"Written combined results to ADLS: {combined_file_path}")
        
        # Write individual component files for granular access
        if "chunk_embed_result" in results:
            chunks_path = f"{folder_path}/chunks.json"
            metadata_path = f"{folder_path}/metadata.json"
            
            _write_json_to_adls(file_system_client, chunks_path, results["chunk_embed_result"])
            _write_json_to_adls(file_system_client, metadata_path, results["chunk_embed_result"].get("metadata", {}))
            
            logger.info(f"Written chunk/embedding components to ADLS folder: {folder_path}")
        
        if "contract_extraction" in results:
            contract_path = f"{folder_path}/extracted_contract.json"
            _write_json_to_adls(file_system_client, contract_path, results["contract_extraction"]["extraction_result"])
            logger.info(f"Written contract extraction to ADLS: {contract_path}")
        
        return combined_file_path
        
    except ImportError:
        logger.error("azure-storage-file-datalake package not installed. Install with: pip install azure-storage-file-datalake")
        return None
    except Exception as e:
        logger.error(f"Failed to write document results to ADLS: {str(e)}")
        return None


def _write_json_to_adls(file_system_client, file_path: str, data: Dict[str, Any]) -> None:
    """Write JSON data to ADLS file using DataLakeServiceClient."""
    try:
        # Get file client
        file_client = file_system_client.get_file_client(file_path)
        
        # Convert to JSON
        json_data = json.dumps(data, indent=2, ensure_ascii=False)
        
        # Upload file (overwrite if exists)
        file_client.upload_data(
            data=json_data.encode('utf-8'),
            overwrite=True
        )
        
    except Exception as e:
        logger.error(f"Failed to write file {file_path} to ADLS: {str(e)}")
        raise


def _make_safe_filename(filename: str) -> str:
    """Make filename safe for ADLS paths."""
    # Replace problematic characters
    safe_name = filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
    # Remove file extension for folder naming
    if "." in safe_name:
        safe_name = safe_name.rsplit(".", 1)[0]
    return safe_name


def create_adls_writer(adls_config):
    """Factory function that returns the write function configured with ADLS config."""
    logger.info("Creating ADLS writer function")
    
    def write_results(document_id: str, filename: str, results: Dict[str, Any]) -> Optional[str]:
        return write_document_results_to_adls(adls_config, document_id, filename, results)
    
    return write_results