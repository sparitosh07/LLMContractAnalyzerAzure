"""
Azure Function to write processed document data to Cosmos DB.
Triggered by ADLS blob storage events for processed documents.
"""

import logging
import json
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from pathlib import Path
import azure.functions as func

# Import shared config
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'DocParserFunc'))
from config import get_app_config, AppConfig
from utils.logging_utils import track_activity, LoggingConfig, exception_handler

# Setup logging
LoggingConfig.setup_logging(level=logging.INFO)
logger = LoggingConfig.get_function_logger("cosmos_writer")

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
    Azure Function triggered by HTTP requests to write data to Cosmos DB.
    
    Expected request body:
    {
        "document_id": "unique_document_id",
        "filename": "document.pdf",
        "results": {...}  // Complete processing results from DocParserFunc
    }
    """
    with track_activity(logger, "cosmos_writer") as activity_logger:
        try:
            logger.info("=== Cosmos Writer Function Started ===")
            
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
            
            # Extract required parameters
            document_id = req_body.get('document_id')
            filename = req_body.get('filename')
            results = req_body.get('results')
            
            if not all([document_id, filename, results]):
                logger.error("Missing required parameters: document_id, filename, results")
                return func.HttpResponse(
                    json.dumps({"error": "Missing required parameters: document_id, filename, results"}),
                    status_code=400,
                    mimetype="application/json"
                )
            
            logger.info(f"Processing document: {document_id}, filename: {filename}")
            
            activity_logger.set_activity_info("document_id", document_id)
            activity_logger.set_activity_info("filename", filename)
            
            # Get configuration
            config = get_app_config_cached()
            
            if not config.cosmos_db:
                logger.info("Cosmos DB not configured, skipping write")
                return func.HttpResponse(
                    json.dumps({"message": "Cosmos DB not configured"}),
                    status_code=200,
                    mimetype="application/json"
                )
            
            # Initialize Cosmos DB client
            logger.info("=== Initializing Cosmos DB Connection ===")
            cosmos_client = create_cosmos_client(config.cosmos_db)
            
            # Create metadata for Cosmos write
            metadata = {
                "document_id": document_id,
                "filename": filename,
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "result_type": "combined_results",
                "full_blob_name": f"{document_id}_{filename}"
            }
            
            # Write to Cosmos DB (4 different tables)
            logger.info("Writing complete document results to 4 Cosmos DB tables")
            write_success = write_to_multiple_cosmos_tables(
                cosmos_client, 
                config.cosmos_db, 
                results, 
                metadata
            )
            
            if write_success:
                logger.info("=== Successfully wrote document to Cosmos DB ===")
                activity_logger.set_activity_info("write_success", True)
                
                response = {
                    "document_id": document_id,
                    "filename": filename,
                    "cosmos_write_success": True,
                    "message": "Document successfully written to Cosmos DB"
                }
                
                return func.HttpResponse(
                    json.dumps(response),
                    status_code=200,
                    mimetype="application/json"
                )
            else:
                logger.error("Failed to write document to Cosmos DB")
                activity_logger.set_activity_info("write_success", False)
                
                return func.HttpResponse(
                    json.dumps({"error": "Failed to write to Cosmos DB"}),
                    status_code=500,
                    mimetype="application/json"
                )
                
        except Exception as e:
            logger.error(f"Cosmos writer function failed: {str(e)}")
            activity_logger.error(f"Function error: {str(e)}")
            return func.HttpResponse(
                json.dumps({"error": f"Processing failed: {str(e)}"}),
                status_code=500,
                mimetype="application/json"
            )


def parse_blob_name(blob_name: str) -> Optional[Dict[str, str]]:
    """
    Parse blob name to extract document metadata.
    
    Expected formats:
    - contract-processing/processed-documents/20250901_185805_test-full-123_test-full/extracted_contract.json
    - contract-processing/processed-documents/20250901_185805_test-full-123_test-full/combined_results.json
    """
    try:
        # Remove container prefix if present
        if blob_name.startswith("contract-processing/processed-documents/"):
            blob_name = blob_name[len("contract-processing/processed-documents/"):]
        
        # Split path into folder and file
        parts = blob_name.split('/')
        if len(parts) < 2:
            return None
        
        folder_name = parts[0]
        file_name = parts[1]
        
        # Parse folder name: {timestamp}_{document_id}_{original_filename}
        folder_parts = folder_name.split('_', 2)
        if len(folder_parts) < 3:
            return None
        
        timestamp_str = folder_parts[0] + '_' + folder_parts[1]  # Reconstruct timestamp
        document_id = folder_parts[2].split('_')[0]  # Extract document ID
        original_filename = '_'.join(folder_parts[2].split('_')[1:])  # Rest is filename
        
        # Get result type from file name
        result_type = file_name.replace('.json', '')
        
        return {
            "timestamp": timestamp_str,
            "document_id": document_id,
            "filename": original_filename,
            "result_type": result_type,
            "full_blob_name": blob_name
        }
        
    except Exception as e:
        logger.error(f"Failed to parse blob name {blob_name}: {str(e)}")
        return None


def create_cosmos_client(cosmos_config):
    """Create Cosmos DB client."""
    try:
        from azure.cosmos import CosmosClient
        
        client = CosmosClient(
            url=cosmos_config.endpoint,
            credential=cosmos_config.key
        )
        
        # Ensure database and container exist
        database = client.create_database_if_not_exists(cosmos_config.database_name)
        container = database.create_container_if_not_exists(
            id=cosmos_config.container_name,
            partition_key={"paths": [cosmos_config.partition_key], "kind": "Hash"}
        )
        
        logger.info(f"Cosmos DB client initialized - Database: {cosmos_config.database_name}, Container: {cosmos_config.container_name}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create Cosmos DB client: {str(e)}")
        raise


def write_to_multiple_cosmos_tables(
    cosmos_client, 
    cosmos_config, 
    data: Dict[str, Any], 
    metadata: Dict[str, str]
) -> bool:
    """Write document data to 4 different Cosmos DB tables with 5 columns each."""
    try:
        logger.info(f"Writing document {metadata['document_id']} to 4 Cosmos DB tables")
        
        # Get database
        database = cosmos_client.get_database_client(cosmos_config.database_name)
        
        # Define table configurations
        tables = [
            {"name": "documents_table", "columns": ["doc_id", "doc_name", "doc_status", "doc_type", "doc_timestamp"]},
            {"name": "contracts_table", "columns": ["contract_id", "policy_number", "coverage_amount", "premium_amount", "contract_terms"]},
            {"name": "chunks_table", "columns": ["chunk_id", "chunk_text", "chunk_embeddings", "chunk_metadata", "chunk_order"]},
            {"name": "processing_table", "columns": ["process_id", "process_time", "process_status", "process_metrics", "process_errors"]}
        ]
        
        write_results = []
        
        for table_config in tables:
            try:
                # Create container if not exists
                container = database.create_container_if_not_exists(
                    id=table_config["name"],
                    partition_key={"paths": ["/partition_key"], "kind": "Hash"}
                )
                
                # Extract data for this table
                table_data = extract_table_data(data, metadata, table_config)
                
                # Write to table
                result = container.upsert_item(table_data)
                logger.info(f"Successfully wrote to {table_config['name']} with ID: {result['id']}")
                write_results.append(True)
                
            except Exception as e:
                logger.error(f"Failed to write to {table_config['name']}: {str(e)}")
                write_results.append(False)
        
        # Return True if all writes succeeded
        success = all(write_results)
        logger.info(f"Multi-table write completed: {sum(write_results)}/{len(tables)} tables successful")
        return success
        
    except Exception as e:
        logger.error(f"Failed to write to multiple Cosmos DB tables: {str(e)}")
        return False


def extract_table_data(data: Dict[str, Any], metadata: Dict[str, str], table_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and map data to table columns."""
    
    base_document = {
        "id": f"{metadata['document_id']}_{metadata['timestamp']}_{table_config['name']}",
        "partition_key": metadata['document_id'],
        "created_timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    if table_config["name"] == "documents_table":
        # Map to: doc_id, doc_name, doc_status, doc_type, doc_timestamp
        return {
            **base_document,
            "doc_id": metadata['document_id'],
            "doc_name": metadata['filename'],
            "doc_status": "processed",
            "doc_type": data.get("metadata", {}).get("content_type", "unknown"),
            "doc_timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    elif table_config["name"] == "contracts_table":
        # Map to: contract_id, policy_number, coverage_amount, premium_amount, contract_terms
        contract_data = data.get("contract_extraction", {}).get("extraction_result", {}).get("extracted_contract", {})
        
        return {
            **base_document,
            "contract_id": metadata['document_id'],
            "policy_number": contract_data.get("unique_market_reference", {}).get("policy_number", ""),
            "coverage_amount": str(contract_data.get("limits", [])),
            "premium_amount": str(contract_data.get("premiums", [])),
            "contract_terms": str(contract_data.get("coverages", []))
        }
        
    elif table_config["name"] == "chunks_table":
        # Map to: chunk_id, chunk_text, chunk_embeddings, chunk_metadata, chunk_order
        chunk_data = data.get("chunk_embed_result", {})
        chunks = chunk_data.get("chunks", [])
        
        # For multiple chunks, create a summary document
        return {
            **base_document,
            "chunk_id": f"{metadata['document_id']}_chunks",
            "chunk_text": f"Total chunks: {len(chunks)}",
            "chunk_embeddings": str(chunk_data.get("embedding_stats", {})),
            "chunk_metadata": str(chunk_data.get("processing_stats", {})),
            "chunk_order": len(chunks)
        }
        
    elif table_config["name"] == "processing_table":
        # Map to: process_id, process_time, process_status, process_metrics, process_errors
        timing_stats = data.get("timing_stats", {})
        
        return {
            **base_document,
            "process_id": f"{metadata['document_id']}_processing",
            "process_time": timing_stats.get("total_processing_time", 0),
            "process_status": "completed",
            "process_metrics": str(extract_document_metrics(data)),
            "process_errors": str(data.get("errors", []))
        }
    
    else:
        # Default mapping
        return base_document


def write_document_to_cosmos(
    cosmos_client, 
    cosmos_config, 
    data: Dict[str, Any], 
    blob_metadata: Dict[str, str]
) -> bool:
    """Write complete document processing results to Cosmos DB."""
    try:
        logger.info(f"Writing document {blob_metadata['document_id']} to Cosmos DB")
        
        # Get database and container
        database = cosmos_client.get_database_client(cosmos_config.database_name)
        container = database.get_container_client(cosmos_config.container_name)
        
        # Create Cosmos DB document structure
        cosmos_document = {
            "id": f"{blob_metadata['document_id']}_{blob_metadata['timestamp']}",
            "document_id": blob_metadata['document_id'],
            "filename": blob_metadata['filename'],
            "processed_timestamp": datetime.now(timezone.utc).isoformat(),
            "blob_source": blob_metadata['full_blob_name'],
            "processing_type": "complete_document",
            
            # Core processing results
            "document_data": data,
            
            # Extract key metrics for easy querying
            "metrics": extract_document_metrics(data),
            
            # Document classification
            "document_type": "insurance_contract",
            "processing_status": "completed",
            
            # Partition key
            "partition_key": blob_metadata['document_id']
        }
        
        # Upsert to Cosmos DB
        result = container.upsert_item(cosmos_document)
        logger.info(f"Document written to Cosmos DB with ID: {result['id']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write document to Cosmos DB: {str(e)}")
        return False


def write_component_to_cosmos(
    cosmos_client, 
    cosmos_config, 
    data: Dict[str, Any], 
    blob_metadata: Dict[str, str]
) -> bool:
    """Write individual component results to Cosmos DB."""
    try:
        logger.info(f"Writing {blob_metadata['result_type']} component for {blob_metadata['document_id']} to Cosmos DB")
        
        # Get database and container
        database = cosmos_client.get_database_client(cosmos_config.database_name)
        container = database.get_container_client(cosmos_config.container_name)
        
        # Create component document
        cosmos_document = {
            "id": f"{blob_metadata['document_id']}_{blob_metadata['timestamp']}_{blob_metadata['result_type']}",
            "document_id": blob_metadata['document_id'],
            "filename": blob_metadata['filename'],
            "processed_timestamp": datetime.now(timezone.utc).isoformat(),
            "blob_source": blob_metadata['full_blob_name'],
            "processing_type": "component",
            "component_type": blob_metadata['result_type'],
            
            # Component data
            "component_data": data,
            
            # Component-specific metadata
            "metrics": extract_component_metrics(data, blob_metadata['result_type']),
            
            # Partition key
            "partition_key": blob_metadata['document_id']
        }
        
        # Upsert to Cosmos DB
        result = container.upsert_item(cosmos_document)
        logger.info(f"Component {blob_metadata['result_type']} written to Cosmos DB with ID: {result['id']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to write component to Cosmos DB: {str(e)}")
        return False


def extract_document_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract key metrics from document processing results for easy querying."""
    metrics = {
        "processing_time_seconds": 0,
        "total_chunks": 0,
        "embedding_success_rate": 0.0,
        "contract_extraction_success": False,
        "extraction_elements_found": 0
    }
    
    try:
        # Extract timing stats
        if "timing_stats" in data:
            metrics["processing_time_seconds"] = data["timing_stats"].get("total_processing_time", 0)
        
        # Extract chunk/embedding stats
        if "chunk_embed_result" in data:
            chunk_result = data["chunk_embed_result"]
            if "processing_stats" in chunk_result:
                metrics["total_chunks"] = chunk_result["processing_stats"].get("total_chunks", 0)
            if "embedding_stats" in chunk_result:
                metrics["embedding_success_rate"] = chunk_result["embedding_stats"].get("embedding_success_rate", 0.0)
        
        # Extract contract extraction stats
        if "contract_extraction" in data:
            contract_result = data["contract_extraction"]
            metrics["contract_extraction_success"] = contract_result.get("processing_stats", {}).get("extraction_success", False)
            
            # Count extracted elements
            if "extraction_result" in contract_result and "extracted_contract" in contract_result["extraction_result"]:
                contract_data = contract_result["extraction_result"]["extracted_contract"]
                element_count = 0
                element_count += len(contract_data.get("limits", []))
                element_count += len(contract_data.get("premiums", []))
                element_count += len(contract_data.get("coverages", []))
                element_count += len(contract_data.get("exclusions", []))
                metrics["extraction_elements_found"] = element_count
        
    except Exception as e:
        logger.warning(f"Failed to extract some metrics: {str(e)}")
    
    return metrics


def extract_component_metrics(data: Dict[str, Any], component_type: str) -> Dict[str, Any]:
    """Extract metrics specific to component type."""
    metrics = {"component_type": component_type}
    
    try:
        if component_type == "chunks":
            metrics["total_chunks"] = data.get("total_chunks", 0)
            metrics["chunks_count"] = len(data.get("chunks", []))
        elif component_type == "embeddings":
            metrics["embeddings_count"] = len(data.get("embeddings", []))
        elif component_type == "extracted_contract":
            if "extracted_contract" in data:
                contract_data = data["extracted_contract"]
                metrics["limits_count"] = len(contract_data.get("limits", []))
                metrics["premiums_count"] = len(contract_data.get("premiums", []))
                metrics["coverages_count"] = len(contract_data.get("coverages", []))
                metrics["exclusions_count"] = len(contract_data.get("exclusions", []))
                metrics["has_policy_number"] = bool(contract_data.get("unique_market_reference", {}).get("policy_number"))
        elif component_type == "metadata":
            if "processing_stats" in data:
                metrics.update(data["processing_stats"])
                
    except Exception as e:
        logger.warning(f"Failed to extract {component_type} metrics: {str(e)}")
    
    return metrics