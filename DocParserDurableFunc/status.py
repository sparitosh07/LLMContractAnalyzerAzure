"""
Status check endpoint for document processing orchestration.
"""

import logging
import json
import azure.functions as func
import azure.durable_functions as df
from datetime import datetime

logger = logging.getLogger(__name__)


async def main(req: func.HttpRequest, client: str) -> func.HttpResponse:
    """
    Check the status of a document processing orchestration.
    
    Route: GET /api/DocParserStatus/{instanceId}
    
    Returns:
    {
        "instanceId": "orchestration-instance-id",
        "runtimeStatus": "Running|Completed|Failed|Terminated",
        "input": {...},
        "output": {...},
        "createdTime": "2023-01-01T00:00:00Z",
        "lastUpdatedTime": "2023-01-01T00:00:00Z",
        "customStatus": {...}
    }
    """
    try:
        # Get instance ID from route
        instance_id = req.route_params.get('instanceId')
        
        if not instance_id:
            return func.HttpResponse(
                json.dumps({"error": "instanceId is required in the route"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logger.info(f"Checking status for orchestration instance: {instance_id}")
        
        # Create durable client
        durable_client = df.DurableOrchestrationClient(client)
        
        # Get orchestration status
        status = await durable_client.get_status(instance_id)
        
        if not status:
            return func.HttpResponse(
                json.dumps({
                    "error": f"No orchestration found with instance ID: {instance_id}"
                }),
                status_code=404,
                mimetype="application/json"
            )
        
        # Prepare response with enhanced status information
        response_data = {
            "instanceId": instance_id,
            "runtimeStatus": status.runtime_status,
            "createdTime": status.created_time.isoformat() if status.created_time else None,
            "lastUpdatedTime": status.last_updated_time.isoformat() if status.last_updated_time else None,
            "input": status.input,
            "output": status.output,
            "customStatus": status.custom_status
        }
        
        # Add processing progress information
        if status.runtime_status == "Running":
            response_data["message"] = "Document processing is in progress..."
            response_data["progress"] = "Processing document content and extracting information"
        elif status.runtime_status == "Completed":
            response_data["message"] = "Document processing completed successfully"
            
            # Add summary from output if available
            if status.output:
                output = status.output
                if isinstance(output, dict):
                    response_data["summary"] = {
                        "document_id": output.get("document_id"),
                        "filename": output.get("filename"),
                        "status_code": output.get("status_code"),
                        "processing_time": output.get("timing_stats", {}).get("total_processing_time"),
                        "chunks_generated": output.get("processing_stats", {}).get("total_chunks"),
                        "embeddings_created": output.get("embedding_stats", {}).get("embedded_chunks"),
                        "contract_extraction": "contract_extraction" in output,
                        "cosmos_db_write": output.get("cosmos_db_write", {}).get("success", False)
                    }
        elif status.runtime_status == "Failed":
            response_data["message"] = "Document processing failed"
            if status.output and isinstance(status.output, dict) and "error" in status.output:
                response_data["error_details"] = status.output["error"]
        elif status.runtime_status == "Terminated":
            response_data["message"] = "Document processing was terminated"
        else:
            response_data["message"] = f"Document processing status: {status.runtime_status}"
        
        logger.info(f"Status check completed for instance {instance_id}: {status.runtime_status}")
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=200,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Failed to check orchestration status: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": f"Failed to check status: {str(e)}",
                "instanceId": instance_id if 'instance_id' in locals() else None
            }),
            status_code=500,
            mimetype="application/json"
        )