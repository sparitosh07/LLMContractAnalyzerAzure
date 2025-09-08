"""
HTTP Starter for Durable Document Processing Function.
Receives document processing requests and starts orchestration.
"""

import logging
import json
import uuid
import azure.functions as func
import azure.durable_functions as df

logger = logging.getLogger(__name__)


async def main(req: func.HttpRequest, starter: str) -> func.HttpResponse:
    """
    HTTP Starter for document processing orchestration.
    
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
    
    Returns:
    {
        "instanceId": "orchestration-instance-id",
        "statusQueryGetUri": "http://...",
        "sendEventPostUri": "http://...",
        "terminatePostUri": "http://...",
        "purgeHistoryDeleteUri": "http://..."
    }
    """
    try:
        logger.info("=== Starting Durable Document Processing Request ===")
        
        # Parse and validate request
        req_body = req.get_json()
        if not req_body:
            logger.error("No request body provided")
            return func.HttpResponse(
                json.dumps({"error": "Request body is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        # Extract and validate parameters
        filename = req_body.get('filename', 'document.txt')
        document_id = req_body.get('document_id', str(uuid.uuid4()))
        content_type = req_body.get('content_type', 'text/plain')
        
        # Validate required content
        has_text = req_body.get('text')
        has_pdf = req_body.get('pdf_base64')
        
        if not has_text and not has_pdf:
            return func.HttpResponse(
                json.dumps({"error": "Either 'text' or 'pdf_base64' content is required"}),
                status_code=400,
                mimetype="application/json"
            )
        
        logger.info(f"Starting orchestration - Document ID: {document_id}, Filename: {filename}")
        
        # Create orchestrator input
        orchestrator_input = {
            "document_id": document_id,
            "filename": filename,
            "content_type": content_type,
            "text": req_body.get('text'),
            "pdf_base64": req_body.get('pdf_base64'),
            "use_layout_model": req_body.get('use_layout_model', True),
            "extract_contract": req_body.get('extract_contract', False),
            "chunk_size": req_body.get('chunk_size', 1000),
            "chunk_overlap": req_body.get('chunk_overlap', 200),
            "extraction_chunk_size": req_body.get('extraction_chunk_size', 2000),
            "extraction_chunk_overlap": req_body.get('extraction_chunk_overlap', 400)
        }
        
        # Start orchestration
        client = df.DurableOrchestrationClient(starter)
        instance_id = await client.start_new(
            orchestration_function_name="document_processing_orchestrator",
            client_input=orchestrator_input
        )
        
        logger.info(f"Started orchestration with instance ID: {instance_id}")
        
        # Return management URLs
        response = client.create_check_status_response(req, instance_id)
        
        # Add custom response data
        response_data = json.loads(response.get_body())
        response_data["instanceId"] = instance_id
        response_data["document_id"] = document_id
        response_data["filename"] = filename
        
        return func.HttpResponse(
            json.dumps(response_data, indent=2),
            status_code=202,
            mimetype="application/json"
        )
        
    except Exception as e:
        logger.error(f"Failed to start orchestration: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": f"Failed to start processing: {str(e)}"}),
            status_code=500,
            mimetype="application/json"
        )