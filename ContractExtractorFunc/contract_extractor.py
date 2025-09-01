"""
LangGraph-based contract extraction using map-reduce methodology.
Extracts structured insurance contract data from text files.
"""

import json
import asyncio
import os
import time
from typing import List, Dict, Any, Optional, TypedDict
from dataclasses import dataclass
from pathlib import Path

from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langsmith import Client as LangSmithClient
from openai import AzureOpenAI

from .pydantic_models import InsuranceContract
from .utils.logging_utils import get_logger

logger = get_logger(__name__)


# Define LangGraph state schemas
class InputState(TypedDict):
    """Input schema for contract extraction"""
    text_content: str
    filename: str
    document_id: str
    chunk_size: int


class OutputState(TypedDict):
    """Output schema for contract extraction"""
    extracted_contract: Dict[str, Any]
    extraction_metadata: Dict[str, Any]


class InternalState(InputState, OutputState):
    """Internal state combining input and output"""
    text_chunks: List[str]
    chunk_extractions: List[Dict[str, Any]]
    processing_stats: Dict[str, Any]
    start_time: float
    timing_stats: Dict[str, float]


@dataclass
class ContractExtractionConfig:
    """Configuration for contract extraction"""
    chunk_size: int = 2000
    chunk_overlap: int = 400
    max_tokens_per_chunk: int = 4000
    temperature: float = 0.0
    max_retries: int = 3


class ContractExtractor:
    """LangGraph-based contract extractor using map-reduce methodology"""
    
    def __init__(self, openai_config, extraction_config: ContractExtractionConfig):
        """Initialize contract extractor"""
        self.config = extraction_config
        self.openai_config = openai_config
        
        # Initialize LangSmith client for monitoring
        self.langsmith_enabled = False
        langsmith_api_key = os.getenv("LANGSMITH_API_KEY")
        if langsmith_api_key:
            try:
                # Set environment variable for LangChain integration
                os.environ["LANGCHAIN_TRACING_V2"] = "true"
                os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
                os.environ["LANGCHAIN_PROJECT"] = "contract-extraction"
                self.langsmith_enabled = True
                logger.info("LangSmith monitoring enabled with project: contract-extraction")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith: {str(e)}")
        else:
            logger.info("LangSmith API key not found, monitoring disabled")
        
        # Initialize Azure OpenAI client for structured extraction
        # Use a chat model instead of embedding model
        chat_deployment = os.getenv("OPENAI_CHAT_DEPLOYMENT_NAME", "gpt-4o")
        self.llm = AzureChatOpenAI(
            api_key=openai_config.api_key,
            api_version=openai_config.api_version,
            azure_endpoint=openai_config.endpoint,
            deployment_name=chat_deployment,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens_per_chunk
        )
        logger.info(f"Initialized LLM with chat deployment: {chat_deployment}")
        
        # Build LangGraph workflow
        self.graph = self._build_extraction_graph()
        logger.info("Contract extractor initialized with LangGraph workflow")
    
    def _build_extraction_graph(self) -> StateGraph:
        """Build LangGraph workflow for contract extraction"""
        
        # Create graph with input/output schemas
        builder = StateGraph(
            InternalState,
            input_schema=InputState,
            output_schema=OutputState
        )
        
        # Add nodes for map-reduce workflow
        builder.add_node("split_document", self._split_document_node)
        builder.add_node("map_extract", self._map_extract_node)
        builder.add_node("reduce_combine", self._reduce_combine_node)
        builder.add_node("validate_output", self._validate_output_node)
        
        # Define workflow edges
        builder.set_entry_point("split_document")
        builder.add_edge("split_document", "map_extract")
        builder.add_edge("map_extract", "reduce_combine")
        builder.add_edge("reduce_combine", "validate_output")
        builder.set_finish_point("validate_output")
        
        return builder.compile()
    
    def _split_document_node(self, state: InternalState) -> InternalState:
        """Split document into overlapping chunks for map phase"""
        split_start = time.time()
        logger.info("=== Document Splitting Phase ===")
        
        text = state["text_content"]
        chunk_size = state.get("chunk_size", self.config.chunk_size)
        overlap = self.config.chunk_overlap
        
        # Initialize timing if not present
        if "timing_stats" not in state:
            state["timing_stats"] = {}
        if "start_time" not in state:
            state["start_time"] = time.time()
        
        # Simple text splitting with overlap
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # Ensure we don't split mid-word if possible
            if end < len(text) and not text[end].isspace():
                last_space = chunk.rfind(' ')
                if last_space > start + chunk_size * 0.8:  # Don't go too far back
                    end = start + last_space
                    chunk = text[start:end]
            
            chunks.append(chunk.strip())
            
            if end >= len(text):
                break
                
            start = end - overlap
        
        logger.info(f"Split document into {len(chunks)} chunks (avg size: {sum(len(c) for c in chunks) // len(chunks)} chars)")
        
        state["text_chunks"] = chunks
        state["processing_stats"] = {
            "total_chunks": len(chunks),
            "average_chunk_size": sum(len(c) for c in chunks) // len(chunks) if chunks else 0,
            "total_characters": len(text)
        }
        state["timing_stats"]["splitting_time"] = time.time() - split_start
        
        return state
    
    def _map_extract_node(self, state: InternalState) -> InternalState:
        """Map phase: Extract contract elements from each chunk"""
        map_start = time.time()
        logger.info("=== Map Extraction Phase ===")
        
        chunks = state["text_chunks"]
        chunk_extractions = []
        
        extraction_prompt = self._get_extraction_prompt()
        
        # Track LangSmith session if available
        session_id = f"contract_extraction_{state['document_id']}" if self.langsmith_enabled else None
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            try:
                # Create extraction prompt for this chunk
                messages = [
                    HumanMessage(content=f"{extraction_prompt}\n\nDocument Text Chunk:\n{chunk}")
                ]
                
                # Get LLM response with LangSmith tracking
                chunk_start = time.time()
                
                if self.langsmith_enabled and session_id:
                    # LangSmith will automatically track via environment variables
                    logger.info(f"LangSmith tracking enabled for session: {session_id}")
                
                response = self.llm.invoke(messages)
                
                chunk_time = time.time() - chunk_start
                
                # Parse JSON response
                try:
                    # Clean response content to ensure valid JSON
                    content = response.content.strip()
                    if content.startswith("```json"):
                        content = content[7:-3].strip()
                    elif content.startswith("```"):
                        content = content[3:-3].strip()
                    
                    extraction = json.loads(content)
                    chunk_extractions.append({
                        "chunk_index": i,
                        "extraction": extraction,
                        "success": True,
                        "processing_time": chunk_time
                    })
                    logger.debug(f"Successfully extracted from chunk {i+1} in {chunk_time:.2f}s")
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from chunk {i+1}: {str(e)}")
                    logger.warning(f"Raw response: {response.content[:200]}...")
                    chunk_extractions.append({
                        "chunk_index": i,
                        "extraction": {},
                        "success": False,
                        "error": str(e),
                        "raw_response": response.content[:500],
                        "processing_time": chunk_time
                    })
                    
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                chunk_extractions.append({
                    "chunk_index": i,
                    "extraction": {},
                    "success": False,
                    "error": str(e)
                })
        
        successful_extractions = sum(1 for ext in chunk_extractions if ext["success"])
        total_chunk_time = sum(ext.get("processing_time", 0) for ext in chunk_extractions)
        logger.info(f"Map phase completed: {successful_extractions}/{len(chunks)} chunks processed successfully in {total_chunk_time:.2f}s")
        
        state["chunk_extractions"] = chunk_extractions
        state["processing_stats"]["successful_extractions"] = successful_extractions
        state["timing_stats"]["map_extraction_time"] = time.time() - map_start
        state["timing_stats"]["total_chunk_processing_time"] = total_chunk_time
        
        return state
    
    def _reduce_combine_node(self, state: InternalState) -> InternalState:
        """Reduce phase: Combine extractions from all chunks"""
        reduce_start = time.time()
        logger.info("=== Reduce Combination Phase ===")
        
        chunk_extractions = state["chunk_extractions"]
        
        # Initialize combined data structure
        combined_data = {
            "unique_market_reference": {},
            "limits": [],
            "premiums": [],
            "coverages": [],
            "exclusions": []
        }
        
        # Combine extractions from all successful chunks
        for chunk_ext in chunk_extractions:
            if not chunk_ext["success"]:
                continue
                
            extraction = chunk_ext["extraction"]
            
            # Merge unique market references (take first non-empty values)
            if "unique_market_reference" in extraction:
                umr = extraction["unique_market_reference"]
                for key, value in umr.items():
                    if value and not combined_data["unique_market_reference"].get(key):
                        combined_data["unique_market_reference"][key] = value
            
            # Aggregate lists (limits, premiums, coverages, exclusions)
            for list_field in ["limits", "premiums", "coverages", "exclusions"]:
                if list_field in extraction and isinstance(extraction[list_field], list):
                    combined_data[list_field].extend(extraction[list_field])
        
        # Deduplicate and clean up
        combined_data = self._deduplicate_extractions(combined_data)
        
        logger.info(f"Reduce phase completed - Combined: {len(combined_data['limits'])} limits, "
                   f"{len(combined_data['premiums'])} premiums, {len(combined_data['coverages'])} coverages, "
                   f"{len(combined_data['exclusions'])} exclusions")
        
        state["extracted_contract"] = combined_data
        
        # Generate session ID for LangSmith tracking
        session_id = f"contract_extraction_{state['document_id']}" if self.langsmith_enabled else None
        
        state["extraction_metadata"] = {
            "chunks_processed": len(chunk_extractions),
            "successful_chunks": sum(1 for ext in chunk_extractions if ext["success"]),
            "extraction_method": "langgraph_map_reduce",
            "model_used": getattr(self.llm, 'model_name', 'gpt-4o'),
            "langsmith_session": session_id if self.langsmith_enabled else None
        }
        state["timing_stats"]["reduce_combine_time"] = time.time() - reduce_start
        
        return state
    
    def _validate_output_node(self, state: InternalState) -> InternalState:
        """Validate and finalize extracted contract data"""
        validation_start = time.time()
        logger.info("=== Validation Phase ===")
        
        try:
            # Validate against pydantic model
            contract_data = state["extracted_contract"]
            validated_contract = InsuranceContract(**contract_data)
            
            # Convert back to dict for JSON serialization
            state["extracted_contract"] = validated_contract.dict()
            state["extraction_metadata"]["validation_success"] = True
            
            logger.info("Contract validation successful")
            
        except Exception as e:
            logger.warning(f"Validation failed: {str(e)}")
            state["extraction_metadata"]["validation_success"] = False
            state["extraction_metadata"]["validation_error"] = str(e)
        
        # Calculate total processing time
        total_time = time.time() - state["start_time"]
        state["timing_stats"]["validation_time"] = time.time() - validation_start
        state["timing_stats"]["total_processing_time"] = total_time
        
        # Add timing to extraction metadata
        state["extraction_metadata"]["timing_stats"] = state["timing_stats"]
        
        logger.info(f"Total extraction processing time: {total_time:.2f}s")
        
        return state
    
    def _get_extraction_prompt(self) -> str:
        """Get extraction prompt with schema and instructions"""
        schema = InsuranceContract.get_extraction_schema()
        instructions = InsuranceContract.get_extraction_instructions()
        
        return f"""
You are an expert insurance contract analyzer. Extract structured information from the provided document text chunk.

{instructions}

IMPORTANT: 
- Only extract information that is explicitly present in this text chunk
- Use null for missing information
- Return valid JSON only
- Be precise and concise

Target JSON Schema:
{json.dumps(schema, indent=2)}

Extract the following structure from the text chunk:
{{
    "unique_market_reference": {{
        "policy_number": null,
        "quote_number": null,
        "broker_reference": null,
        "insurer_reference": null,
        "umr_code": null,
        "certificate_number": null,
        "endorsement_numbers": []
    }},
    "limits": [],
    "premiums": [],
    "coverages": [],
    "exclusions": []
}}
"""
    
    def _deduplicate_extractions(self, combined_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove duplicate entries from combined extractions"""
        
        # Deduplicate limits by description
        seen_limits = set()
        unique_limits = []
        for limit in combined_data["limits"]:
            limit_key = (limit.get("description", ""), limit.get("amount", {}).get("value"))
            if limit_key not in seen_limits:
                seen_limits.add(limit_key)
                unique_limits.append(limit)
        combined_data["limits"] = unique_limits
        
        # Deduplicate premiums by type and amount
        seen_premiums = set()
        unique_premiums = []
        for premium in combined_data["premiums"]:
            premium_key = (premium.get("premium_type", ""), premium.get("amount", {}).get("value"))
            if premium_key not in seen_premiums:
                seen_premiums.add(premium_key)
                unique_premiums.append(premium)
        combined_data["premiums"] = unique_premiums
        
        # Deduplicate coverages by name
        seen_coverages = set()
        unique_coverages = []
        for coverage in combined_data["coverages"]:
            coverage_key = coverage.get("coverage_name", "")
            if coverage_key not in seen_coverages and coverage_key:
                seen_coverages.add(coverage_key)
                unique_coverages.append(coverage)
        combined_data["coverages"] = unique_coverages
        
        # Deduplicate exclusions by title
        seen_exclusions = set()
        unique_exclusions = []
        for exclusion in combined_data["exclusions"]:
            exclusion_key = exclusion.get("title", "")
            if exclusion_key not in seen_exclusions and exclusion_key:
                seen_exclusions.add(exclusion_key)
                unique_exclusions.append(exclusion)
        combined_data["exclusions"] = unique_exclusions
        
        return combined_data
    
    def extract_contract(self, text_content: str, filename: str, document_id: str) -> Dict[str, Any]:
        """
        Extract structured contract data using LangGraph map-reduce workflow.
        
        Args:
            text_content: Raw text from PDF
            filename: Original filename
            document_id: Unique document identifier
            
        Returns:
            Dictionary with extracted contract data and metadata
        """
        logger.info(f"Starting LangGraph contract extraction for {filename}")
        
        # Prepare input state
        input_state = {
            "text_content": text_content,
            "filename": filename,
            "document_id": document_id,
            "chunk_size": self.config.chunk_size
        }
        
        try:
            # Execute LangGraph workflow
            result = self.graph.invoke(input_state)
            
            logger.info("LangGraph extraction completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"LangGraph extraction failed: {str(e)}")
            return {
                "extracted_contract": {},
                "extraction_metadata": {
                    "success": False,
                    "error": str(e),
                    "extraction_method": "langgraph_map_reduce"
                }
            }


def create_contract_extractor(openai_config, extraction_config: Optional[ContractExtractionConfig] = None) -> ContractExtractor:
    """Factory function to create ContractExtractor instance"""
    if extraction_config is None:
        extraction_config = ContractExtractionConfig()
    
    logger.info("Creating contract extractor with LangGraph")
    return ContractExtractor(openai_config, extraction_config)