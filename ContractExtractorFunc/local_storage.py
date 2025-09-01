"""Local file storage for development and testing - easily removable for production."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LocalStorageConfig:
    """Configuration for local file storage."""
    enabled: bool = True
    base_path: str = "../output"
    save_original_text: bool = True
    save_chunks: bool = True
    save_embeddings: bool = True
    save_metadata: bool = True
    create_timestamp_folders: bool = True
    
    @classmethod
    def from_env(cls) -> "LocalStorageConfig":
        """Create config from environment variables."""
        return cls(
            enabled=os.getenv("LOCAL_STORAGE_ENABLED", "true").lower() in ("true", "1", "yes"),
            base_path=os.getenv("LOCAL_STORAGE_PATH", "../output"),
            save_original_text=os.getenv("SAVE_ORIGINAL_TEXT", "true").lower() in ("true", "1", "yes"),
            save_chunks=os.getenv("SAVE_CHUNKS", "true").lower() in ("true", "1", "yes"),
            save_embeddings=os.getenv("SAVE_EMBEDDINGS", "true").lower() in ("true", "1", "yes"),
            save_metadata=os.getenv("SAVE_METADATA", "true").lower() in ("true", "1", "yes"),
            create_timestamp_folders=os.getenv("CREATE_TIMESTAMP_FOLDERS", "true").lower() in ("true", "1", "yes")
        )


class LocalFileSaver:
    """Modular local file saver for development - easily removable for production."""
    
    def __init__(self, config: LocalStorageConfig):
        """Initialize local file saver."""
        self.config = config
        
        if not self.config.enabled:
            logger.info("Local file saving is disabled")
            return
            
        # Create base directory
        self.base_path = Path(self.config.base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Local file saver initialized - Base path: {self.base_path}")
    
    def save_processing_results(
        self, 
        document_id: str,
        filename: str,
        original_text: str,
        chunks: List[Dict[str, Any]],
        embeddings: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        processing_stats: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Save all processing results to local files.
        
        Returns dictionary with file paths that were created.
        """
        if not self.config.enabled:
            logger.debug("Local saving disabled, skipping file save")
            return {}
            
        logger.info(f"=== Starting local file save for document: {document_id} ===")
        
        # Create document-specific folder
        doc_folder = self._create_document_folder(document_id, filename)
        saved_files = {}
        
        try:
            # Save original text
            if self.config.save_original_text and original_text:
                text_file = self._save_original_text(doc_folder, original_text, filename)
                saved_files["original_text"] = str(text_file)
                logger.info(f"Saved original text: {text_file}")
            
            # Save chunks
            if self.config.save_chunks and chunks:
                chunks_file = self._save_chunks(doc_folder, chunks)
                saved_files["chunks"] = str(chunks_file)
                logger.info(f"Saved {len(chunks)} chunks: {chunks_file}")
            
            # Save embeddings
            if self.config.save_embeddings and embeddings:
                embeddings_file = self._save_embeddings(doc_folder, embeddings)
                saved_files["embeddings"] = str(embeddings_file)
                logger.info(f"Saved {len(embeddings)} embeddings: {embeddings_file}")
            
            # Save metadata and stats
            if self.config.save_metadata:
                metadata_file = self._save_metadata(doc_folder, metadata, processing_stats, document_id, filename)
                saved_files["metadata"] = str(metadata_file)
                logger.info(f"Saved metadata: {metadata_file}")
            
            # Save summary
            summary_file = self._save_summary(doc_folder, document_id, filename, len(chunks), len(embeddings), saved_files)
            saved_files["summary"] = str(summary_file)
            
            logger.info(f"=== Local file save completed for {document_id} - Saved {len(saved_files)} files ===")
            return saved_files
            
        except Exception as e:
            logger.error(f"Failed to save files locally: {str(e)}")
            return saved_files  # Return partial results
    
    def _create_document_folder(self, document_id: str, filename: str) -> Path:
        """Create folder for document files."""
        if self.config.create_timestamp_folders:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            folder_name = f"{timestamp}_{document_id}_{Path(filename).stem}"
        else:
            folder_name = f"{document_id}_{Path(filename).stem}"
        
        doc_folder = self.base_path / folder_name
        doc_folder.mkdir(parents=True, exist_ok=True)
        
        logger.debug(f"Created document folder: {doc_folder}")
        return doc_folder
    
    def _save_original_text(self, doc_folder: Path, text: str, filename: str) -> Path:
        """Save original extracted text as .txt file."""
        text_file = doc_folder / "original_text.txt"
        
        text_file.write_text(text, encoding='utf-8')
        logger.debug(f"Saved original text ({len(text)} chars) to {text_file}")
        return text_file
    
    def _save_chunks(self, doc_folder: Path, chunks: List[Dict[str, Any]]) -> Path:
        """Save document chunks as JSON."""
        chunks_file = doc_folder / "chunks.json"
        
        # Prepare chunks for JSON serialization (remove embeddings to keep file manageable)
        chunks_for_save = []
        for chunk in chunks:
            chunk_copy = chunk.copy()
            if "embedding" in chunk_copy:
                del chunk_copy["embedding"]  # Save embeddings separately
            chunks_for_save.append(chunk_copy)
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_chunks": len(chunks_for_save),
                "chunks": chunks_for_save
            }, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved {len(chunks)} chunks to {chunks_file}")
        return chunks_file
    
    def _save_embeddings(self, doc_folder: Path, embeddings: List[Dict[str, Any]]) -> Path:
        """Save embeddings as JSON."""
        embeddings_file = doc_folder / "embeddings.json"
        
        # Extract just the embeddings and basic metadata
        embeddings_data = []
        for emb in embeddings:
            embedding_entry = {
                "id": emb.get("id"),
                "chunk_index": emb.get("chunk_index"),
                "embedding": emb.get("embedding", []),
                "content_preview": emb.get("content", "")[:100] + "..." if len(emb.get("content", "")) > 100 else emb.get("content", "")
            }
            embeddings_data.append(embedding_entry)
        
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_embeddings": len(embeddings_data),
                "embedding_dimension": len(embeddings_data[0]["embedding"]) if embeddings_data and embeddings_data[0]["embedding"] else 0,
                "embeddings": embeddings_data
            }, f, indent=2)
        
        logger.debug(f"Saved {len(embeddings)} embeddings to {embeddings_file}")
        return embeddings_file
    
    def _save_metadata(
        self, 
        doc_folder: Path, 
        metadata: Dict[str, Any], 
        processing_stats: Dict[str, Any],
        document_id: str,
        filename: str
    ) -> Path:
        """Save metadata and processing statistics."""
        metadata_file = doc_folder / "metadata.json"
        
        full_metadata = {
            "document_info": {
                "document_id": document_id,
                "filename": filename,
                "processed_at": datetime.now().isoformat(),
            },
            "processing_stats": processing_stats,
            "metadata": metadata,
            "config_used": {
                "local_storage_enabled": self.config.enabled,
                "base_path": str(self.base_path)
            }
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved metadata to {metadata_file}")
        return metadata_file
    
    def _save_summary(
        self, 
        doc_folder: Path, 
        document_id: str,
        filename: str,
        num_chunks: int,
        num_embeddings: int,
        saved_files: Dict[str, str]
    ) -> Path:
        """Save processing summary."""
        summary_file = doc_folder / "summary.txt"
        
        summary_content = f"""Document Processing Summary
===========================

Document ID: {document_id}
Filename: {filename}
Processed: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

Results:
- Chunks generated: {num_chunks}
- Embeddings created: {num_embeddings}
- Files saved: {len(saved_files)}

Files created:
"""
        
        for file_type, file_path in saved_files.items():
            summary_content += f"- {file_type}: {file_path}\n"
        
        summary_file.write_text(summary_content, encoding='utf-8')
        logger.debug(f"Saved summary to {summary_file}")
        return summary_file
    
    def cleanup_old_files(self, days_old: int = 7) -> int:
        """Clean up files older than specified days."""
        if not self.config.enabled or not self.base_path.exists():
            return 0
        
        logger.info(f"Cleaning up files older than {days_old} days from {self.base_path}")
        
        cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        deleted_count = 0
        
        for folder in self.base_path.iterdir():
            if folder.is_dir():
                folder_time = folder.stat().st_mtime
                if folder_time < cutoff_time:
                    import shutil
                    shutil.rmtree(folder)
                    deleted_count += 1
                    logger.debug(f"Deleted old folder: {folder}")
        
        logger.info(f"Cleanup completed - Deleted {deleted_count} old folders")
        return deleted_count
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        if not self.config.enabled or not self.base_path.exists():
            return {"enabled": False}
        
        total_folders = sum(1 for p in self.base_path.iterdir() if p.is_dir())
        total_size = sum(f.stat().st_size for f in self.base_path.rglob('*') if f.is_file())
        
        return {
            "enabled": True,
            "base_path": str(self.base_path),
            "total_folders": total_folders,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }


def create_local_file_saver(config: LocalStorageConfig) -> LocalFileSaver:
    """Factory function to create LocalFileSaver instance."""
    logger.info("Creating local file saver instance")
    return LocalFileSaver(config)


# Development helper functions
def enable_local_storage():
    """Enable local storage (for development use)."""
    os.environ["LOCAL_STORAGE_ENABLED"] = "true"
    logger.info("Local storage enabled via environment variable")


def disable_local_storage():
    """Disable local storage (for production deployment)."""
    os.environ["LOCAL_STORAGE_ENABLED"] = "false"
    logger.info("Local storage disabled via environment variable")