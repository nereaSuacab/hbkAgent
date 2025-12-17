#!/usr/bin/env python3
"""
Modified ingestion script that creates TWO separate indices:
1. Dense/Vector index with 512 token chunks
2. BM25 index with 256 token chunks

This allows each retrieval method to work with optimal chunk sizes.
"""

import argparse
import logging
from pathlib import Path

from private_gpt.di import global_injector
from private_gpt.settings.settings import Settings

from private_gpt.server.ingest.dual_ingest_service import DualIngestService

logger = logging.getLogger(__name__)


class DualIngestWorker:
    """
    Worker that ingests files into BOTH dense and BM25 indices
    with different chunking strategies.
    """
    
    def __init__(self, dual_ingest_service: DualIngestService, setting: Settings) -> None:
        self.dual_ingest_service = dual_ingest_service

        self.total_documents = 0
        self.current_document_count = 0

        self._files_under_root_folder: list[Path] = []

        self.is_local_ingestion_enabled = setting.data.local_ingestion.enabled
        self.allowed_local_folders = setting.data.local_ingestion.allow_ingest_from

    def _validate_folder(self, folder_path: Path) -> None:
        if not self.is_local_ingestion_enabled:
            raise ValueError(
                "Local ingestion is disabled. "
                "You can enable it in settings `ingestion.enabled`"
            )

        # Allow all folders if wildcard is present
        if "*" in self.allowed_local_folders:
            return

        for allowed_folder in self.allowed_local_folders:
            if not folder_path.is_relative_to(allowed_folder):
                raise ValueError(f"Folder {folder_path} is not allowed for ingestion")

    def _find_all_files_in_folder(self, root_path: Path, ignored: list[str]) -> None:
        """Search all files under the root folder recursively."""
        for file_path in root_path.iterdir():
            if file_path.is_file() and file_path.name not in ignored:
                self.total_documents += 1
                self._validate_folder(file_path)
                self._files_under_root_folder.append(file_path)
            elif file_path.is_dir() and file_path.name not in ignored:
                self._find_all_files_in_folder(file_path, ignored)

    def ingest_folder(self, folder_path: Path, ignored: list[str]) -> None:
        """Ingest all files in folder into BOTH indices."""
        logger.info("=" * 60)
        logger.info("DUAL INGESTION STARTED")
        logger.info("=" * 60)
        logger.info(f"Folder: {folder_path}")
        logger.info(f"Each file will be chunked TWICE:")
        logger.info(f"  1. Dense/Vector: 512 token chunks (semantic search)")
        logger.info(f"  2. BM25: 256 token chunks (keyword search)")
        logger.info("=" * 60)
        
        # Count total documents
        self._find_all_files_in_folder(folder_path, ignored)
        logger.info(f"Found {self.total_documents} files to ingest")
        
        # Ingest all files
        self._ingest_all(self._files_under_root_folder)
        
        # Get BM25 documents for saving
        bm25_docs = self.dual_ingest_service.get_bm25_documents()
        
        logger.info("=" * 60)
        logger.info("DUAL INGESTION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.total_documents}")
        logger.info(f"Dense chunks created: {self.total_documents * 'varies'}")
        logger.info(f"BM25 chunks created: {len(bm25_docs)}")
        logger.info("=" * 60)
        
        # Save BM25 documents for later use
        self._save_bm25_documents(bm25_docs)

    def _ingest_all(self, files_to_ingest: list[Path]) -> None:
        logger.info(f"Ingesting {len(files_to_ingest)} files into BOTH indices...")
        logger.info(f"Files: {[f.name for f in files_to_ingest]}")
        
        result = self.dual_ingest_service.bulk_ingest(
            [(str(p.name), p) for p in files_to_ingest]
        )
        
        logger.info(f"Dual ingestion complete:")
        logger.info(f"  - Dense index: {len(result['dense'])} documents")
        logger.info(f"  - BM25 index: {len(result['bm25'])} nodes")

    def _save_bm25_documents(self, bm25_docs: list) -> None:
        """Save BM25 documents to a pickle file for later retrieval."""
        import pickle
        output_file = Path("bm25_documents.pkl")
        
        with open(output_file, 'wb') as f:
            pickle.dump(bm25_docs, f)
        
        logger.info(f"Saved {len(bm25_docs)} BM25 documents to {output_file}")

    def ingest_on_watch(self, changed_path: Path) -> None:
        logger.info(f"Detected change at path={changed_path}, dual ingesting...")
        self._do_ingest_one(changed_path)

    def _do_ingest_one(self, changed_path: Path) -> None:
        try:
            if changed_path.exists():
                logger.info(f"Started dual ingestion of file={changed_path}")
                result = self.dual_ingest_service.ingest_file(
                    changed_path.name, 
                    changed_path
                )
                logger.info(f"Completed dual ingestion of file={changed_path}")
                logger.info(f"  - Dense: {len(result['dense'])} chunks")
                logger.info(f"  - BM25: {len(result['bm25'])} chunks")
                
                # Update saved BM25 documents
                bm25_docs = self.dual_ingest_service.get_bm25_documents()
                self._save_bm25_documents(bm25_docs)
        except Exception:
            logger.exception(
                f"Failed to ingest document: {changed_path}, find the exception attached"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ingest_folder_dual.py",
        description="Ingest documents into BOTH Dense (512 chunks) and BM25 (256 chunks) indices"
    )
    parser.add_argument("folder", help="Folder to ingest")
    parser.add_argument(
        "--watch",
        help="Watch for changes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--ignored",
        nargs="*",
        help="List of files/directories to ignore",
        default=[],
    )
    parser.add_argument(
        "--log-file",
        help="Optional path to a log file. If provided, logs will be written to this file.",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode="a")
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    root_path = Path(args.folder)
    if not root_path.exists():
        raise ValueError(f"Path {args.folder} does not exist")

    # Get the dual ingest service
    dual_ingest_service = global_injector.get(DualIngestService)
    settings = global_injector.get(Settings)
    
    worker = DualIngestWorker(dual_ingest_service, settings)
    worker.ingest_folder(root_path, args.ignored)

    if args.ignored:
        logger.info(f"Skipping following files and directories: {args.ignored}")

    if args.watch:
        logger.info(f"Watching {args.folder} for changes, press Ctrl+C to stop...")
        from private_gpt.server.ingest.ingest_watcher import IngestWatcher
        
        watcher = IngestWatcher(root_path, worker.ingest_on_watch)
        watcher.start()