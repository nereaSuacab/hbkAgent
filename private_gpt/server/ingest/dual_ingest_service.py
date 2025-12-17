import logging
import time
import tempfile
from pathlib import Path
from typing import AnyStr, BinaryIO
import re

from injector import inject, singleton
from llama_index.core.node_parser import SentenceWindowNodeParser, SentenceSplitter
from llama_index.core.storage import StorageContext
from llama_index.core.indices import VectorStoreIndex

from private_gpt.components.embedding.embedding_component import EmbeddingComponent
from private_gpt.components.ingest.ingest_component import get_ingestion_component
from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.components.node_store.node_store_component import NodeStoreComponent
from private_gpt.components.vector_store.vector_store_component import (
    VectorStoreComponent,
)
from private_gpt.server.ingest.model import IngestedDoc
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)


@singleton
class DualIngestService:
    """
    Service that maintains TWO separate indices:
    1. Dense/Vector index with larger chunks (512 tokens)
    2. BM25 index with smaller chunks (256 tokens)
    
    This allows each retrieval method to work with optimal chunk sizes.
    """
    
    @inject
    def __init__(
        self,
        llm_component: LLMComponent,
        vector_store_component: VectorStoreComponent,
        embedding_component: EmbeddingComponent,
        node_store_component: NodeStoreComponent,
    ) -> None:
        logger.info("=== DualIngestService initialization started ===")
        overall_start = time.time()

        self.llm_service = llm_component
        self.embedding_component = embedding_component

        # ==========================================
        # INDEX 1: Dense/Vector (Larger chunks)
        # ==========================================
        logger.info("Creating Dense/Vector storage context...")
        self.dense_storage_context = StorageContext.from_defaults(
            vector_store=vector_store_component.vector_store,
            docstore=node_store_component.doc_store,
            index_store=node_store_component.index_store,
        )
        
        # Larger chunks for semantic search
        self.dense_node_parser = SentenceSplitter(
            chunk_size=256, 
            chunk_overlap=50,  
            paragraph_separator="\n\n\n",
            separator="\n\n",
        )
        
        self.dense_ingest_component = get_ingestion_component(
            self.dense_storage_context,
            embed_model=embedding_component.embedding_model,
            transformations=[self.dense_node_parser, embedding_component.embedding_model],
            settings=settings(),
        )
        
        # ==========================================
        # INDEX 2: BM25 (Smaller chunks)
        # ==========================================
        logger.info("Creating BM25 storage context...")
        # Note: For BM25, we don't need vector embeddings, but we still need storage
        # You might want to create a separate docstore for this
        
        # Smaller chunks for keyword matching
        self.bm25_node_parser = SentenceWindowNodeParser.from_defaults()


        # BM25 documents will be stored separately (in memory or separate docstore)
        self.bm25_documents = []  # Will hold TextNode objects for BM25
        
        logger.info(f"=== DualIngestService initialized in {time.time() - overall_start:.2f}s ===")

    def normalize_product_names(self, text: str) -> str:
        """Normalize all B&K product variations to standard format"""
        text = re.sub(
            r'\b(B\s*&\s*K|BK|HBK)\s+(\d{4})\b',
            r'HBK \2',
            text,
            flags=re.IGNORECASE
        )
        return text

    def _ingest_data(self, file_name: str, file_data: AnyStr) -> dict:
        """
        Ingests data into BOTH indices with different chunking.
        
        Returns:
            dict with keys 'dense' and 'bm25' containing respective IngestedDocs
        """
        logger.debug("Got file data of size=%s to ingest", len(file_data))
        
        # Normalize product names
        if isinstance(file_data, bytes):
            text = file_data.decode('utf-8')
            text = self.normalize_product_names(text)
            file_data = text.encode('utf-8')
        else:
            file_data = self.normalize_product_names(str(file_data))
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            try:
                path_to_tmp = Path(tmp.name)
                if isinstance(file_data, bytes):
                    path_to_tmp.write_bytes(file_data)
                else:
                    path_to_tmp.write_text(str(file_data))
                return self.ingest_file(file_name, path_to_tmp)
            finally:
                tmp.close()
                path_to_tmp.unlink()

    def ingest_file(self, file_name: str, file_data: Path) -> dict:
        """
        Ingest a file into BOTH dense and BM25 indices.
        
        Returns:
            dict: {'dense': [IngestedDoc], 'bm25': [TextNode]}
        """
        logger.info(f"Dual ingestion starting for file_name={file_name}")
        
        # ==========================================
        # 1. Ingest into Dense/Vector Index
        # ==========================================
        logger.info(f"Ingesting {file_name} into Dense index...")
        dense_docs = self.dense_ingest_component.ingest(file_name, file_data)
        dense_ingested = [IngestedDoc.from_document(doc) for doc in dense_docs]
        
        # ==========================================
        # 2. Process for BM25 Index
        # ==========================================
        logger.info(f"Processing {file_name} for BM25 index...")
        
        # Load and parse the document with BM25 chunking
        from llama_index.core import SimpleDirectoryReader
        reader = SimpleDirectoryReader(input_files=[file_data])
        documents = reader.load_data()
        
        # Parse with smaller chunks
        bm25_nodes = self.bm25_node_parser.get_nodes_from_documents(documents)
        
        # Store original text in metadata for BM25 indexing
        for node in bm25_nodes:
            if not node.metadata:
                node.metadata = {}
            node.metadata['original_text'] = node.text
            node.metadata['window'] = node.text
            node.metadata['file_name'] = file_name
        
        # Add to BM25 document collection
        self.bm25_documents.extend(bm25_nodes)
        
        logger.info(f"Finished dual ingestion for file_name={file_name}")
        logger.info(f"  - Dense: {len(dense_ingested)} chunks")
        logger.info(f"  - BM25: {len(bm25_nodes)} chunks")
        
        return {
            'dense': dense_ingested,
            'bm25': bm25_nodes
        }

    def ingest_text(self, file_name: str, text: str) -> dict:
        logger.debug("Ingesting text data with file_name=%s", file_name)
        return self._ingest_data(file_name, text)

    def ingest_bin_data(self, file_name: str, raw_file_data: BinaryIO) -> dict:
        logger.debug("Ingesting binary data with file_name=%s", file_name)
        file_data = raw_file_data.read()
        return self._ingest_data(file_name, file_data)

    def bulk_ingest(self, files: list[tuple[str, Path]]) -> dict:
        """
        Bulk ingest multiple files into both indices.
        
        Returns:
            dict: {'dense': [IngestedDoc], 'bm25': [TextNode]}
        """
        logger.info(f"Bulk dual ingestion starting for {len(files)} files")
        
        all_dense_docs = []
        all_bm25_nodes = []
        
        for file_name, file_path in files:
            result = self.ingest_file(file_name, file_path)
            all_dense_docs.extend(result['dense'])
            all_bm25_nodes.extend(result['bm25'])
        
        logger.info(f"Finished bulk dual ingestion")
        logger.info(f"  - Total Dense chunks: {len(all_dense_docs)}")
        logger.info(f"  - Total BM25 chunks: {len(all_bm25_nodes)}")
        
        return {
            'dense': all_dense_docs,
            'bm25': all_bm25_nodes
        }

    def get_bm25_documents(self) -> list:
        """Get all documents for BM25 indexing"""
        return self.bm25_documents

    def get_dense_storage_context(self) -> StorageContext:
        """Get the storage context for dense retrieval"""
        return self.dense_storage_context

    def list_ingested(self) -> list[IngestedDoc]:
        """List ingested documents (from dense index)"""
        ingested_docs: list[IngestedDoc] = []
        try:
            docstore = self.dense_storage_context.docstore
            ref_docs = docstore.get_all_ref_doc_info()

            if not ref_docs:
                return ingested_docs

            for doc_id, ref_doc_info in ref_docs.items():
                doc_metadata = None
                if ref_doc_info is not None and ref_doc_info.metadata is not None:
                    doc_metadata = IngestedDoc.curate_metadata(ref_doc_info.metadata)
                ingested_docs.append(
                    IngestedDoc(
                        object="ingest.document",
                        doc_id=doc_id,
                        doc_metadata=doc_metadata,
                    )
                )
        except ValueError:
            logger.warning("Got an exception when getting list of docs", exc_info=True)
        
        logger.debug("Found count=%s ingested documents", len(ingested_docs))
        return ingested_docs

    def delete(self, doc_id: str) -> None:
        """Delete an ingested document from both indices"""
        logger.info(f"Deleting document={doc_id} from both indices")
        
        # Delete from dense index
        self.dense_ingest_component.delete(doc_id)
        
        # Delete from BM25 documents
        self.bm25_documents = [
            node for node in self.bm25_documents 
            if node.metadata.get('doc_id') != doc_id
        ]