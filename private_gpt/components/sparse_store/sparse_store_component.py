import traceback
from bm25s import BM25
from injector import singleton
import numpy as np
from llama_index.core.schema import NodeWithScore
from private_gpt.components.retrievers.bm25_retriever import BM25Retriever
import logging

logger = logging.getLogger(__name__)

@singleton
class SparseStoreComponent:
    def __init__(self, storage_context=None):
        """
        Initialize sparse store component.
        
        Args:
            storage_context: Optional storage context for auto-backfill
        """
        self.index = None
        self.nodes = []
        self.doc_ids = []
        self.tokenized_corpus = []
        self._is_indexed = False
        self._storage_context = storage_context
        self._backfill_attempted = False

        # Log when instance is created and WHERE it was created
        instance_id = id(self)
        logger.info(f"NEW SparseStoreComponent instance created - ID: {instance_id}")
        logger.info(f"Creation stack trace:\n{''.join(traceback.format_stack())}")

    def _auto_backfill_from_storage(self):
        """
        Automatically backfill BM25 from existing documents in storage.
        This is called automatically on first retriever request.
        """
        if self._backfill_attempted:
            return  # Only try once
        
        self._backfill_attempted = True
        
        if self._storage_context is None:
            logger.warning("No storage context provided, cannot auto-backfill BM25")
            return
        
        try:
            logger.info("Attempting to auto-backfill BM25 from existing documents...")
            
            # Get docstore
            docstore = self._storage_context.docstore
            
            # Get all document IDs
            all_doc_ids = list(docstore.docs.keys())
            
            if not all_doc_ids:
                logger.info("No documents found in storage, BM25 will be populated on first ingest")
                return
            
            logger.info(f"Found {len(all_doc_ids)} documents in storage, backfilling BM25...")
            
            # Extract texts and nodes
            texts = []
            doc_ids = []
            nodes = []
            
            for doc_id in all_doc_ids:
                try:
                    node = docstore.get_node(doc_id)
                    
                    if node:
                        # Extract text
                        if hasattr(node, 'text'):
                            text = node.text
                        elif hasattr(node, 'get_content'):
                            text = node.get_content()
                        else:
                            text = str(node)
                        
                        if text and text.strip():
                            texts.append(text)
                            doc_ids.append(doc_id)
                            nodes.append(node)
                            
                except Exception as e:
                    logger.debug(f"Could not process node {doc_id}: {e}")
                    continue
            
            if texts:
                # Ingest into BM25
                self.ingest(texts, doc_ids)
                self.nodes = nodes
                logger.info(f"✓ Auto-backfilled {len(texts)} documents into BM25")
                logger.info(f"✓ BM25 vocabulary size: {len(self.index.vocab_dict)}")
            else:
                logger.warning("No valid texts extracted during backfill")
                
        except Exception as e:
            logger.error(f"Auto-backfill failed: {e}", exc_info=True)

    def ingest(self, documents: list[str], doc_ids: list[str]):
        """
        Ingest documents into the BM25 index.
        
        Args:
            documents: List of document texts or Node objects
            doc_ids: List of document IDs
        """
        if not documents:
            logger.warning("No documents provided for ingestion")
            return
            
        logger.info(f"Ingesting {len(documents)} documents into BM25 index")
        
        # Create a new BM25 instance
        self.index = BM25()
        
        # Tokenize documents
        self.tokenized_corpus = [doc.lower().split() for doc in documents]
        logger.info(f"Tokenized {len(self.tokenized_corpus)} documents")
        
        # Build BM25 index - CRITICAL: Must call both fit() and index()
        # logger.info("Fitting BM25 index...")
        # self.index.fit(self.tokenized_corpus)
        
        logger.info("Indexing BM25 corpus...")
        self.index.index(self.tokenized_corpus)  # This creates vocab_dict
        
        # Verify vocab_dict was created
        if not hasattr(self.index, 'vocab_dict'):
            raise RuntimeError(
                "BM25 index failed to create vocab_dict after calling .index()"
            )
        
        logger.info(f"BM25 vocab size: {len(self.index.vocab_dict)}")
        
        # Store the nodes and doc IDs
        self.nodes = documents
        self.doc_ids = doc_ids
        self._is_indexed = True
        
        logger.info("BM25 index successfully created and validated")

    def get_retriever(self, context_filter=None, top_k: int = 5) -> BM25Retriever:
        """
        Returns a BM25Retriever compatible with ContextChatEngine.
        Automatically backfills from storage if not yet indexed.

        Args:
            context_filter: Optional ContextFilter object containing doc_ids to limit the search.
            top_k: Number of top results to return.

        Returns:
            BM25Retriever instance configured with filtered nodes.
        """
        # Auto-backfill if not yet indexed
        if not self._is_indexed:
            logger.info("BM25 not indexed yet, attempting auto-backfill...")
            self._auto_backfill_from_storage()
        
        # If still not indexed after backfill, raise error
        if not self._is_indexed:
            raise RuntimeError(
                "Cannot create retriever: BM25 index has not been ingested yet. "
                "No documents found in storage for auto-backfill. "
                "Please ingest documents first."
            )
        
        if self.index is None:
            raise RuntimeError("BM25 index is None. Call ingest() first.")
        
        # Verify the index is properly initialized
        if not hasattr(self.index, 'vocab_dict'):
            raise RuntimeError(
                "BM25 index is missing vocab_dict. The index may not have been "
                "properly initialized with .index() method."
            )
        
        # If a context filter is provided, filter nodes to only include those doc_ids
        if context_filter is not None:
            from private_gpt.server.chat.chat_service import ContextFilter
            filtered_nodes = [
                node for node in self.nodes
                if hasattr(node, 'ref_doc_id') and node.ref_doc_id in context_filter.docs_ids
            ]
            logger.info(f"Filtered to {len(filtered_nodes)} nodes based on context filter")
        else:
            filtered_nodes = self.nodes
            logger.info(f"Using all {len(filtered_nodes)} nodes (no context filter)")

        # Return the BM25Retriever with the filtered nodes
        return BM25Retriever(self.index, filtered_nodes, top_k=top_k)

    def retrieve(self, query: str, top_k=5):
        """
        Retrieve top-k documents for a query.
        
        Args:
            query: Query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (doc_id, score)
        """
        if not self._is_indexed:
            raise RuntimeError("BM25 index has not been ingested yet. Call ingest() first.")
            
        query_tokens = query.lower().split()
        scores = self.index.get_scores(query_tokens)
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], scores[i]) for i in top_idx]