import traceback
from bm25s import BM25
from injector import singleton
import numpy as np
from llama_index.core.schema import NodeWithScore
from private_gpt.components.retrievers.bm25_retriever import BM25Retriever
from llama_index.core.indices import VectorStoreIndex, load_index_from_storage
from injector import inject, singleton
from private_gpt.settings.settings import Settings
from llama_index.core.storage import StorageContext


import logging

logger = logging.getLogger(__name__)


@singleton
class SparseStoreComponent:
    @inject
    def __init__(self, settings: Settings):
        self.persist_dir = str(settings.data.local_data_folder)
        logger.info(f"Loading sparse store from: {self.persist_dir}")
        
        self.storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
        
        # No need to load the full index if you only need docs for BM25
        self.documents = list(self.storage_context.docstore.docs.values())
        logger.info(f"Loaded {len(self.documents)} documents from sparse store")
        
        self.retriever = BM25Retriever(self.documents, top_k=5)

    def get_retriever(self, context_filter=None, top_k=None):
        if top_k:
            self.retriever.top_k = top_k
            self.retriever.retriever.similarity_top_k = top_k
        return self.retriever
