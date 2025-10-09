from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.schema import QueryBundle
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    def __init__(self, bm25_index, nodes, top_k: int = 5):
        self.bm25 = bm25_index
        self.nodes = nodes
        self.top_k = top_k
        self.retriever_name = "BM25"  # Add identifier

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_text = query_bundle.query_str.lower()
        query_tokens = query_text.split()

        # Get all scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Log score statistics
        logger.info(f"[{self.retriever_name}] Query: '{query_bundle.query_str}'")
        logger.info(f"[{self.retriever_name}] Total documents: {len(scores)}")
        logger.info(f"[{self.retriever_name}] Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")
        
        # Get top-k indices
        top_k = min(self.top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        # Log top-k results
        logger.info(f"[{self.retriever_name}] Top {top_k} results:")
        for rank, i in enumerate(top_idx, 1):
            node = self.nodes[i]
            score = scores[i]
            
            # Get preview of node content
            if isinstance(node, str):
                preview = node[:100]
            else:
                preview = node.text[:100] if hasattr(node, 'text') else str(node)[:100]
            
            logger.info(f"[{self.retriever_name}]   Rank {rank}: Index={i}, Score={score:.4f}, Preview='{preview}...'")
        
        # Create results
        results = []
        for i in top_idx:
            node = self.nodes[i]
            
            # Convert string to TextNode if necessary
            if isinstance(node, str):
                node = TextNode(text=node)
            
            results.append(NodeWithScore(node=node, score=float(scores[i])))
        
        return results