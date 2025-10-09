from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
import logging

logger = logging.getLogger(__name__)

class LoggingVectorIndexRetriever(VectorIndexRetriever):
    """Wrapper around VectorIndexRetriever that adds logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever_name = "Dense/Vector"
    
    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        logger.info(f"[{self.retriever_name}] Query: '{query_bundle.query_str}'")
        logger.info(f"[{self.retriever_name}] Similarity top_k: {self.similarity_top_k}")
        
        # Call the parent class retrieve method
        results = super()._retrieve(query_bundle)
        
        # Log results
        logger.info(f"[{self.retriever_name}] Retrieved {len(results)} nodes")
        logger.info(f"[{self.retriever_name}] Top {len(results)} results:")
        
        for rank, result in enumerate(results, 1):
            score = result.score if hasattr(result, 'score') else 'N/A'
            preview = result.node.text[:100] if hasattr(result.node, 'text') else str(result.node)[:100]
            logger.info(f"[{self.retriever_name}]   Rank {rank}: Score={score:.4f}, Preview='{preview}...'")
            
            if hasattr(result.node, 'metadata'):
                logger.info(f"[{self.retriever_name}]     Metadata: {result.node.metadata}")
        
        return results