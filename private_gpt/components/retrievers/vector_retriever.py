from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
import logging
import re

import json
from pathlib import Path


logger = logging.getLogger(__name__)

RESULTS_FILE = Path("retrieval_results.json")

class LoggingVectorIndexRetriever(VectorIndexRetriever):
    """Wrapper around VectorIndexRetriever that adds logging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever_name = "Dense/Vector"
    
    def normalize_product_names(self, text: str) -> str:
        """Normalize all B&K product variations to standard format"""
        # HBK 2255, BK 2255, B&K 2255, B & K 2255 → HBK2255
        text = re.sub(
            r'\b(B\s*&\s*K|BK|HBK)\s+(\d{{4}})\b',
            r'HBK \2',
            text,
            flags=re.IGNORECASE
        )
        return text

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        # --- Normalize query ---
        normalized_query = self.normalize_product_names(query_bundle.query_str)
        # Replace the query inside the QueryBundle (if allowed)  
        # or create a new modified QueryBundle
        # Create a new QueryBundle with normalized query
        normalized_bundle = QueryBundle(
            query_str=normalized_query,
            embedding=query_bundle.embedding  # Only pass embedding if it exists
        )

        logger.info(f"[{self.retriever_name}] Original Query: '{query_bundle.query_str}'")
        logger.info(f"[{self.retriever_name}] Normalized Query: '{normalized_query}'")
        logger.info(f"[{self.retriever_name}] Similarity top_k: {self.similarity_top_k}")


        logger.info(f"[{self.retriever_name}] Query: '{query_bundle.query_str}'")
        logger.info(f"[{self.retriever_name}] Similarity top_k: {self.similarity_top_k}")
        
        # Call the parent class retrieve method
        results = super()._retrieve(normalized_bundle)


        # Prepare results for JSON
        results_list = []
        for rank, result in enumerate(results, 1):
            score = result.score if hasattr(result, 'score') else None
            text = result.node.text if hasattr(result.node, 'text') else str(result.node)
            metadata = result.node.metadata if hasattr(result.node, 'metadata') else {}
            
            results_list.append({
                "rank": rank,
                "score": score,
                "text": text,
                "metadata": metadata
            })
        
        # Load existing data if file exists
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = {}
        
        # Add or update this query
        all_data[query_bundle.query_str] = results_list
        
        # Save back to file
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"[{self.retriever_name}] Updated results saved to {RESULTS_FILE}")
        
        

        
        # Log results
        # logger.info(f"[{self.retriever_name}] Retrieved {len(results)} nodes")
        # logger.info(f"[{self.retriever_name}] Top {len(results)} results:")
        
        # for rank, result in enumerate(results, 1):
        #     score = result.score if hasattr(result, 'score') else 'N/A'
        #     preview = result.node.text[:100] if hasattr(result.node, 'text') else str(result.node)[:100]
        #     logger.info(f"[{self.retriever_name}]   Rank {rank}: Score={score:.4f}, Preview='{preview}...'")
            
        #     if hasattr(result.node, 'metadata'):
        #         logger.info(f"[{self.retriever_name}]     Metadata: {result.node.metadata}")
        
        return results