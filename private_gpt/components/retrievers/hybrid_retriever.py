from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from typing import List
import logging

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever that combines:
    - Dense/Vector retriever (with 512 token chunks)
    - BM25 retriever (with 256 token chunks)
    
    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """
    
    def __init__(
        self,
        dense_retriever,
        bm25_retriever,
        dense_weight: float = 0.5,
        bm25_weight: float = 0.5,
        top_k: int = 5,
        fusion_method: str = "rrf"  # "rrf" or "weighted"
    ):
        """
        Args:
            dense_retriever: Vector/semantic retriever (512 token chunks)
            bm25_retriever: Keyword-based retriever (256 token chunks)
            dense_weight: Weight for dense retriever (0-1)
            bm25_weight: Weight for BM25 retriever (0-1)
            top_k: Number of final results to return
            fusion_method: "rrf" (Reciprocal Rank Fusion) or "weighted" (weighted score)
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        self.fusion_method = fusion_method
        
        logger.info(f"HybridRetriever initialized:")
        logger.info(f"  - Dense weight: {dense_weight}")
        logger.info(f"  - BM25 weight: {bm25_weight}")
        logger.info(f"  - Fusion method: {fusion_method}")
        logger.info(f"  - Top K: {top_k}")

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """
        Retrieve and fuse results from both retrievers.
        """
        logger.info(f"[HybridRetriever] Query: '{query_bundle.query_str}'")
        
        # ==========================================
        # 1. Retrieve from Dense (512 token chunks)
        # ==========================================
        logger.info("[HybridRetriever] Retrieving from Dense index...")
        dense_results = self.dense_retriever._retrieve(query_bundle)
        logger.info(f"[HybridRetriever] Dense retrieved {len(dense_results)} results")
        
        # ==========================================
        # 2. Retrieve from BM25 (256 token chunks)
        # ==========================================
        logger.info("[HybridRetriever] Retrieving from BM25 index...")
        bm25_results = self.bm25_retriever._retrieve(query_bundle)
        logger.info(f"[HybridRetriever] BM25 retrieved {len(bm25_results)} results")
        
        # ==========================================
        # 3. Fuse results
        # ==========================================
        if self.fusion_method == "rrf":
            fused_results = self._reciprocal_rank_fusion(dense_results, bm25_results)
        else:
            fused_results = self._weighted_fusion(dense_results, bm25_results)
        
        # ==========================================
        # 4. Return top-k
        # ==========================================
        final_results = fused_results[:self.top_k]
        
        logger.info(f"[HybridRetriever] Final top-{self.top_k} results:")
        for i, result in enumerate(final_results, 1):
            source = result.node.metadata.get('retriever_source', 'unknown')
            chunk_size = result.node.metadata.get('chunk_size', 'unknown')
            score = result.score
            preview = result.node.text[:100] if hasattr(result.node, 'text') else ''
            logger.info(f"  Rank {i}: Score={score:.4f}, Source={source}, "
                       f"ChunkSize={chunk_size}, Preview='{preview}...'")
        
        return final_results

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[NodeWithScore], 
        bm25_results: List[NodeWithScore],
        k: int = 60
    ) -> List[NodeWithScore]:
        """
        Reciprocal Rank Fusion (RRF) algorithm.
        
        RRF Score = sum(1 / (k + rank)) for each retriever
        
        This method is scale-free and doesn't require score normalization.
        """
        logger.info("[HybridRetriever] Applying Reciprocal Rank Fusion...")
        
        # Create a dictionary to accumulate RRF scores
        node_scores = {}
        node_objects = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            node_id = result.node.node_id
            rrf_score = self.dense_weight / (k + rank)
            
            if node_id not in node_scores:
                node_scores[node_id] = 0
                node_objects[node_id] = result.node
                # Tag the source
                if not result.node.metadata:
                    result.node.metadata = {}
                result.node.metadata['retriever_source'] = 'dense'
                result.node.metadata['chunk_size'] = '512'
            
            node_scores[node_id] += rrf_score
        
        # Process BM25 results
        for rank, result in enumerate(bm25_results, 1):
            node_id = result.node.node_id
            rrf_score = self.bm25_weight / (k + rank)
            
            if node_id not in node_scores:
                node_scores[node_id] = 0
                node_objects[node_id] = result.node
                # Tag the source
                if not result.node.metadata:
                    result.node.metadata = {}
                result.node.metadata['retriever_source'] = 'bm25'
                result.node.metadata['chunk_size'] = '256'
            else:
                # Node appears in both retrievers
                if not node_objects[node_id].metadata:
                    node_objects[node_id].metadata = {}
                node_objects[node_id].metadata['retriever_source'] = 'both'
            
            node_scores[node_id] += rrf_score
        
        # Sort by score and create NodeWithScore objects
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = [
            NodeWithScore(node=node_objects[node_id], score=score)
            for node_id, score in sorted_nodes
        ]
        
        logger.info(f"[HybridRetriever] RRF produced {len(fused_results)} unique results")
        return fused_results

    def _weighted_fusion(
        self, 
        dense_results: List[NodeWithScore], 
        bm25_results: List[NodeWithScore]
    ) -> List[NodeWithScore]:
        """
        Weighted score fusion with normalization.
        
        Final Score = (dense_weight * norm_dense_score) + (bm25_weight * norm_bm25_score)
        """
        logger.info("[HybridRetriever] Applying Weighted Fusion...")
        
        # Normalize scores to [0, 1]
        def normalize_scores(results: List[NodeWithScore]) -> dict:
            if not results:
                return {}
            
            scores = [r.score for r in results]
            min_score = min(scores)
            max_score = max(scores)
            
            if max_score == min_score:
                return {r.node.node_id: 1.0 for r in results}
            
            return {
                r.node.node_id: (r.score - min_score) / (max_score - min_score)
                for r in results
            }
        
        dense_norm = normalize_scores(dense_results)
        bm25_norm = normalize_scores(bm25_results)
        
        # Combine scores
        node_scores = {}
        node_objects = {}
        
        # Add dense scores
        for result in dense_results:
            node_id = result.node.node_id
            node_scores[node_id] = self.dense_weight * dense_norm[node_id]
            node_objects[node_id] = result.node
            
            if not result.node.metadata:
                result.node.metadata = {}
            result.node.metadata['retriever_source'] = 'dense'
            result.node.metadata['chunk_size'] = '512'
        
        # Add BM25 scores
        for result in bm25_results:
            node_id = result.node.node_id
            bm25_contribution = self.bm25_weight * bm25_norm[node_id]
            
            if node_id in node_scores:
                node_scores[node_id] += bm25_contribution
                node_objects[node_id].metadata['retriever_source'] = 'both'
            else:
                node_scores[node_id] = bm25_contribution
                node_objects[node_id] = result.node
                
                if not result.node.metadata:
                    result.node.metadata = {}
                result.node.metadata['retriever_source'] = 'bm25'
                result.node.metadata['chunk_size'] = '256'
        
        # Sort and create results
        sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = [
            NodeWithScore(node=node_objects[node_id], score=score)
            for node_id, score in sorted_nodes
        ]
        
        logger.info(f"[HybridRetriever] Weighted fusion produced {len(fused_results)} unique results")
        return fused_results