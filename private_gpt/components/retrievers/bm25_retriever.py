from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.schema import QueryBundle
import numpy as np
import logging

import json
from pathlib import Path

logger = logging.getLogger(__name__)

RESULTS_FILE = Path("retrieval_results_sparse.json")


class BM25Retriever(BaseRetriever):
    def __init__(self, bm25_index, nodes, top_k: int = 5):
        self.bm25 = bm25_index
        self.nodes = nodes
        self.top_k = top_k
        self.retriever_name = "BM25"  # Add identifier

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_text = query_bundle.query_str.lower()
        query_tokens = query_text.split()

        # Compute scores
        scores = self.bm25.get_scores(query_tokens)

        # Logging basic info
        logger.info(f"[{self.retriever_name}] Query: '{query_bundle.query_str}'")
        logger.info(f"[{self.retriever_name}] Total documents: {len(scores)}")
        logger.info(f"[{self.retriever_name}] Score range: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        # Get top-k
        top_k = min(self.top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        results_list = []
        for rank, i in enumerate(top_idx, 1):
            node = self.nodes[i]
            score = float(scores[i])

            # Convert to TextNode if string
            if isinstance(node, str):
                node = TextNode(text=node)

            results.append(NodeWithScore(node=node, score=score))

            # Prepare JSON entry
            text = node.text if hasattr(node, 'text') else str(node)
            metadata = node.metadata if hasattr(node, 'metadata') else {}

            results_list.append({
                "rank": rank,
                "score": score,
                "text": text,
                "metadata": metadata
            })

            # Log top-k preview
            preview = text[:100]
            logger.info(f"[{self.retriever_name}]   Rank {rank}: Score={score:.4f}, Preview='{preview}...'")

        # Load existing JSON
        if RESULTS_FILE.exists():
            with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                all_data = json.load(f)
        else:
            all_data = {}

        # Add/update query
        all_data[query_bundle.query_str] = results_list

        # Save JSON
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

        logger.info(f"[{self.retriever_name}] Updated results saved to {RESULTS_FILE}")

        return results