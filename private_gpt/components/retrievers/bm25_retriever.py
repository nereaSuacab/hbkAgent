from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import QueryBundle
import numpy as np

class BM25Retriever(BaseRetriever):
    def __init__(self, bm25_index, nodes, top_k: int = 5):
        self.bm25 = bm25_index
        self.nodes = nodes
        self.top_k = top_k

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        query_text = query_bundle.query_str.lower()
        query_tokens = query_text.split()

        scores = self.bm25.get_scores(query_tokens)
        top_k = min(self.top_k, len(scores))
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = [
            NodeWithScore(node=self.nodes[i], score=float(scores[i]))
            for i in top_idx
        ]
        return results


