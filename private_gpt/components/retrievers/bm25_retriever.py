# from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode, TextNode
from pathlib import Path
import json
import logging
import re
import unicodedata
import bm25s
import Stemmer
from datetime import datetime


logger = logging.getLogger(__name__)

RESULTS_FILE = Path("retrieval_results_sparse.json")


class BM25Retriever(BaseRetriever):
    """
    Wrapper around BM25 that:
      - uses bm25s for official BM25 scoring
      - logs top results
      - saves each retrieval to a JSON file
    """
    def __init__(self, documents, top_k: int = 5):
        """
        BM25 retriever using bm25s, compatible with TextNode objects.
        Indexes `original_text` but returns `window` + metadata.
        """
        self.top_k = top_k
        self.retriever_name = "BM25 (bm25s)"
        self.stemmer = Stemmer.Stemmer("english")

        corpus = []
        self.doc_info = []  # to map back results to metadata

        for doc in documents:
            # Get the full text for indexing
            full_text = None
            if hasattr(doc, "metadata"):
                full_text = doc.metadata.get("original_text", None)
            if not full_text and hasattr(doc, "text"):
                full_text = doc.text

            if not full_text:
                logger.warning(f"No valid text found for node {doc.id_}, skipping.")
                continue

            corpus.append(full_text)

            # Save node info for retrieval
            self.doc_info.append({
                "id": doc.id_,
                "node": doc,  # ✅ Keep original node object
                "window": doc.metadata.get("window", ""),
                "original_text": full_text,
                "metadata": doc.metadata,
            })
        
        self.protected_phrases = ["sound level meter", "hand held device", "ISO 3382", "ISO 18233", "ISO 16283", "ISO 9612", "ISO 10140", "HBK 2755", "B&K 2245", "sound source", "Type 4292-L", "HBK 2755"]
        corpus = [self.preprocess_text(text) for text in corpus]

        # Tokenize and index
        logger.info("Tokenizing corpus...")
        corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=self.stemmer)

        logger.info("Building BM25 index...")
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

        logger.info(f"Indexed {len(corpus)} documents.")
        self.corpus = corpus
        
        # Initialize results file if it doesn't exist
        if not RESULTS_FILE.exists():
            with open(RESULTS_FILE, 'w') as f:
                json.dump([], f)
    
    def preprocess_text(self, text):
        """Replace multi-word phrases with single tokens."""
        for phrase in self.protected_phrases:
            token = phrase.replace(" ", "_")
            text = re.sub(
                r'\b' + re.escape(phrase) + r'\b', 
                token, 
                text, 
                flags=re.IGNORECASE
            )
        return text

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve using bm25s retriever, log and save JSON output."""
        # Original query string
        query = query_bundle.query_str

        # Preprocess query the same way as the corpus
        query_processed = self.preprocess_text(query)

        # Tokenize the preprocessed query
        query_tokens = bm25s.tokenize([query_processed], stemmer=self.stemmer)
        
        # request top-k results (use configured top_k)
        results, scores = self.retriever.retrieve(query_tokens, k=self.top_k)

        # Build lists for both LlamaIndex (NodeWithScore) and JSON output (dicts)
        nodes_with_scores = []  # For LlamaIndex
        json_results = []       # For saving to file

        for i in range(results.shape[1]):
            doc_or_id, score = results[0, i], scores[0, i]
            score = float(score)

            # default values
            doc_id = None
            window_text = None
            original_text = None
            metadata = {}
            node = None

            # try to treat returned value as an index into our corpus/doc_info
            try:
                idx = int(doc_or_id)
                if 0 <= idx < len(self.doc_info):
                    info = self.doc_info[idx]
                    doc_id = info.get("id")
                    node = info.get("node")
                    window_text = info.get("window", "")
                    original_text = info.get("original_text", "")
                    metadata = info.get("metadata", {}) or {}
                else:
                    # index out of range — fall back to string representation
                    window_text = str(doc_or_id)
            except Exception:
                # not an index; assume the retriever returned the text directly
                window_text = str(doc_or_id)

            # print for debugging/visibility
            print(f"Rank {i+1} (score: {score:.2f}): {window_text[:200]}...")

            # Create NodeWithScore for LlamaIndex
            if node:
                node_with_score = NodeWithScore(node=node, score=score)
                nodes_with_scores.append(node_with_score)

            # Create dict for JSON output
            json_results.append({
                "rank": i + 1,
                "id": doc_id,
                "window": window_text,
                "original_text": original_text,
                "metadata": metadata,
                "score": score,
            })

        # Save to JSON file
        self._save_results_to_json(query, json_results)
        
        # Return NodeWithScore objects for LlamaIndex
        return nodes_with_scores

    def _save_results_to_json(self, query: str, results: list[dict]):
        """Save retrieval results to JSON file."""
        
        # Create result entry
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retriever": self.retriever_name,
            "top_k": self.top_k,
            "results": results
        }
        
        # Load existing results
        try:
            with open(RESULTS_FILE, 'r') as f:
                all_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_results = []
        
        # Append new result
        all_results.append(result_entry)
        
        # Save back to file
        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Saved results to {RESULTS_FILE}")