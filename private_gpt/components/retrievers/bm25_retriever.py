from llama_index.retrievers.bm25 import BM25Retriever as LlamaBM25Retriever
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, BaseNode
from pathlib import Path
import json
import logging
import re
import unicodedata


logger = logging.getLogger(__name__)

RESULTS_FILE = Path("retrieval_results_sparse.json")


class BM25Retriever(BaseRetriever):
    """
    Wrapper around LlamaIndex's BM25Retriever that:
      - uses official BM25 scoring
      - logs top results
      - saves each retrieval to a JSON file
    """
    def __init__(self, documents, top_k: int = 5):
        self.top_k = top_k
        self.retriever_name = "BM25"
        self.results_file = RESULTS_FILE

        # ✅ Convert documents to nodes if needed
        nodes = []
        for doc in documents:
            if isinstance(doc, BaseNode):
                nodes.append(doc)
            else:
                # If it's not already a node, it might be a Document
                # Documents have a get_content() or text attribute
                logger.warning(f"Document is not a BaseNode, type: {type(doc)}")
                nodes.append(doc)
        
        logger.info(f"Preparing to initialize BM25Retriever with {len(nodes)} nodes")

        # ✅ Use from_defaults as shown in the available methods
        self.retriever = LlamaBM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k
        )
        logger.info(f"Successfully initialized BM25Retriever with {len(nodes)} nodes")

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """Retrieve with extensive debugging"""
        query = query_bundle.query_str
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔍 QUERY: '{query}'")
        logger.info(f"{'='*80}")
        logger.info(f"Query length: {len(query)} chars")
        logger.info(f"Query words: {query.split()}")
        
        # Show likely tokenization
        token_pattern = r'(?u)\b\w\w+\b'
        tokens = re.findall(token_pattern, query.lower())
        logger.info(f"Likely tokens after tokenization: {tokens}")

        # Run retrieval
        logger.info(f"\n⏳ Running BM25 retrieval...")
        results = self.retriever.retrieve(query)
        
        logger.info(f"\n📊 Retrieved {len(results)} results")
        
        if not results:
            logger.warning(f"❌ NO RESULTS FOUND!")
            logger.info(f"\n🔍 Investigating why no results...")
            
            # Manual search in first 50 documents
            query_terms_lower = [t.lower() for t in query.split()]
            logger.info(f"Searching for terms: {query_terms_lower}")
            
            found_count = 0
            for i, node in enumerate(self.nodes[:50]):
                text = getattr(node, "text", "").lower()
                matches = [term for term in query_terms_lower if term in text]
                if matches:
                    found_count += 1
                    logger.info(f"\n  ✓ Node {i} contains: {matches}")
                    logger.info(f"    Preview: {text[:150]}...")
            
            logger.info(f"\nFound {found_count} documents (out of first 50) containing query terms")
            return results

        # Log all results in detail
        query_terms_lower = [t.lower() for t in query.split()]
        
        for rank, res in enumerate(results, start=1):
            text = getattr(res.node, "text", "")
            metadata = getattr(res.node, "metadata", {})
            score = float(res.score) if res.score is not None else 0.0
            
            logger.info(f"\n{'─'*80}")
            logger.info(f"📄 RANK {rank}")
            logger.info(f"{'─'*80}")
            logger.info(f"Score: {score:.6f}")
            logger.info(f"Text length: {len(text)} chars")
            
            # Check which query terms appear
            text_lower = text.lower()
            matches = {}
            for term in query_terms_lower:
                count = text_lower.count(term)
                matches[term] = count
                if count > 0:
                    logger.info(f"  ✓ '{term}' appears {count} times")
                else:
                    logger.info(f"  ✗ '{term}' not found")
            
            # Show text preview
            logger.info(f"\nText preview (first 300 chars):")
            logger.info(f"{text[:300]}...")
            
            if metadata:
                logger.info(f"\nMetadata: {metadata}")
            
            # Highlight query terms in text
            snippet = text[:500]
            for term in query_terms_lower:
                if term in snippet.lower():
                    # Find first occurrence
                    idx = snippet.lower().find(term)
                    context_start = max(0, idx - 50)
                    context_end = min(len(snippet), idx + len(term) + 50)
                    context = snippet[context_start:context_end]
                    logger.info(f"\nContext around '{term}': ...{context}...")
                    break

        # Warn if scores seem low
        if results and results[0].score < 1.0:
            logger.warning(f"\n⚠️  Top score is only {results[0].score:.6f} - this seems low!")
            logger.info(f"This might indicate:")
            logger.info(f"  - Query terms are very common (high IDF penalty)")
            logger.info(f"  - Documents are very long (length normalization)")
            logger.info(f"  - Stemming is affecting matches")

        logger.info(f"\n{'='*80}\n")
        return results
    
    def normalize_text(text: str) -> str:
        if not text:
            return ""
        # Normalize Unicode and remove non-breaking spaces
        text = unicodedata.normalize("NFKC", text)
        text = text.replace("\xa0", " ")
        # Replace hyphens/underscores with spaces
        text = text.replace("-", " ").replace("_", " ")
        # Collapse multiple spaces and lowercase
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()