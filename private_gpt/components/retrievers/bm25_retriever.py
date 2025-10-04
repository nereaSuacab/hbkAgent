from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, ingested_docs):
        """
        ingested_docs: lista de objetos ingested_document
        Cada doc debe tener .doc_id y .text
        """
        self.docs = ingested_docs
        self.corpus = [doc.text.split() for doc in ingested_docs]
        self.bm25 = BM25Okapi(self.corpus)

    def get_top_docs(self, query: str, top_k: int = 5):
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.docs[i].doc_id for i in top_indices]