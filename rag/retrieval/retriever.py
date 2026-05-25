from typing import List, Dict, Any
import logging
from langfuse import observe
from langchain_core.documents import Document
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

from app.config import settings
from app.utils.langfuse_client import langfuse_client
from rag.embeddings.store import VectorStoreManager

logger = logging.getLogger("retriever")

class FranchiseRetriever:
    """
    Hybrid Retriever utilizing BM25 + FAISS Dense Retrieval with Metadata Boosting.
    Fixed to cache BM25 and avoid dropping chunks.
    """

    def __init__(self):
        self.manager = VectorStoreManager()
        self.bm25_retriever = None

    # =====================================================
    # MAIN RETRIEVE
    # =====================================================

    @observe(name="retrieval")
    def retrieve(self, query: str, k: int = None, search_query: str = None, initial_k: int = None) -> List[Dict[str, Any]]:
        top_k = k or settings.TOP_K_RESULTS
        faiss_query = search_query if search_query else query

        try:
            # -------------------------------------------------
            # Vector Store
            # -------------------------------------------------
            if self.manager.vector_store is None:
                self.manager.load_or_create_index()

            vector_store = self.manager.vector_store

            if vector_store is None:
                return []

            # -------------------------------------------------
            # DENSE RETRIEVER
            # -------------------------------------------------
            dense_retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": top_k * 3,
                    "fetch_k": top_k * 6,
                    "lambda_mult": 0.5
                }
            )

            # -------------------------------------------------
            # BM25 RETRIEVER CACHING
            # -------------------------------------------------
            if self.bm25_retriever is None:
                all_docs = vector_store.docstore._dict.values()
                self.bm25_retriever = BM25Retriever.from_documents(list(all_docs))
            
            self.bm25_retriever.k = top_k * 3

            # -------------------------------------------------
            # HYBRID RETRIEVER
            # -------------------------------------------------
            hybrid_retriever = EnsembleRetriever(
                retrievers=[self.bm25_retriever, dense_retriever],
                weights=[0.4, 0.6]
            )

            # -------------------------------------------------
            # RETRIEVE
            # -------------------------------------------------
            docs = hybrid_retriever.invoke(faiss_query)

            # -------------------------------------------------
            # POST FILTERING
            # -------------------------------------------------
            docs = self.post_process_results(docs, top_k)

            # -------------------------------------------------
            # FORMAT RESULTS
            # -------------------------------------------------
            results = []
            for rank, doc in enumerate(docs):
                results.append({
                    "rank": rank + 1,
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })

            # -------------------------------------------------
            # LANGFUSE LOGGING
            # -------------------------------------------------
            langfuse_client.update_current_span(
                input={"query": faiss_query, "top_k": top_k},
                output={"retrieved_chunks": results}
            )

            logger.info(f"Retrieved {len(results)} chunks for query: {faiss_query}")
            return results

        except Exception as e:
            logger.exception(f"Retriever failed: {e}")
            return []

    # =====================================================
    # POST PROCESSING
    # =====================================================

    def post_process_results(self, docs: List[Document], top_k: int) -> List[Document]:
        seen_texts = set()
        unique_docs = []

        for doc in docs:
            chunk_text = doc.page_content.strip()
            if not chunk_text or chunk_text in seen_texts:
                continue

            seen_texts.add(chunk_text)
            unique_docs.append(doc)

        # Prefer FAQ + table chunks for factual queries
        unique_docs = sorted(unique_docs, key=self.relevance_boost, reverse=True)
        return unique_docs[:top_k]

    # =====================================================
    # RELEVANCE BOOSTING
    # =====================================================

    def relevance_boost(self, doc: Document) -> float:
        score = 0
        block_type = doc.metadata.get("block_type", "")

        if block_type == "table":
            score += 2
        if "faq" in block_type.lower():
            score += 1.5
            
        section = doc.metadata.get("section", "").lower() or doc.metadata.get("section_path", "").lower()
        if "pricing" in section or "investment" in section or "item 7" in section:
            score += 1

        return score
