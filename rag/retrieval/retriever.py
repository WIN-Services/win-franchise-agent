from typing import List, Dict, Any
import logging
from langfuse import observe
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
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
        # Pre-initialize the HyDE LLM once to avoid per-call constructor overhead
        self._hyde_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )

    def _generate_hyde_document(self, query: str) -> str:
        """
        Uses an LLM to generate a hypothetical document that answers the query.
        This provides a highly semantic dense target for FAISS.
        """
        try:
            messages = [
                SystemMessage(content="You are an expert on the WIN Home Inspection franchise. Please write a short, hypothetical document that answers the user's question with precise factual-sounding statements. Do not use conversational filler. This document will be used to search a vector database."),
                HumanMessage(content=query)
            ]
            result = self._hyde_llm.invoke(messages)
            return result.content
        except Exception as e:
            logger.exception(f"HyDE generation failed: {e}")
            return query

    # =====================================================
    # MAIN RETRIEVE
    # =====================================================

    @observe(name="retrieval")
    def retrieve(self, query: str, intent: str = "general", k: int = None) -> List[Dict[str, Any]]:
        top_k = k or settings.TOP_K_RESULTS
        
        # Select strategy
        strategy = "dense"
        search_query = query
        weights = [0.0, 1.0] # default to Dense
        
        if intent in ["investment", "fdd_financial"]:
            strategy = "hyde_dense"
            search_query = self._generate_hyde_document(query)
            weights = [0.0, 1.0] # Dense only, but with HyDE query
        elif intent == "process":
            strategy = "bm25_boosted_hybrid"
            weights = [0.7, 0.3] # 70% BM25, 30% Dense
        else:
            strategy = "dense"
            weights = [0.0, 1.0] # Dense only

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
                weights=weights
            )

            # -------------------------------------------------
            # RETRIEVE
            # -------------------------------------------------
            # Note: EnsembleRetriever takes a single query and passes it to all sub-retrievers.
            docs = hybrid_retriever.invoke(search_query)

            # -------------------------------------------------
            # POST FILTERING
            # -------------------------------------------------
            docs = self.post_process_results(docs, top_k, intent)

            # -------------------------------------------------
            # FORMAT RESULTS
            # -------------------------------------------------
            results = []
            for rank, doc in enumerate(docs):
                results.append({
                    "rank": rank + 1,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "source_url": doc.metadata.get("url", "")
                })

            # -------------------------------------------------
            # LANGFUSE LOGGING
            # -------------------------------------------------
            langfuse_client.update_current_span(
                input={"query": search_query, "intent": intent, "top_k": top_k},
                output={"retrieval_strategy": strategy, "retrieved_chunks": results}
            )

            logger.info(f"Retrieved {len(results)} chunks for query: {search_query} (Strategy: {strategy})")
            return results

        except Exception as e:
            logger.exception(f"Retriever failed: {e}")
            return []

    # =====================================================
    # POST PROCESSING
    # =====================================================

    def post_process_results(self, docs: List[Document], top_k: int, intent: str = "general") -> List[Document]:
        seen_texts = set()
        seen_sections = set()
        unique_docs = []

        for doc in docs:
            chunk_text = doc.page_content.strip()
            if not chunk_text or chunk_text in seen_texts:
                continue

            section = (doc.metadata.get("section", "") or "").strip().lower()
            subsection = (doc.metadata.get("subsection", "") or "").strip().lower()
            
            # --- Intent Filtering ---
            # Exclude FDD legal terms (ITEM 17, ITEM 23, etc.) for non-FDD queries
            if intent not in ["fdd_financial", "investment"]:
                if "item 17" in section or "item 23" in section or "receipts" in section:
                    continue

            # --- Semantic deduplication ---
            # Prevent near-duplicate FAQ chunks (e.g., "salary in Oregon" vs "salary in Mississippi")
            # by deduplicating on section+subsection identity
            # Normalize common FAQ patterns: "Frequently Asked Questions" sections with 
            # different state-specific subsections are effectively duplicates
            if section in ["frequently asked questions", "faqs", "faq"]:
                # Allow max 1 FAQ chunk per unique subsection pattern
                # Strip state names to catch "salary of home inspectors in X" duplicates
                import re
                normalized_sub = re.sub(r'\b(in|of|for)\s+\w+(\s+\w+)?\??$', '', subsection).strip()
                faq_key = f"faq::{normalized_sub}"
                if faq_key in seen_sections:
                    continue
                seen_sections.add(faq_key)

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
