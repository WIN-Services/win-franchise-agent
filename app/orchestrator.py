"""
Orchestrator for the Franchise Chatbot pipeline.

Responsibilities:
  1. Intent classification (keyword + semantic routing)
  2. Dispatch to the correct tool
  3. Pass retrieved context to the FranchiseAgent for LLM generation
  4. Return a structured response
"""

import re
from typing import Dict, Any, List

from langfuse import observe
from app.utils.langfuse_client import langfuse_client
from app.tools import (
    get_franchise_info,
    get_investment_details,
    get_process_steps,
    fallback_no_answer,
)
from app.agents import FranchiseAgent

# ---------------------------------------------------------------------------
# Intent rules: ordered list of (pattern, handler)
# ---------------------------------------------------------------------------
_INVESTMENT_PATTERNS = re.compile(
    r"\b(fee|fees|cost|costs|investment|invest|capital|price|pricing|financ|afford)\b",
    re.IGNORECASE,
)
_PROCESS_PATTERNS = re.compile(
    r"\b(step|steps|process|how to|how do|timeline|apply|application|join|become|start)\b",
    re.IGNORECASE,
)
_FALLBACK_PATTERNS = re.compile(
    r"\b(competitor|other franchise|alternative|similar brand|rival)\b",
    re.IGNORECASE,
)
_SMALL_TALK_PATTERNS = re.compile(
    r"^\s*(hi|hello|hey|howdy|greetings|good\s*(morning|afternoon|evening|day)|sup|what'?s\s*up|hiya|yo|namaste|hola|ola|ciao|bonjour|hiii+|heyyy+|helloooo+)[!?,\.\s]*$",
    re.IGNORECASE,
)
_THANKS_PATTERNS = re.compile(
    r"^\s*(thanks|thank you|ty|thx|thank\s*u|cheers|appreciate\s*it|great|awesome|perfect|got it|sounds good|ok|okay|cool|nice|wonderful)[!?,\.\s]*$",
    re.IGNORECASE,
)

_GREETING_RESPONSE = (
    "👋 Hello! Welcome to WIN Home Inspection — one of North America's fastest-growing franchise brands.\n\n"
    "I'm your Franchise Assistant, here to help you explore the WIN opportunity. "
    "I can help you with:\n"
    "  • 💰 Investment & fees\n"
    "  • 📋 Steps to become a franchise owner\n"
    "  • 🏠 What WIN Home Inspection does\n"
    "  • 📈 Training, support & territory\n\n"
    "Ready to take the first step toward owning your own business? Ask me anything!"
)

_THANKS_RESPONSE = (
    "You're welcome! 😊 Is there anything else you'd like to know about the WIN franchise opportunity? "
    "I'm happy to answer questions about investment details, the onboarding process, or anything else."
)

# Singleton agent – initialised once per process
_agent = FranchiseAgent()


class Orchestrator:
    """Routes a user query through the correct tool and the LLM agent."""

    @observe(name="tool_routing")
    def route(self, query: str) -> Dict[str, Any]:
        """
        Classifies the intent of the query and calls the appropriate tool.
        Returns the raw tool output (list of chunks or a fallback dict).
        """
        langfuse_client.update_current_span(input={"query": query})

        if _SMALL_TALK_PATTERNS.match(query):
            tool_name = "small_talk_greeting"
            result = {"status": "success", "answer": _GREETING_RESPONSE}
        elif _THANKS_PATTERNS.match(query):
            tool_name = "small_talk_thanks"
            result = {"status": "success", "answer": _THANKS_RESPONSE}
        elif _FALLBACK_PATTERNS.search(query):
            tool_name = "fallback_no_answer"
            result = fallback_no_answer()
        elif _INVESTMENT_PATTERNS.search(query):
            tool_name = "get_investment_details"
            result = get_investment_details()
        elif _PROCESS_PATTERNS.search(query):
            tool_name = "get_process_steps"
            result = get_process_steps()
        else:
            tool_name = "get_franchise_info"
            result = get_franchise_info(query=query)

        langfuse_client.update_current_span(
            output={"tool_used": tool_name, "result_status": result.get("status")}
        )
        return result

    @observe(name="franchise-chatbot")
    def run(self, query: str, history: list = None) -> Dict[str, Any]:
        """
        Full pipeline:
          route → tool → agent → structured response
        Sets the top-level trace input/output.
        """
        history = history or []
        langfuse_client.update_current_span(input={"query": query, "history_len": len(history)})

        # 1. Tool routing
        tool_result = self.route(query)

        # 2. Early-exit for fallback (no LLM needed)
        if tool_result.get("status") == "success" and "answer" in tool_result:
            response = {
                "answer": tool_result["answer"],
                "sources": [],
            }
            langfuse_client.update_current_span(output=response)
            return response

        # 3. LLM generation using retrieved chunks
        chunks: List[Dict[str, Any]] = tool_result.get("retrieved_chunks", [])
        answer, demographics = _agent.generate_response(query=query, context_chunks=chunks, history=history)

        # 4. Build sources from chunk metadata
        seen = set()
        sources = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            title = meta.get("source", "WIN Franchise Knowledge Base")
            url = meta.get("url")
            section = meta.get("section")
            key = (title, url)
            if key not in seen:
                seen.add(key)
                sources.append({"title": title, "url": url, "section": section})

        response = {
            "answer": answer, 
            "sources": sources,
            "demographics": demographics
        }
        langfuse_client.update_current_span(output=response)
        return response
