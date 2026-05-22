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
    r"\b(fee|fees|cost|costs|investment|invest|capital|price|pricing|financ|afford|money|budget|pay|how much)\b",
    re.IGNORECASE,
)
_PROCESS_PATTERNS = re.compile(
    r"\b(step|steps|process|how to|how do|timeline|apply|application|join|become|start|licens|licensing|training|certification|certif|qualification|requirement|requirements|exam|inspector)\b",
    re.IGNORECASE,
)
_COMPETITOR_PATTERNS = re.compile(
    r"\b(competitor|other franchise|alternative|similar brand|rival|pillar to post|national property|amerispec|housemaster|us inspect|hometeam|brightside|a\-pro|win vs|better than|compared to|comparison|versus)\b",
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

import random

_USPS = [
    # Rankings & Awards
    "WIN Home Inspection has been ranked #1 year over year on the Entrepreneur Franchise 500 list for home inspection franchises!",
    "WIN is the #1 ranked and fastest-growing home inspection franchise in the US.",
    "WIN has been consistently ranked as one of the top franchises for veterans in the US.",
    # Training & Certifications
    "WIN provides in-house training and certifications for 35+ essential home inspection services — more than any other franchise, at no additional cost.",
    "WIN's training covers high-demand services like drone inspections, sewer scopes, radon testing, infrared scans, pool & spa inspections, and more.",
    "WIN's state-of-the-art training curriculum helps franchise owners launch with zero prior home inspection experience.",
    # Cost & Investment
    "WIN is the lowest-cost franchise in the home inspection industry with an all-inclusive model and no hidden fees.",
    "The total initial investment to own a WIN franchise ranges between just $41,200 and $49,800 — covering everything you need to launch.",
    "WIN offers a 10% discount on the initial franchise fee for Veterans and First Responders.",
    # Support & Marketing
    "WIN is the only franchise in the US offering in-house, end-to-end marketing support to help you generate new business year-round.",
    "WIN has assembled the largest support team in franchising on a per capita basis — including trainers, coaches, marketers, and technologists.",
    "WIN has the largest peer mentorship network in the industry, so you're never alone on your journey.",
    # Technology & Innovation
    "WIN uses AI-driven cloud services and proprietary tools like WINspect, WIN Concierge, and W-PASS to help franchise owners delight clients and grow faster.",
    "WIN franchise owners deliver inspection reports to clients within 24 hours using the proprietary WINspect software.",
    # Business Model
    "WIN's business model requires no storefront, no inventory, and no upfront staff — keeping overhead minimal.",
    "Home inspection is a recession-resistant industry, and WIN franchise owners can build multiple income streams year-round.",
    "WIN franchise owners can operate as a sole inspector or scale their team to multiple inspectors and locations.",
    # Community & Legacy
    "Since 1993, WIN has been supporting entrepreneurs nationwide with a proven business model and a track record of success.",
    "36% of WIN's franchise owners are Veterans and First Responders — the largest percentage in the industry!",
    # Success Stories
    "One WIN franchise owner surpassed $600,000 in revenue within just two years and completed over 1,000 inspections in a single year.",
    # State-Specific Training & Licensing
    "WIN offers state-specific training and licensing programs across most states in the US — and is the only home inspection company approved by the Texas Real Estate Commission (TREC).",
]


def get_greeting_response() -> str:
    usp = random.choice(_USPS)
    return (
        f"Hello! Welcome to WIN Home Inspection. Did you know? {usp}\n\n"
        "I'm your Franchise Assistant. Before we continue, are you looking to start a business or exploring career opportunities?"
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
    def route(self, query: str, history: list = None) -> Dict[str, Any]:
        """
        Classifies the intent of the query and calls the appropriate tool.
        Returns the raw tool output (list of chunks or a fallback dict).
        """
        history = history or []
        routing_text = query
        
        langfuse_client.update_current_span(input={"query": query})

        # 1. Explicit matches on the current query
        if _SMALL_TALK_PATTERNS.match(query):
            tool_name = "small_talk_greeting"
            result = {"status": "success", "answer": get_greeting_response()}
        elif _THANKS_PATTERNS.match(query):
            tool_name = "small_talk_thanks"
            result = {"status": "success", "answer": _THANKS_RESPONSE}
        elif _COMPETITOR_PATTERNS.search(query):
            tool_name = "get_franchise_info"
            # Rewrite query to retrieve WIN's USPs instead of competitor data
            usp_query = "Why WIN Home Inspection is the best franchise opportunity USPs advantages strengths " + query
            result = get_franchise_info(query=usp_query)
        elif _INVESTMENT_PATTERNS.search(query):
            tool_name = "get_investment_details"
            result = get_investment_details(query=query)
        elif _PROCESS_PATTERNS.search(query):
            tool_name = "get_process_steps"
            result = get_process_steps(query=query)
        else:
            # 2. Inherit intent from history if query is short (e.g. providing contact info, saying 'Yes')
            if len(query.split()) < 15 and len(history) > 0:
                history_user_msgs = " ".join([m["content"] for m in history if m["role"] == "user"])
                if _INVESTMENT_PATTERNS.search(history_user_msgs):
                    tool_name = "get_investment_details"
                    result = get_investment_details(query=query)
                elif _PROCESS_PATTERNS.search(history_user_msgs):
                    tool_name = "get_process_steps"
                    result = get_process_steps(query=query)
                else:
                    tool_name = "get_franchise_info"
                    result = get_franchise_info(query=query)
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
        tool_result = self.route(query, history=history)

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
