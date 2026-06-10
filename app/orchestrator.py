"""
Orchestrator for the Franchise Chatbot pipeline.

Responsibilities:
  1. Intent classification (dynamic LLM routing)
  2. Dispatch to the unified retrieve tool or fallback
  3. Pass retrieved context to the FranchiseAgent for LLM generation
  4. Return a structured response
"""

import random
from typing import Dict, Any, List, Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from langfuse import observe
from app.utils.langfuse_client import langfuse_client
from app.config import settings
from app.tools import (
    retrieve,
    fallback_no_answer,
)
from app.agents import FranchiseAgent

class QueryIntent(BaseModel):
    """Extraction model for classifying user intent."""
    intent: Literal["investment", "fdd_financial", "process", "getting_started", "competitor", "small_talk", "thanks", "employment", "general"] = Field(
        description="The intent category of the user query."
    )
    topic: str = Field(
        description="A short 1-4 word description of the specific topic the user is asking about (e.g. 'franchise costs', 'marketing support', 'training program', 'veteran discounts')."
    )
    persona: Literal["ready", "exploring", "comparing"] = Field(
        description="The prospect's persona based on signals. Urgency ('how soon', 'ready to invest') -> ready. Vague ('just looking') -> exploring. Comparison ('compare to AmeriSpec') -> comparing.",
        default="exploring"
    )


# ---------------------------------------------------------------------------
# Quick-reply button mappings – maps exact button text to intent + retrieval
# ---------------------------------------------------------------------------
_QUICK_REPLY_MAP = {
    "🔍 Explore WIN Franchise Opportunity": {
        "intent": "general",
        "topic": "WIN franchise opportunity",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise opportunity tools training process support technology brand what makes WIN different",
    },
    "💰 Costs and Investment": {
        "intent": "investment",
        "topic": "costs and investment",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise costs investment fees initial investment total cost breakdown next steps",
    },
    "🚀 How to Get Started": {
        "intent": "getting_started",
        "topic": "how to get started",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise steps to get started application process how to start onboarding licensing requirements by state",
    },
}

# ---------------------------------------------------------------------------
# US State names for detecting state-reply follow-ups
# ---------------------------------------------------------------------------
_US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
    "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
    "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada",
    "new hampshire", "new jersey", "new mexico", "new york",
    "north carolina", "north dakota", "ohio", "oklahoma", "oregon",
    "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington",
    "west virginia", "wisconsin", "wyoming",
    # Common abbreviations
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga",
    "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
    "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
    "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy",
}

# Abbreviation → full name mapping for canonical state names
_STATE_ABBREV_TO_FULL = {
    "al": "Alabama", "ak": "Alaska", "az": "Arizona", "ar": "Arkansas",
    "ca": "California", "co": "Colorado", "ct": "Connecticut", "de": "Delaware",
    "fl": "Florida", "ga": "Georgia", "hi": "Hawaii", "id": "Idaho",
    "il": "Illinois", "in": "Indiana", "ia": "Iowa", "ks": "Kansas",
    "ky": "Kentucky", "la": "Louisiana", "me": "Maine", "md": "Maryland",
    "ma": "Massachusetts", "mi": "Michigan", "mn": "Minnesota",
    "ms": "Mississippi", "mo": "Missouri", "mt": "Montana", "ne": "Nebraska",
    "nv": "Nevada", "nh": "New Hampshire", "nj": "New Jersey",
    "nm": "New Mexico", "ny": "New York", "nc": "North Carolina",
    "nd": "North Dakota", "oh": "Ohio", "ok": "Oklahoma", "or": "Oregon",
    "pa": "Pennsylvania", "ri": "Rhode Island", "sc": "South Carolina",
    "sd": "South Dakota", "tn": "Tennessee", "tx": "Texas", "ut": "Utah",
    "vt": "Vermont", "va": "Virginia", "wa": "Washington",
    "wv": "West Virginia", "wi": "Wisconsin", "wy": "Wyoming",
}

# Singleton agent – initialised once per process
_agent = FranchiseAgent()

class Orchestrator:
    """Routes a user query through the correct tool and the LLM agent."""

    def __init__(self):
        # Initialize an LLM specifically for fast intent classification
        self.intent_classifier = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        ).with_structured_output(QueryIntent)

    @observe(name="tool_routing")
    def route(self, query: str, history: list = None, demographics: dict = None, session_state: dict = None) -> Dict[str, Any]:
        """
        Classifies the intent of the query using an LLM and calls the appropriate tool.
        Returns the raw tool output (list of chunks or a fallback dict) along with the topic.
        """
        history = history or []
        demographics = demographics or {}
        session_state = session_state or {}
        
        langfuse_client.update_current_span(input={"query": query})

        # ── Quick-reply shortcut ──────────────────────────────────────
        # If the incoming text exactly matches a preset button, skip
        # the LLM classifier and use the pre-mapped intent directly.
        stripped_query = query.strip()
        quick_reply = _QUICK_REPLY_MAP.get(stripped_query)

        if quick_reply:
            classified_intent = quick_reply["intent"]
            topic = quick_reply["topic"]
            persona = quick_reply["persona"]
            retrieval_query = quick_reply["retrieval_query"]

            state = demographics.get("state")
            if state:
                retrieval_query += f" {state} requirements information"

            tool_name = "retrieve"
            result = retrieve(query=retrieval_query, intent=classified_intent if classified_intent != "getting_started" else "process")
            result["classified_intent"] = classified_intent
            result["topic"] = topic
            result["persona"] = persona

            langfuse_client.update_current_span(
                output={"tool_used": tool_name, "classified_intent": classified_intent, "topic": topic, "persona": persona, "result_status": result.get("status"), "quick_reply": True}
            )
            return result

        # ── State-reply detection ─────────────────────────────────────
        # When the bot asked "Which state are you looking to start in?"
        # and the user replies with just a state name, the LLM classifier
        # would misclassify it as small_talk/general. Detect this pattern
        # and force the getting_started intent with state-specific retrieval.
        topics_covered = session_state.get("topics_covered", [])
        query_lower = stripped_query.lower().strip(".,!?")

        if query_lower in _US_STATES and "how to get started" in topics_covered:
            # Resolve to canonical state name
            detected_state = _STATE_ABBREV_TO_FULL.get(query_lower, stripped_query.title())
            
            # Inject state into demographics immediately so it's used THIS turn
            demographics["state"] = detected_state

            gs_query = (
                f"WIN Home Inspection franchise steps to get started in {detected_state} "
                f"application process onboarding licensing certification requirements "
                f"{detected_state} home inspection license training regulations"
            )

            tool_name = "retrieve"
            result = retrieve(query=gs_query, intent="process")
            result["classified_intent"] = "getting_started"
            result["topic"] = f"getting started in {detected_state}"
            result["persona"] = "exploring"
            result["detected_state"] = detected_state

            langfuse_client.update_current_span(
                output={"tool_used": tool_name, "classified_intent": "getting_started", "topic": f"getting started in {detected_state}", "result_status": result.get("status"), "state_reply_detected": True}
            )
            return result

        # ── LLM-based intent classification ───────────────────────────
        messages = [
            SystemMessage(content=(
                "You are an intent classifier for a Franchise Chatbot. "
                "Classify the user's intent into one of the following categories:\n"
                "- fdd_financial: Questions about the Franchise Disclosure Document (FDD), financials, earnings, revenue, profit, legal terms, legal stipulations, agreements, contracts, termination, royalties, or Item 19.\n"
                "- investment: Questions about fees, costs, capital, budget, price, affordability (if not specifically FDD/earnings).\n"
                "- process: Questions about how the franchise works, the business model, tools, training, support system, or \"how it works\".\n"
                "- getting_started: Questions about steps to start, timeline, application, joining, onboarding, licensing, certification, or state-specific requirements to begin.\n"
                "- competitor: Questions comparing WIN to other franchises like Pillar To Post, AmeriSpec, HouseMaster, etc.\n"
                "- employment: Questions looking for a job, hiring, salary, resume, employment, vacancy, openings, or working for the company as an employee.\n"
                "- small_talk: Basic greetings (hello, hi, howdy).\n"
                "- thanks: Expressions of gratitude (thanks, thank you, cool, nice).\n"
                "- general: Anything else related to the franchise, business model, support, or general knowledge."
            ))
        ]
        
        # Include a bit of history to help classification if the query is very short
        if len(query.split()) < 15 and len(history) > 0:
            history_user_msgs = " ".join([m["content"] for m in history[-3:] if m["role"] == "user"])
            messages.append(HumanMessage(content=f"Recent History: {history_user_msgs}"))
            
        messages.append(HumanMessage(content=f"Query: {query}"))
        
        try:
            intent_result = self.intent_classifier.invoke(messages)
            classified_intent = intent_result.intent
            topic = intent_result.topic
            persona = intent_result.persona
        except Exception as e:
            print("Failed to classify intent:", e)
            classified_intent = "general"
            topic = "general information"
            persona = "exploring"

        # 2. Handle intent
        state = demographics.get("state")
        state_suffix = f" {state} requirements information" if state else ""
        
        if classified_intent == "employment":
            result = {"status": "employment", "answer": "Thank you for reaching out to WIN! We only provide franchising opportunities here and do not offer employment opportunities. Please check out other job portals for employment openings. We appreciate your interest."}
            tool_name = "direct_response"
        elif classified_intent in ["small_talk", "thanks"]:
            tool_name = "retrieve"
            # Retrieve USPs to weave into the greeting/small talk
            usp_query = "WIN Home Inspection advantages, USPs, franchise benefits, reasons to choose WIN" + (f" in {state}" if state else "")
            result = retrieve(query=usp_query, intent="general")
        elif classified_intent == "competitor":
            tool_name = "retrieve"
            # Rewrite query to retrieve WIN's USPs instead of competitor data
            usp_query = "Why WIN Home Inspection is the best franchise opportunity USPs advantages strengths " + query + state_suffix
            result = retrieve(query=usp_query, intent="general")
        elif classified_intent == "getting_started":
            tool_name = "retrieve"
            # Retrieve process/onboarding content but keep intent as getting_started
            gs_query = "WIN Home Inspection franchise steps to get started application process onboarding licensing requirements" + state_suffix
            result = retrieve(query=gs_query, intent="process")
        else:
            tool_name = "retrieve"
            result = retrieve(query=query + state_suffix, intent=classified_intent if classified_intent != "getting_started" else "process")

        result["classified_intent"] = classified_intent
        result["topic"] = topic
        result["persona"] = persona

        langfuse_client.update_current_span(
            output={"tool_used": tool_name, "classified_intent": classified_intent, "topic": topic, "persona": persona, "result_status": result.get("status")}
        )
        return result

    @observe(name="franchise-chatbot")
    def run(self, query: str, history: list = None, session_state: dict = None, demographics: dict = None) -> Dict[str, Any]:
        """
        Full pipeline:
          route → tool → agent → structured response
        Sets the top-level trace input/output.
        """
        history = history or []
        session_state = session_state or {
            "persona": "exploring",
            "topics_covered": [],
            "state_detected": None,
            "phone_collected": False
        }
        demographics = demographics or {}
        langfuse_client.update_current_span(input={"query": query, "history_len": len(history), "session_state": session_state})

        # 1. Tool routing (pass session_state so route() can detect state-reply follow-ups)
        tool_result = self.route(query, history=history, demographics=demographics, session_state=session_state)
        current_topic = tool_result.get("topic", "general information")
        new_persona = tool_result.get("persona")
        
        # If state-reply detection fired, ensure the detected state is in demographics
        # so it flows into the agent prompt AND persists via conversation manager
        if tool_result.get("detected_state"):
            demographics["state"] = tool_result["detected_state"]
            session_state["state_detected"] = tool_result["detected_state"]
        
        # Determine effective persona: if intent classifier provides a confident specific persona, persist it
        if new_persona and new_persona in ["ready", "comparing"]:
            session_state["persona"] = new_persona
            current_persona = new_persona
        else:
            current_persona = session_state["persona"]

        # 2. Early-exit for fallback or direct responses
        if tool_result.get("status") in ["fallback", "employment"] and "answer" in tool_result:
            response = {
                "answer": tool_result["answer"],
                "sources": [],
                "demographics": demographics,
                "retrieved_chunks": [],
                "topic": current_topic,
                "persona": current_persona,
                "session_state": session_state
            }
            langfuse_client.update_current_span(output=response)
            return response

        # 3. LLM generation using retrieved chunks
        chunks = tool_result.get("retrieved_chunks", [])
        current_intent = tool_result.get("classified_intent", "general")
        answer, extracted_demographics = _agent.generate_response(
            query=query, 
            context_chunks=chunks, 
            history=history,
            topics_covered=session_state["topics_covered"],
            demographics=demographics,
            persona=current_persona,
            phone_collected=session_state["phone_collected"],
            intent=current_intent
        )

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

        response_data = {
            "answer": answer,
            "sources": sources,
            "demographics": extracted_demographics,
            "retrieved_chunks": chunks,
            "topic": current_topic,
            "persona": current_persona,
            "classified_intent": current_intent,
            "session_state": session_state
        }
        langfuse_client.update_current_span(output=response_data)
        return response_data
