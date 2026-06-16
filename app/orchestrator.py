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
    "Costs & Investment": {
        "intent": "investment",
        "topic": "costs and investment",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise costs investment fees initial investment total cost breakdown next steps",
    },
    "🚀 How to Start My WIN Journey": {
        "intent": "getting_started",
        "topic": "how to get started",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise steps to get started application process how to start onboarding licensing requirements by state",
    },
    "Tools & Technology": {
        "intent": "general",
        "topic": "tools and technology",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection proprietary software tools CRM AI-powered reporting technology",
    },
    "Training Program": {
        "intent": "general",
        "topic": "training",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection training certifications mentorship coaching",
    },
    "Franchise Model": {
        "intent": "process",
        "topic": "business model",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise model business low cost home based timeline",
    },
    # --- New varied CTA and exploration button mappings ---
    "Book a Consultation": {
        "intent": "getting_started",
        "topic": "book consultation",
        "persona": "ready",
        "retrieval_query": "WIN Home Inspection franchise steps to get started application process onboarding consultation",
    },
    "Connect With Our Team": {
        "intent": "getting_started",
        "topic": "connect with team",
        "persona": "ready",
        "retrieval_query": "WIN Home Inspection franchise team consultation connect next steps application",
    },
    "Explore Territories": {
        "intent": "general",
        "topic": "territory availability",
        "persona": "comparing",
        "retrieval_query": "WIN Home Inspection franchise territory availability coverage area map",
    },
    "Territory Availability": {
        "intent": "general",
        "topic": "territory availability",
        "persona": "comparing",
        "retrieval_query": "WIN Home Inspection franchise territory availability coverage area map",
    },
    "Get a Personalized Plan": {
        "intent": "getting_started",
        "topic": "personalized plan",
        "persona": "ready",
        "retrieval_query": "WIN Home Inspection franchise personalized plan steps to get started onboarding timeline",
    },
    "Marketing Support": {
        "intent": "general",
        "topic": "marketing support",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection marketing support SEO paid ads lead generation online scheduler",
    },
    "Scalability": {
        "intent": "general",
        "topic": "scalability",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection scalability growth low overhead high return scaling business",
    },
    "Financing Options": {
        "intent": "investment",
        "topic": "financing",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection financing options in-house financing financial partners funding",
    },
    "Licensing Info": {
        "intent": "getting_started",
        "topic": "licensing requirements",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection licensing requirements state by state home inspector license training",
    },
    # --- Investment-specific drill-down buttons ---
    "What's Included": {
        "intent": "investment",
        "topic": "what's included in investment",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise investment what's included startup cost covers training technology marketing support",
    },
    "ROI & Payback": {
        "intent": "investment",
        "topic": "ROI and payback timeline",
        "persona": "comparing",
        "retrieval_query": "WIN Home Inspection franchise ROI return on investment payback period recoup initial investment timeline",
    },
    "Revenue Potential": {
        "intent": "investment",
        "topic": "revenue potential",
        "persona": "comparing",
        "retrieval_query": "WIN Home Inspection franchise revenue potential earnings income low overhead high return business model",
    },
    "Veteran Discounts": {
        "intent": "investment",
        "topic": "veteran discounts",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise veteran discount military first responder incentive",
    },
    "Compare Investment": {
        "intent": "investment",
        "topic": "investment comparison",
        "persona": "comparing",
        "retrieval_query": "WIN Home Inspection franchise investment compare lowest cost industry independent inspector startup cost comparison",
    },
    # --- Getting-started-specific drill-down buttons ---
    "Licensing Requirements": {
        "intent": "getting_started",
        "topic": "licensing requirements",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection licensing requirements state by state home inspector license certification regulations",
    },
    "Training Timeline": {
        "intent": "getting_started",
        "topic": "training timeline",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection training program timeline duration self-paced learning live sessions mentorship",
    },
    "State Requirements": {
        "intent": "getting_started",
        "topic": "state requirements",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection state specific requirements licensing training regulations certification by state",
    },
    "Application Process": {
        "intent": "getting_started",
        "topic": "application process",
        "persona": "ready",
        "retrieval_query": "WIN Home Inspection franchise application process steps onboarding how to apply join",
    },
    "What to Expect": {
        "intent": "getting_started",
        "topic": "what to expect",
        "persona": "exploring",
        "retrieval_query": "WIN Home Inspection franchise what to expect first year onboarding launch timeline milestones",
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
        # Determine intent pool and pool ctas
        pool_intent = current_intent if current_intent in ["investment", "getting_started"] else "general"
        
        # Look across all pools and remove the user's query if matched exactly. Also detect if it's a custom query.
        query_strip = query.strip()
        matched_cta = False
        for pool_key, pool in session_state.get("cta_pools", {}).items():
            if query_strip in pool:
                pool.remove(query_strip)
                matched_cta = True
                break
        
        # If the user typed something that didn't match any remaining CTA, they are in the custom flow
        if not matched_cta and turn_count >= 1:
            # We check turn_count >= 1 to not immediately flag the first ever message as custom 
            # unless it's a very specific environment where the first message isn't a CTA click.
            # Actually, the user starts the chat by sending a first message which is custom usually.
            # Let's just flag it as custom if it doesn't match any CTA.
            session_state["has_custom_query"] = True
                
        is_custom_flow = session_state.get("has_custom_query", False)
        pool_ctas = session_state.get("cta_pools", {}).get(pool_intent, [])
        
        # Threshold Logic:
        # Button Flow: show booking button on 3rd response (turn_count >= 2) OR exhausted pool
        # Custom Flow: show booking button on 2nd response (turn_count >= 1) OR exhausted pool
        if is_custom_flow:
            show_conversion_cta = (turn_count >= 1) or (len(pool_ctas) == 0)
        else:
            show_conversion_cta = (turn_count >= 2) or (len(pool_ctas) == 0)

        answer, extracted_demographics, wants_options = _agent.generate_response(
            query=query, 
            context_chunks=chunks, 
            history=history,
            topics_covered=session_state["topics_covered"],
            demographics=demographics,
            persona=current_persona,
            phone_collected=session_state["phone_collected"],
            intent=current_intent,
            show_conversion_cta=show_conversion_cta,
            is_custom_flow=is_custom_flow
        )

        # 3b. Programmatic CTA Options Override
        options = []
        if show_conversion_cta:
            options = ["Book a Free Consultation"]
        else:
            if is_custom_flow:
                # Custom flow: agent smartly decides whether to show the remaining generic options
                if wants_options:
                    options = list(pool_ctas)
                else:
                    options = []
            else:
                # Button flow: always aggressively provide the next options
                options = list(pool_ctas)

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
            "options": options,
            "demographics": extracted_demographics,
            "retrieved_chunks": chunks,
            "topic": current_topic,
            "persona": current_persona,
            "classified_intent": current_intent,
            "session_state": session_state
        }
        langfuse_client.update_current_span(output=response_data)
        return response_data
