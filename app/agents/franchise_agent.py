import re
import json
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
from langfuse import observe
import tiktoken

from app.config import settings
from app.utils.langfuse_client import langfuse_client, get_franchise_assistant_prompt


class Demographics(BaseModel):
    """Extraction model for prospect demographics."""
    name: Optional[str] = Field(None, description="Full name of the prospect (must be First_Name Last_Name). If only a first name is given, return None.")
    email: Optional[str] = Field(None, description="Email address of the prospect (must be in valid varchar@domain format). If invalid, return None.")
    phone_number: Optional[str] = Field(None, description="Phone number of the prospect (must be in +1-XXXXXXXXXX format). If invalid, return None.")
    pin_code: Optional[str] = Field(None, description="Pin code or Zip code if provided")
    address: Optional[str] = Field(None, description="Address or location of the prospect if provided")
    state: Optional[str] = Field(None, description="The US State of the prospect if provided (e.g. Texas, Florida, California)")


# Pre-warm tiktoken encoder as a module-level singleton to avoid per-call overhead
_tiktoken_cache: Dict[str, tiktoken.Encoding] = {}

def _get_encoder(model: str = "gpt-4o-mini") -> tiktoken.Encoding:
    """Returns a cached tiktoken encoder for the given model."""
    if model not in _tiktoken_cache:
        try:
            _tiktoken_cache[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            _tiktoken_cache[model] = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_cache[model]

def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Returns the token count for a string using tiktoken."""
    return len(_get_encoder(model).encode(text))


import urllib.parse

def _truncate_context(chunks: List[Dict[str, Any]], max_tokens: int, model: str) -> str:
    """
    Formats retrieved chunks into a structured context string, truncating to stay
    within max_tokens. Follows strict numbered [Source X] formatting with Section, URL & Content tags.
    """
    parts = []
    used = 0
    for i, chunk in enumerate(chunks):
        # Pull section from metadata, fall back to "General" if not present
        section = chunk.get("metadata", {}).get("section", "General Information")
        source_url = chunk.get("metadata", {}).get("source_url", "")
        
        url_line = ""
        if source_url and source_url.startswith("http"):
            # Create a text fragment using the first 30 characters of the chunk text
            # to make the browser scroll directly to the exact location.
            snippet = chunk['text'][:50].strip()
            # Modern browsers scroll to the text if appended with #:~:text=
            encoded_snippet = urllib.parse.quote(snippet)
            # Make sure not to append if URL already has a hash
            if "#" not in source_url:
                source_url = f"{source_url}#:~:text={encoded_snippet}"
            
            url_line = f"\nURL: {source_url}"
        
        # Format to exact specifications
        text = f"[Source {i + 1}]\nSection: {section}{url_line}\nContent:\n{chunk['text']}"
        
        tok = _count_tokens(text, model)
        if used + tok > max_tokens:
            break
        parts.append(text)
        used += tok
    return "\n\n".join(parts)


class FranchiseAgent:
    """
    Agent responsible for generating a strictly guardrailed LLM response
    based on the retrieved context and conversation history.
    Also extracts prospect demographics in the background.
    """

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
            max_tokens=settings.MAX_OUTPUT_TOKENS,
        )
        self.extractor_llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        ).with_structured_output(Demographics)
        # Thread pool for running demographics extraction in parallel with main LLM
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agent")
        # Pre-warm the tiktoken encoder at init time
        _get_encoder(settings.LLM_MODEL)

    @observe(name="generate_response")
    def generate_response(self, 
                          query: str, 
                          context_chunks: List[Dict[str, Any]], 
                          history: list = None,
                          topics_covered: list = None,
                          demographics: dict = None,
                          persona: str = "exploring",
                          phone_collected: bool = False,
                          intent: str = "general",
                          show_conversion_cta: bool = False,
                          is_custom_flow: bool = False) -> Tuple[str, dict, bool]:
        """
        Generates a final answer utilizing the provided chunks.
        Also performs an inline extraction of demographics to determine if the lead is 'hot'.
        Returns a tuple: (answer_string, updated_demographics_dict, wants_options)
        """
        history = history or []
        topics_covered = topics_covered or []
        demographics = demographics or {}
        
        # 1. Build token-budgeted context
        context_text = _truncate_context(
            context_chunks,
            max_tokens=settings.MAX_CONTEXT_TOKENS,
            model=settings.LLM_MODEL,
        )

        # 2. Determine lead status from ALREADY-KNOWN demographics (previous turns)
        #    This avoids blocking on the extractor LLM before the main generation.
        lead_profile_complete = bool(demographics.get("name") and demographics.get("email"))

        # 3. Build the primary response generation prompt
        system_prompt = get_franchise_assistant_prompt(
            context=context_text, 
            query=query, 
            lead_profile_complete=lead_profile_complete, 
            topics_covered=topics_covered,
            demographics=demographics,
            persona=persona,
            phone_collected=phone_collected,
            intent=intent
        )
        
        enforced_reminders = [
            "1. BRANDING RULE (CRITICAL): You MUST always refer to the business as 'WIN' or 'WIN Home Inspection'. YOU ARE STRICTLY FORBIDDEN from outputting the exact sequence of words 'the franchise' unless it is immediately followed by 'agreement', 'owner', 'team', 'system', or 'fee'. In all other cases, you MUST replace 'the franchise', 'this franchise', or 'the company' with 'WIN'. Furthermore, the word 'WIN' MUST appear naturally at least once in your response. This is an absolute, non-negotiable rule.",
            "2. Make sure to include an inline clickable markdown link to the provided source URL if one is present in the context, e.g. [Read more](https://wini.com/...).",
            "3. If PREVIOUS TOPICS DISCUSSED is provided in the system prompt, you MUST smartly detect if the user is asking about a past topic again, and ONLY THEN start your response with bridging language (e.g. 'Since we talked about...'). DO NOT bridge on every single response.",
            "4. If a STATE-SPECIFIC ADDENDUM is requested in the system prompt, you MUST automatically weave the relevant state-specific facts (e.g., licensing, training required for that state) from the provided context into your answer without the user explicitly asking.",
            "5. TONE AND CONSULTATIVE STYLE (CRITICAL): You MUST adopt the Franchise Development team's consultative communication style. Your responses must feel like a 1-on-1 conversation with a knowledgeable franchise consultant — warm, direct, and personalized. NEVER sound like a scripted FAQ answer or a brochure. KEY BEHAVIORS: "
            "a) Give SHORT, direct answers first (2-3 sentences), then end with a NATURAL follow-up question that advances the conversation (e.g., 'What state do you plan to operate in?', 'How are you planning to fund your business?', 'Is there an area of marketing you'd like to learn more about?'). "
            "b) Use conversational connecting phrases based on what the user has shared (e.g., 'From what you've shared...', 'Since you're looking at [state]...'). "
            "c) When a question is state-dependent (like licensing), ASK which state instead of giving a generic answer. "
            "d) Frame WIN's advantages through the user's lens, not as a sales pitch (e.g., 'the potential is in your hands' instead of 'you will earn a lot'). "
            "e) When there are nearby WIN inspectors, frame it as an ADVANTAGE (existing brand awareness, mentorship network), not a concern. "
            "f) Normalize common concerns — e.g., 'No worries! Many of our franchise owners don't have an inspection background.' "
            "g) Always be honest about variability (e.g., costs depend on area, timeline depends on state licensing). Do NOT over-promise.",
            "6. FORMATTING RULE (CRITICAL): You MUST always bold ONLY the adjectives and features that directly describe WIN's ranking, size, or specific achievements (e.g., **Rank 1**, **#1 Ranked**, **35+ in-house certifications**). DO NOT bold generic adjectives.",
            "7. TECHNOLOGY RULE (CRITICAL): Whenever mentioning tools, apps, or technology, keep the response generic (e.g. 'proprietary software', 'mobile app'). DO NOT mention specific names like 'InspectorTech', 'WINspect', or 'WINconnect'. DO NOT use casual terms like 'state of the art'.",
            "8. RANKING INTRO RULE (CRITICAL): You MUST naturally weave the fact that WIN is the **#1 Ranked** franchise into the very first opening sentence of your response, regardless of the topic. Every single response MUST establish this prestige right at the start.",
            "9. LAYERED CONVERSATION RULE (CRITICAL): Do NOT output long bulleted lists. Provide ONE bite-sized piece of high-value information (2-3 sentences max). "
            "a) ALWAYS end with a natural follow-up question that advances the conversation (e.g., 'What state do you plan to operate in?', 'How are you planning to fund your business?'). "
            "b) ANTI-PATTERN: NEVER start a sentence with 'Absolutely!' or 'Great question!' Be natural.",
        ]
        
        # --- PHASE 1: Natural Spice & Conversational Variability ---
        if len(history) == 0:
            # FIRST INTERACTION: Use storytelling to build trust
            if intent == "general":
                enforced_reminders.append(
                    "10. WIN FRANCHISE OPPORTUNITY: Structure your response to be SHORT, PRECISE, and CONVERSATIONAL. "
                    "a) Open with 1-2 punchy lines about what makes WIN a unique franchise (NO generic adjectives like 'amazing', 'incredible' — use SPECIFICS). "
                    "b) Keep the response short to encourage them to click one of the buttons to dig deeper. "
                    "c) For your BUTTON OPTIONS, you MUST prioritize 'Tools & Technology' as the FIRST button, followed by 'Training Program' and 'Franchise Model'. Do NOT include 'Costs & Investment' or 'How to Start My WIN Journey' as first-turn buttons — lead with value. "
                    "ANTI-PATTERN RULE (CRITICAL): Do NOT use the phrases 'Here are a few key benefits...', 'Here's how...'. Sound like a passionate consultant, not a brochure."
                )
            elif intent == "investment":
                enforced_reminders.append(
                    "10. COSTS & INVESTMENT: Structure your response to be SHORT, PRECISE, and CONVERSATIONAL. "
                    "a) FORMAT RULE (CRITICAL): Do NOT generate a Markdown table of costs. Do NOT break down the cost line by line. "
                    "b) CONVERSATIONAL COST FRAMING: Present the investment conversationally — mention they can expect to invest around $40,000-$50,000 within their first six months, and that it's one of the lowest in the industry. This covers everything needed to launch and grow. "
                    "c) If they want a detailed line-by-line breakdown, guide them to connect with the team or review the Franchise Disclosure Document. "
                    "d) END with a natural follow-up question like 'How are you planning to fund your business?' or 'How soon are you looking to get started?' "
                    "e) ANTI-PATTERN RULE (CRITICAL): Do NOT use generic openers."
                )
            elif intent == "getting_started":
                enforced_reminders.append(
                    "10. GETTING STARTED: "
                    "Structure your response to be SHORT, PRECISE, and CONVERSATIONAL. "
                    "a) Be excited but concise — acknowledge their interest warmly. "
                    "b) Mention the average timeline is about 8-12 weeks, but it can vary depending on state licensing, training, and their vision. "
                    "c) End with a natural follow-up: ask what state they plan to operate in (if not known) or suggest connecting with the team for a personalized walkthrough. "
                    "d) Do NOT give them numbered steps or a rigid process list."
                )
            elif intent == "process":
                enforced_reminders.append(
                    "10. STORYTELLING (FIRST INTERACTION): Structure your response to be SHORT and PUNCHY. "
                    "a) Open with the Inspection Services Industry's massive scale and potential ($6+ Billion industry). "
                    "b) Explain how the WIN Franchise approach gives them an established brand name with built-in trust. "
                    "c) Do NOT output bullet points. Keep it to 2-3 sentences. "
                )
            elif intent == "competitor":
                enforced_reminders.append(
                    "10. STORYTELLING (FIRST INTERACTION): Build trust with a SHORT, excited narrative comparing WIN to other options. "
                    "a) Address their question head-on, keeping it to 2-3 sentences. "
                    "b) Explicitly mention WIN's in-house tech or AI-Driven tech as an advantage. "
                    "c) Do NOT use bullet points. "
                )
            else:
                enforced_reminders.append(
                    "10. STORYTELLING (FIRST INTERACTION): Build trust with a SHORT, excited narrative. "
                    "a) Address their question head-on. "
                    "b) Do NOT use bullet points. "
                    "c) Keep the overall answer brief. Max 3-4 sentences. "
                    "d) End with a natural follow-up question to keep the conversation going."
                )
        else:
            # SUBSEQUENT INTERACTIONS: Direct, concise, no story needed
            enforced_reminders.append(
                "10. DIRECT ANSWERING (FOLLOW-UP): You have already established rapport. "
                "Answer the user's specific question DIRECTLY and CONCISELY. No storytelling structure needed. "
                "Provide highly relevant information strictly matching their intent—do NOT bleed into unrelated topics. "
                "Keep it tight, conversational, and enthusiastic. 2-4 sentences max unless the question demands more detail. "
                "IMPORTANT: End with a NATURAL follow-up question that advances the conversation — ask about their specific situation, their state, their timeline, their funding, etc. This builds trust and keeps the dialogue flowing. "
                "ANTI-PATTERN RULE: Do NOT start with 'Great question' or 'Absolutely'."
            )
        
        if intent == "competitor":
            enforced_reminders.append(
                "8. COMPETITOR RULE (CRITICAL): You MUST explicitly state in your response that WIN is the number one ranked franchise in the inspection services industry. Keep your tone consultative and confident, but NEVER be dismissive of competitors. Wherever you mention 'Ranked #1 - Entrepreneur' or 'Entrepreneur' magazine, you MUST make the word 'Entrepreneur' bold as well (e.g., Ranked #1 - **Entrepreneur**)."
            )
            enforced_reminders.append(
                "8a. WIN ADVANTAGE RULE (CRITICAL): When comparing WIN with other options, you MUST refer to and heavily leverage 'The WIN Advantage' points from the context. Present these advantages as very crisp, short bullet points to make the comparison highly impactful and easy to read."
            )
            
        if intent == "process":
            enforced_reminders.append(
                "10. PROCESS FORMATTING RULE (CRITICAL): Do NOT output bullet points. Summarize the process in 2-3 short sentences."
            )
            
        if intent == "investment":
            enforced_reminders.append(
                "10. INVESTMENT FORMATTING RULE (CRITICAL): Do NOT output a Markdown table for costs. Focus on the value and the '$40-$50k' conversational verbiage."
            )

        # Add the directional phrase rule at the very end to guarantee recency bias
        if show_conversion_cta:
            enforced_reminders.append(
                "11. FINAL STEP - CONVERSION CALL TO ACTION (CRITICAL): Right at the end of your response, you MUST write a smart directional phrase inviting them to book a free consultation or intro call. "
                "Examples: 'If you're ready to dive deeper, let's hop on a quick call.', 'I'd love to connect you with our team when you're ready.', 'Tap below to book a free consultation!'"
            )
        else:
            if is_custom_flow:
                enforced_reminders.append(
                    "11. FINAL STEP - CUSTOM CHAT (CRITICAL): The user is asking custom questions. "
                    "If you feel that showing generic quick-reply options (like 'Tools & Technology' or 'Financing Options') would be helpful right now, add the exact tag `[SHOW_OPTIONS]` at the very end of your response. "
                    "If you want to just have a natural conversation without showing buttons, do NOT include `[SHOW_OPTIONS]`."
                )
            else:
                enforced_reminders.append(
                    "11. FINAL STEP - EXPLORATION (CRITICAL): You are showing the user some quick-reply buttons. DO NOT add any extra 'pick one below' directional phrases right now. Just ask your natural follow-up question and end there."
                )

        # Append the enforced reminders to the system prompt so the LLM must obey them
        system_prompt += "\n\nCRITICAL FORMATTING INSTRUCTIONS:\n" + "\n".join(enforced_reminders)
        
        # 4. Build message list: system → history → current user message
        messages = [SystemMessage(content=system_prompt)]
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=query))

        # 5. Estimate token usage for Langfuse logging
        full_input_text = " ".join(m.content for m in messages if hasattr(m, 'content'))
        est_input_tokens = _count_tokens(full_input_text, settings.LLM_MODEL)

        langfuse_client.update_current_span(
            input={
                "model": self.llm.model_name,
                "max_output_tokens": settings.MAX_OUTPUT_TOKENS,
                "est_input_tokens": est_input_tokens,
                "history_messages": len(history),
                "context_tokens_used": _count_tokens(context_text, settings.LLM_MODEL),
                "final_prompt_preview": system_prompt[:400] + "...",
                "lead_profile_complete": lead_profile_complete,
                "prompt_label": "fdd_compliant" if intent == "fdd_financial" else "production"
            }
        )

        # 6. Run main LLM generation and demographics extraction IN PARALLEL
        #    Demographics extraction doesn't affect the current response — it only
        #    updates the stored profile for the *next* turn.
        def _extract_demographics():
            """Background task: extract demographics from conversation."""
            extractor_messages = [
                SystemMessage(content="Extract the user's demographics (Name, Email, Phone, Pin Code, Address, State) from the conversation history. If not present, leave fields null.")
            ]
            if len(history) > 0:
                history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
                extractor_messages.append(HumanMessage(content=f"History:\n{history_text}"))
            extractor_messages.append(HumanMessage(content=f"Current Query: {query}"))
            
            try:
                demographics_result = self.extractor_llm.invoke(extractor_messages)
                return {
                    k: v for k, v in demographics_result.model_dump().items() if v is not None
                }
            except Exception:
                return {}

        try:
            # Submit both tasks to run concurrently
            demo_future = self._executor.submit(_extract_demographics)
            
            # Main LLM call runs on the current thread (this is the critical path)
            ai_message = self.llm.invoke(messages)
            response_text = ai_message.content
            
            wants_options = False
            if "[SHOW_OPTIONS]" in response_text:
                wants_options = True
                response_text = response_text.replace("[SHOW_OPTIONS]", "").strip()
            
            # Collect demographics result (should be done by now since main LLM is slower)
            extracted_demographics_dict = demo_future.result(timeout=10)
            
            # Merge previously known demographics with newly extracted ones
            combined_demographics = {**demographics, **extracted_demographics_dict}

            est_output_tokens = _count_tokens(response_text, settings.LLM_MODEL)

            langfuse_client.update_current_span(
                output={
                    "response": response_text,
                    "demographics_extracted": combined_demographics,
                    "est_output_tokens": est_output_tokens,
                    "status": "success",
                }
            )
            return response_text, combined_demographics, wants_options

        except Exception as e:
            langfuse_client.update_current_span(
                level="ERROR",
                status_message=str(e),
            )
            return "I'm currently unable to access the information. Could you please provide your phone number so our team can reach out to you directly?", demographics, False
