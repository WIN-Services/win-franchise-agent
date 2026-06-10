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
                          intent: str = "general") -> Tuple[str, dict]:
        """
        Generates a final answer utilizing the provided chunks.
        Also performs an inline extraction of demographics to determine if the lead is 'hot'.
        Returns a tuple: (answer_string, updated_demographics_dict)
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
            "5. TONE AND CONSULTATIVE STYLE (CRITICAL): You MUST adopt the GO Team communication standards (Yogesh Kandpal). Your responses must feel highly consultative, personalized, and never like a scripted FAQ answer. Use conversational connecting phrases based on what the user has shared (e.g., 'Basis you are from [state] and interested in...', 'From what you\\'ve shared, WIN could align well with your goals...', 'Looks like you are getting closer to evaluating the right franchise fit...'). Always tie their specific context into your answers.",
            "6. FORMATTING RULE (CRITICAL): You MUST always bold adjectives that describe WIN's ranking, size, or achievements (e.g., **Rank 1**, **#1 Ranked**, **35+ services**).",
        ]
        
        # --- PHASE 1: Natural Spice & Conversational Variability ---
        if len(history) == 0:
            # FIRST INTERACTION: Use storytelling to build trust
            if intent == "general":
                enforced_reminders.append(
                    "7. WIN FRANCHISE OPPORTUNITY (FIRST INTERACTION): The user wants to explore the WIN franchise opportunity. "
                    "Structure your response to be SHORT, PRECISE, and STRUCTURED for a small chat window. "
                    "a) Open with 1-2 punchy lines about what makes WIN a unique franchise (NO generic adjectives like 'amazing', 'incredible' — use SPECIFICS). "
                    "b) FORMAT RULE (CRITICAL): You MUST output exactly 3-4 concise markdown bullet points (using '-') covering: "
                    "   • What tools & tech WIN provides (e.g., proprietary InspectorTech platform, AI-powered reporting, CRM) "
                    "   • What training looks like (e.g., 35+ certifications, hands-on mentorship, ongoing coaching) "
                    "   • The process/model (e.g., low overhead, home-based, 6-week launch timeline) "
                    "   DO NOT write paragraphs. Keep each bullet to 1 line max. "
                    "c) Close with a quick franchise owner quote — pick randomly from context. Max 2 lines. "
                    "CRITICAL: Total response must fit a small chat window. No fluff. Every word earns its place. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. "
                    "Do NOT use the phrases 'Here are a few key benefits...', 'Here's how...', or start any sentence with 'At WIN Home Inspection, we...'. Sound like a passionate consultant, not a brochure."
                )
            elif intent == "investment":
                enforced_reminders.append(
                    "7. COSTS & INVESTMENT (FIRST INTERACTION): The user wants to understand costs and investment. "
                    "Structure your response to be SHORT, PRECISE, and STRUCTURED for a small chat window. "
                    "a) Open with 1 punchy line framing WIN as a smart, low-risk investment. "
                    "b) FORMAT RULE (CRITICAL): You MUST render a clean Markdown table showing the cost breakdown from the context. "
                    "   Keep the table compact — use short column headers. "
                    "c) After the table, add 2-3 bullet points covering: next steps to move forward, financing options if mentioned in context, and what's included in the investment. "
                    "d) DO NOT end with a long paragraph. Keep the closing to 1 line max. "
                    "CRITICAL: Total response must fit a small chat window. Be precise — no filler words. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. "
                    "Do NOT use generic openers. Sound direct and knowledgeable."
                )
            elif intent == "getting_started":
                enforced_reminders.append(
                    "7. HOW TO GET STARTED (FIRST INTERACTION): The user wants to know how to start a WIN franchise. "
                    "Structure your response to be SHORT, PRECISE, and STRUCTURED for a small chat window. "
                    "a) FIRST — check if the user's state is known (from KNOWN USER INFO). "
                    "   • If YES: Open with 1 excited line about their state, then give steps. "
                    "   • If NO: Open with 1 excited line, then ASK: 'Which state are you looking to start in? I'll give you the exact steps for your area.' "
                    "b) FORMAT RULE (CRITICAL): You MUST output numbered steps (using '1.', '2.', etc.). "
                    "   Each step MUST be max 5-8 words. Example format: "
                    "   1. Submit your application online "
                    "   2. Attend WIN Discovery Day "
                    "   3. Get approved & sign agreement "
                    "   4. Complete training & certification "
                    "   5. Launch your WIN business "
                    "   NO explanations or paragraphs inside steps. Just crisp action items. "
                    "c) Total response: 1 intro line + steps + 1 closing line. Nothing more. "
                    "ANTI-PATTERN RULE (CRITICAL): Do NOT start with 'Great question', 'Absolutely', or 'That's a fantastic'."
                )
            elif intent == "process":
                enforced_reminders.append(
                    "7. STORYTELLING (FIRST INTERACTION): Structure your response to be SHORT and PUNCHY. "
                    "a) Open with the Home Inspection Industry's massive scale and potential. You MUST include scale-based details (e.g., $6+ Billion industry, growing market demand, high frequency of inspections during home sales, etc.). "
                    "b) FORMAT RULE (CRITICAL): You MUST output exactly 3 markdown bullet points (using '-') highlighting how the WIN Franchise approach is better to start with, specifically explaining how it gives them an established brand name that already has built-in trust with clients, along with other key benefits from the context. DO NOT write this as a paragraph. "
                    "c) Close with a quick validation point. DO NOT keep it blunt or one-line. Instead, add an emotionally excited, engaging setup (maximum 2 lines) and then present the testimony quote from a Franchise owner. (CRITICAL: Randomly select a different testimony from the provided context each time to avoid repeating the same quote.) "
                    "CRITICAL: Keep the overall answer brief. Max 3-4 sentences outside the bullets. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. Do NOT use the phrases 'Here are a few...', 'Here's how...', or 'Here is a quick overview'. Speak like a charismatic consultant over coffee—be unpredictable and passionate."
                )
            elif intent == "competitor":
                enforced_reminders.append(
                    "7. STORYTELLING (FIRST INTERACTION): Build trust with a SHORT, excited narrative comparing WIN to other options. "
                    "a) Address their question head-on. "
                    "b) FORMAT RULE (CRITICAL): You MUST output 4 to 5 markdown bullet points (using '-') highlighting WIN's strengths over competitors. You MUST explicitly include in-house tech or AI-Driven tech as the 4th point. DO NOT write this as a paragraph. "
                    "c) Close with a quick validation point. DO NOT keep it blunt or one-line. Instead, add an emotionally excited, engaging setup (maximum 2 lines) and then present the testimony quote from a Franchise owner. (CRITICAL: Randomly select a different testimony from the provided context each time to avoid repeating the same quote.) "
                    "CRITICAL: Keep the overall answer brief. Max 3-4 sentences outside the bullets. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. Do NOT use generic openers like 'Here are a few...', or 'Here's how...'. Vary your structure. Sound human and passionate."
                )
            else:
                enforced_reminders.append(
                    "7. STORYTELLING (FIRST INTERACTION): Build trust with a SHORT, excited narrative. "
                    "a) Address their question head-on. "
                    "b) FORMAT RULE (CRITICAL): You MUST output exactly 3 markdown bullet points (using '-') highlighting WIN's strengths relevant to their question. DO NOT write this as a paragraph. "
                    "c) Close with a quick validation point. DO NOT keep it blunt or one-line. Instead, add an emotionally excited, engaging setup (maximum 2 lines) and then present the testimony quote from a Franchise owner. (CRITICAL: Randomly select a different testimony from the provided context each time to avoid repeating the same quote.) "
                    "CRITICAL: Keep the overall answer brief. Max 3-4 sentences outside the bullets. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. Do NOT use generic openers like 'Here are a few...', or 'Here's how...'. Vary your structure. Sound human and passionate."
                )
        else:
            # SUBSEQUENT INTERACTIONS: Direct, concise, no story needed
            if intent == "getting_started":
                # User likely just provided their state — give state-specific steps
                enforced_reminders.append(
                    "7. STATE-SPECIFIC GETTING STARTED (FOLLOW-UP): The user has just provided their state. "
                    "a) Open with 1 excited line about their state as a WIN market. "
                    "b) FORMAT RULE (CRITICAL): Output numbered steps (1., 2., etc.) — each step MAX 1 short line. "
                    "   Weave in state-specific licensing/certification from the context. "
                    "   Example: '4. Complete [State] home inspection licensing (WIN covers training)' "
                    "   If no state-specific info in context, say WIN's team will guide them through their state's requirements. "
                    "c) After steps, add 1 line CTA to book consultation for a personalized walkthrough. "
                    "TOTAL response: 1 intro line + steps + 1 CTA line. No paragraphs. No filler. "
                    "ANTI-PATTERN: Do NOT give generic steps. Mention the state name in at least 2 steps. "
                    "Do NOT start with 'Great question' or 'Absolutely'."
                )
            else:
                enforced_reminders.append(
                    "7. DIRECT ANSWERING (FOLLOW-UP): You have already established rapport. "
                    "Answer the user's specific question DIRECTLY and CONCISELY. No storytelling structure needed. "
                    "Provide highly relevant information strictly matching their intent—do NOT bleed into unrelated topics. "
                    "Keep it tight, conversational, and enthusiastic. 3-5 sentences max unless the question demands more detail. "
                    "ANTI-PATTERN RULE: Do NOT start with 'Great question' or 'Absolutely'."
                )
        
        if intent == "competitor":
            enforced_reminders.append(
                "8. COMPETITOR RULE (CRITICAL): You MUST explicitly state in your response that WIN is the number one ranked franchise in the home inspection industry. Keep your tone consultative and confident, but NEVER be dismissive of competitors. Wherever you mention 'Ranked #1 - Entrepreneur' or 'Entrepreneur' magazine, you MUST make the word 'Entrepreneur' bold as well (e.g., Ranked #1 - **Entrepreneur**)."
            )
            enforced_reminders.append(
                "8a. WIN ADVANTAGE RULE (CRITICAL): When comparing WIN with other options, you MUST refer to and heavily leverage 'The WIN Advantage' points from the context. Present these advantages as very crisp, short bullet points to make the comparison highly impactful and easy to read."
            )
            
        if intent == "process":
            enforced_reminders.append(
                "9. PROCESS FORMATTING RULE (CRITICAL): When answering questions about how the process works or getting started, you MUST include the process of registration (as mentioned in the context) and convert the steps into very short 2-3 word bullet points. After listing these bullet points, seamlessly transition into discussing the rest of your response (e.g., support, technology), ensuring the overall answer stays concise."
            )
            
        if intent == "investment":
            enforced_reminders.append(
                "8. INVESTMENT FORMATTING RULE (CRITICAL): When answering questions about costs, fees, or investment, you MUST construct and render a clean, properly formatted Markdown table showing the complete breakdown of the total investment from the provided context. Preserve expenditure names, amount ranges, and values EXACTLY as they appear. Provide a short intro sentence before the table and a seamless transition after it."
            )
        
        if intent == "getting_started":
            enforced_reminders.append(
                "9. GETTING STARTED RULE (CRITICAL): You MUST format the startup journey as numbered steps (1., 2., 3., etc.). "
                "If the user's state is NOT known, you MUST ask 'Which state are you looking to start in?' before listing steps. "
                "If the state IS known, weave in state-specific licensing and certification requirements from the context. "
                "Keep each step to ONE short line. Do NOT write paragraphs."
            )
        
        if intent not in ["fdd_financial", "getting_started"]:
            enforced_reminders.append(
                "10. CTA RULE (CRITICAL): You MUST end your response with a natural CTA guiding them to 'press the button below to Book a Consultation with us'. "
                "You MUST randomly pick ONE of the following transitions so you never sound repetitive. Do NOT use casual phrases. Use professional transitions like: "
                "1) 'To know more details, please press the button below...' "
                "2) 'For next steps, simply press the button below...' "
                "3) 'To explore this opportunity further, press the button below...' "
                "4) 'For a detailed discussion tailored to your goals, press the button below...' "
                "Do NOT use the exact same transition twice in a row. NEVER ask for their name or email."
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
            return response_text, combined_demographics

        except Exception as e:
            langfuse_client.update_current_span(level="ERROR", status_message=str(e))
            raise
