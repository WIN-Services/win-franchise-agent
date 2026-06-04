from typing import List, Dict, Any, Tuple, Optional
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


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Returns the token count for a string using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


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

        # 2. Extract demographics first to determine lead status
        extractor_messages = [
            SystemMessage(content="Extract the user's demographics (Name, Email, Phone, Pin Code, Address, State) from the conversation history. If not present, leave fields null.")
        ]
        if len(history) > 0:
            history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
            extractor_messages.append(HumanMessage(content=f"History:\n{history_text}"))
        extractor_messages.append(HumanMessage(content=f"Current Query: {query}"))
        
        try:
            demographics_result = self.extractor_llm.invoke(extractor_messages)
            extracted_demographics_dict = {
                k: v for k, v in demographics_result.model_dump().items() if v is not None
            }
        except Exception:
            extracted_demographics_dict = {}

        # Merge previously known demographics with newly extracted ones
        combined_demographics = {**demographics, **extracted_demographics_dict}
        
        # We consider a lead complete if they have name and email
        lead_profile_complete = bool(combined_demographics.get("name") and combined_demographics.get("email"))

        # 3. Build the primary response generation prompt
        system_prompt = get_franchise_assistant_prompt(
            context=context_text, 
            query=query, 
            lead_profile_complete=lead_profile_complete, 
            topics_covered=topics_covered,
            demographics=combined_demographics,
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
            if intent == "process":
                enforced_reminders.append(
                    "7. STORYTELLING (FIRST INTERACTION): Structure your response to be SHORT and PUNCHY. "
                    "a) Open with the Home Inspection Industry's massive scale and potential. You MUST include scale-based details (e.g., $6+ Billion industry, growing market demand, high frequency of inspections during home sales, etc.). "
                    "b) FORMAT RULE (CRITICAL): You MUST output exactly 3 markdown bullet points (using '-') highlighting how the WIN Franchise approach is better to start with, specifically explaining how it gives them an established brand name that already has built-in trust with clients, along with other key benefits from the context. DO NOT write this as a paragraph. "
                    "c) Close with a quick validation point. DO NOT keep it blunt or one-line. Instead, add an emotionally excited, engaging setup (maximum 2 lines) and then present the testimony quote from a Franchise owner. (CRITICAL: Randomly select a different testimony from the provided context each time to avoid repeating the same quote.) "
                    "CRITICAL: Keep the overall answer brief. Max 3-4 sentences outside the bullets. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. Do NOT use the phrases 'Here are a few...', 'Here's how...', or 'Here is a quick overview'. Speak like a charismatic consultant over coffee—be unpredictable and passionate."
                )
            elif intent == "general":
                enforced_reminders.append(
                    "7. STORYTELLING (FIRST INTERACTION): Structure your response to be SHORT and PUNCHY. "
                    "a) Open with genuine enthusiasm about their specific question—directly address it first. "
                    "b) FORMAT RULE (CRITICAL): You MUST output exactly 3 markdown bullet points (using '-') highlighting WIN's key differentiators (e.g., Support, Tech, Brand). DO NOT write this as a paragraph. "
                    "c) Close with a quick validation point. DO NOT keep it blunt or one-line. Instead, add an emotionally excited, engaging setup (maximum 2 lines) and then present the testimony quote from a Franchise owner. (CRITICAL: Randomly select a different testimony from the provided context each time to avoid repeating the same quote.) "
                    "CRITICAL: Keep the overall answer brief. Max 3-4 sentences outside the bullets. "
                    "ANTI-PATTERN RULE (CRITICAL): You are STRICTLY FORBIDDEN from starting your response with 'Great question', 'That's a great question', 'Absolutely', or 'That's a fantastic'. Do NOT use the phrases 'Here are a few key benefits...', 'Here's how...', or start any sentence with 'At WIN Home Inspection, we...'. Sound like a passionate human, not a brochure."
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
            enforced_reminders.append(
                "7. DIRECT ANSWERING (FOLLOW-UP): You have already established rapport. "
                "Answer the user's specific question DIRECTLY and CONCISELY. No storytelling structure needed. "
                "Provide highly relevant information strictly matching their intent—do NOT bleed into unrelated topics. "
                "Keep it tight, conversational, and enthusiastic. 3-5 sentences max unless the question demands more detail. "
                "ANTI-PATTERN RULE: Do NOT start with 'Great question' or 'Absolutely'."
            )
        
        if intent == "competitor":
            enforced_reminders.append(
                "8. COMPETITOR RULE (CRITICAL): You MUST explicitly state in your response that WIN is the number one ranked franchise in the home inspection industry. Keep your tone consultative and confident, but NEVER be dismissive of competitors. YOU ARE STRICTLY FORBIDDEN from citing, mentioning, or referencing 'Entrepreneur' magazine in any form. If the context contains 'Entrepreneur' (e.g. '#1 Ranked Franchise - Entrepreneur'), you MUST omit the word completely and just say '#1 Ranked Franchise'."
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
        
        if intent != "fdd_financial":
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

        # 6. Call the LLM
        try:
            ai_message = self.llm.invoke(messages)
            response_text = ai_message.content
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
