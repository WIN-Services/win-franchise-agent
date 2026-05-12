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
    name: Optional[str] = Field(None, description="Name of the prospect if provided")
    phone_number: Optional[str] = Field(None, description="Phone number of the prospect if provided")
    pin_code: Optional[str] = Field(None, description="Pin code or Zip code if provided")
    address: Optional[str] = Field(None, description="Address or location of the prospect if provided")


def _count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Returns the token count for a string using tiktoken."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _truncate_context(chunks: List[Dict[str, Any]], max_tokens: int, model: str) -> str:
    """
    Formats retrieved chunks into a context string, truncating to stay
    within max_tokens. Includes source attribution per chunk.
    """
    parts = []
    used = 0
    for chunk in chunks:
        source = chunk.get("metadata", {}).get("source", "Unknown")
        text = f"[Source: {source}]\n{chunk['text']}"
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

    @observe(name="llm_generation")
    def generate_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        history: List[Dict[str, str]] = None,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generates an LLM response and extracts demographics.
        Returns: (response_text, demographics_dict)
        """
        history = history or []

        # 1. Build token-budgeted context
        context_text = _truncate_context(
            context_chunks,
            max_tokens=settings.MAX_CONTEXT_TOKENS,
            model=settings.LLM_MODEL,
        )

        # 2. Compile the system/user prompt from Langfuse (or fallback)
        system_prompt = get_franchise_assistant_prompt(context=context_text, query=query)

        # 3. Build message list: system → history → current user message
        messages = [SystemMessage(content=system_prompt)]

        # We'll also collect the latest interaction for the extractor
        extractor_messages = []
        
        for msg in history:
            if msg["role"] == "user":
                m = HumanMessage(content=msg["content"])
                messages.append(m)
                extractor_messages.append(m)
            elif msg["role"] == "assistant":
                m = AIMessage(content=msg["content"])
                messages.append(m)
                extractor_messages.append(m)

        user_m = HumanMessage(content=query)
        messages.append(user_m)
        extractor_messages.append(user_m)

        # 4. Estimate token usage for Langfuse logging
        full_input_text = " ".join(m.content for m in messages if hasattr(m, 'content'))
        est_input_tokens = _count_tokens(full_input_text, settings.LLM_MODEL)

        langfuse_client.update_current_span(
            input={
                "model": self.llm.model_name,
                "max_output_tokens": settings.MAX_OUTPUT_TOKENS,
                "est_input_tokens": est_input_tokens,
                "history_messages": len(history),
                "context_tokens_used": _count_tokens(context_text, settings.LLM_MODEL),
                "final_prompt_preview": system_prompt[:400] + "..."
            }
        )

        # 5. Call the LLM
        try:
            ai_message = self.llm.invoke(messages)
            response_text = ai_message.content
            est_output_tokens = _count_tokens(response_text, settings.LLM_MODEL)
            
            # 6. Extract demographics using structured output
            # Add instruction for extractor
            extractor_messages.insert(0, SystemMessage(
                content="Extract the user's demographics (Name, Phone, Pin Code, Address) from the conversation history if present. If not present, leave fields null."
            ))
            
            demographics_result = self.extractor_llm.invoke(extractor_messages)
            demographics_dict = {
                k: v for k, v in demographics_result.model_dump().items() if v is not None
            }

            langfuse_client.update_current_span(
                output={
                    "response": response_text,
                    "demographics_extracted": demographics_dict,
                    "est_output_tokens": est_output_tokens,
                    "status": "success",
                }
            )
            return response_text, demographics_dict

        except Exception as e:
            langfuse_client.update_current_span(level="ERROR", status_message=str(e))
            raise
