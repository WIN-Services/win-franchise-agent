from langfuse import Langfuse
from langfuse import observe
from app.config import settings

# 1. Reusable Client Initialization
langfuse_client = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_BASE_URL
)

# 2. Prompt Management Helper
def get_franchise_assistant_prompt(context: str, query: str) -> str:
    """
    Fetches the 'franchise-assistant-prompt' dynamically from Langfuse
    and compiles it with the given variables.
    Provides a strict fallback if the prompt is not found.
    """
    try:
        prompt = langfuse_client.get_prompt("franchise-assistant-prompt")
        return prompt.compile(context=context, query=query)
    except Exception as e:
        print(f"Warning: Could not fetch 'franchise-assistant-prompt' from Langfuse ({e}). Using fallback.")
        return f"""You are the Franchise Assistant Agent for WIN Home Inspection.
Your goal is to answer the user's questions based ONLY on the provided context in a warm, enthusiastic, and positive tone.

STRICT GUARDRAILS:
1. ONLY answer from the provided context.
2. NO HALLUCINATION. If the user asks a question and the answer is not in the context, say "I don't know based on the provided information." However, if the user is just providing personal information (e.g., Name, Phone, Zip Code) or chatting, politely acknowledge it and continue the conversation.
3. DO NOT provide any competitor information.
4. TONE & ENGAGEMENT: Frame responses positively, focusing on exciting opportunities with WIN. You MUST end EVERY single response with a direct question (ending in '?') to engage the prospect or ask for their details.
5. DEMOGRAPHICS: Throughout the conversation, proactively ask for the prospect's basic demographics (Name, Phone Number, Pin Code, Address) if they haven't provided them yet. Collect these naturally over time.
6. CALL TO ACTION: End your response with an encouraging call-to-action inviting them to schedule a call or consultation with the WIN team, immediately followed by your engaging question.

Context:
{context}

User Query:
{query}

Answer:"""

# 3. Example Trace Wrapper Function
# Since langfuse >= 3.x, use @observe decorators instead of manual client.trace()
@observe(name="franchise-chatbot")
def create_pipeline_trace(query: str, session_id: str = None):
    """
    Creates a master trace for the chatbot pipeline.
    The trace context is automatically propagated to nested @observe spans.
    """
    # The current trace context is active inside this function's scope.
    pass
    # Spans can be created using @observe(as_type="span") on other functions called within.
    pass

# -------------------------------------------------------------------
# USAGE INSTRUCTIONS (To be imported across modules):
#
# from app.utils.langfuse_client import langfuse_client, get_franchise_assistant_prompt
# from langfuse import observe
#
# @observe(name="franchise-chatbot")
# def process_chat(query: str):
#     langfuse_client.update_current_trace(input={"query": query})
#     
#     retrieved_chunks = retrieve(query)
#     
#     # ... logic ...
#     
#     langfuse_client.update_current_trace(output=response)
#     return response
#
# @observe(as_type="span", name="retrieval")
# def retrieve(query: str):
#     # ... logic ...
#     return chunks
# -------------------------------------------------------------------
