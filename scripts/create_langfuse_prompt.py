"""
Run this script ONCE to create the 'franchise-assistant-prompt' in your
local Langfuse instance (http://localhost:3000).

Usage:
    PYTHONPATH=. python3 scripts/create_langfuse_prompt.py
"""
import sys
from app.utils.langfuse_client import langfuse_client

PROMPT_NAME = "franchise-assistant-prompt"

PROMPT_TEXT = """\
You are the Franchise Assistant Agent for WIN Home Inspection.
Your goal is to help potential Strategic Partners understand the WIN franchise opportunity.
Answer the user's question based ONLY on the context provided below.

STRICT GUARDRAILS:
1. ONLY answer from the provided context. Do not invent facts.
2. NO HALLUCINATION. If the user asks a question and the answer is not in the context, respond with: "I don't have that information in my knowledge base. Please reach out to a WIN representative." However, if the user is just providing personal information (e.g., Name, Phone, Zip Code) or chatting, politely acknowledge it and continue the conversation.
3. DO NOT provide information about competitor franchise brands.
4. TONE & ENGAGEMENT: Frame responses positively, focusing on exciting opportunities with WIN. You MUST end EVERY single response with a direct question (ending in '?') to engage the prospect or ask for their details.
5. DEMOGRAPHICS: Throughout the conversation, proactively ask for the prospect's basic demographics (Name, Phone Number, Pin Code, Address) if they haven't provided them yet. Collect these naturally over time.
6. CALL TO ACTION: End your response with an encouraging call-to-action inviting them to schedule a call or consultation with the WIN team, immediately followed by your engaging question.

---
Context:
{{context}}

---
User Question:
{{query}}

---
Answer:\
"""

def main():
    print(f"Creating prompt '{PROMPT_NAME}' in Langfuse at {langfuse_client._base_url}...")

    try:
        prompt = langfuse_client.create_prompt(
            name=PROMPT_NAME,
            prompt=PROMPT_TEXT,
            labels=["production"],
            config={
                "model": "gpt-4-turbo",
                "temperature": 0,
                "variables": ["context", "query"]
            }
        )
        print(f"\n✅ Prompt created successfully!")
        print(f"   Name    : {PROMPT_NAME}")
        print(f"   Version : {prompt.version}")
        print(f"\nPrompt text (as stored):\n")
        print("-" * 60)
        print(PROMPT_TEXT)
        print("-" * 60)
        print("\nYou can now view and edit it in your Langfuse UI → Prompts tab.")
        
    except Exception as e:
        print(f"\n❌ Failed to create prompt: {e}")
        print("Make sure your LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set correctly in .env")
        sys.exit(1)
    finally:
        langfuse_client.flush()


if __name__ == "__main__":
    main()
