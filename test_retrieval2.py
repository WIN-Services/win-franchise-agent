import os
import sys

# Ensure app is in path
sys.path.append(os.getcwd())

from app.tools.franchise_tools import get_investment_details

res = get_investment_details(query="investment breakdown")
chunks = res.get("retrieved_chunks", [])
print(f"Retrieved {len(chunks)} chunks.")
for i, chunk in enumerate(chunks):
    text = chunk.get("text", "")
    if "Amount" in text:
        print(f"[{i}] {text[:100].strip()}...")
