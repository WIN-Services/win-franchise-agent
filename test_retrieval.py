import json
from app.tools.franchise_tools import get_investment_details

res = get_investment_details(query="investment breakdown")
for i, chunk in enumerate(res.get("retrieved_chunks", [])):
    print(f"\n--- Chunk {i} ---")
    print(chunk.get("text"))
