import asyncio
from app.tools.franchise_tools import get_investment_details
result = get_investment_details("Yes Please")
for c in result['retrieved_chunks']:
    print("---")
    print("Content:", c['text'])
