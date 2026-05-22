import asyncio
from app.orchestrator import Orchestrator

async def main():
    orc = Orchestrator()
    history = [{"role": "user", "content": "How much cost to start this business"}, {"role": "assistant", "content": "If you're interested, I can provide a breakdown of the costs involved in starting your WIN Home Inspection franchise. Would you like to see that? Also, could you let me know your zip code?"}]
    result = orc.run(query="Yes Please and my zip code is 94102", history=history)
    print("OUTPUT:\n", result["answer"])
    print("\nSOURCES:\n", result["sources"])

if __name__ == "__main__":
    asyncio.run(main())
