import asyncio
from app.orchestrator import Orchestrator

async def main():
    orc = Orchestrator()
    
    print("--- TURN 1 ---")
    res1 = orc.run("How much cost to start this business")
    print("Bot:", res1["answer"])
    
    print("\n--- TURN 2 ---")
    history = [{"role": "user", "content": "How much cost to start this business"}, {"role": "assistant", "content": res1["answer"]}]
    res2 = orc.run("John Doe, johndoe@gmail.com, +1-473298221", history=history)
    print("Bot:", res2["answer"])
    
    print("\n--- TURN 3 ---")
    history.append({"role": "user", "content": "John Doe, johndoe@gmail.com, +1-473298221"})
    history.append({"role": "assistant", "content": res2["answer"]})
    res3 = orc.run("Yes Please and my zip code is 94102", history=history)
    print("Bot:", res3["answer"])

if __name__ == "__main__":
    asyncio.run(main())
