import traceback
try:
    from app.orchestrator import Orchestrator
    o = Orchestrator()
    session_state = {
        "persona": "exploring",
        "topics_covered": [],
        "state_detected": None,
        "phone_collected": False,
        "has_custom_query": False,
        "cta_pools": {
            "general": ["Tools & Technology", "Training Program", "Marketing Support"],
            "investment": ["What's Included", "Financing Options", "ROI & Payback"],
            "getting_started": ["Licensing Requirements", "Training Timeline", "State Requirements"]
        }
    }
    result = o.run(query="How much money do I need?", history=[], session_state=session_state, demographics={})
    print("SUCCESS")
    print("Options:", result.get("options", []))
    print("Answer:", result.get("answer", "")[:200])
except Exception as e:
    traceback.print_exc()
