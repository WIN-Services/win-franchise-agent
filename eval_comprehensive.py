"""
Comprehensive Evaluation Script for WIN Franchise Chatbot.
Tests all major categories with multiple phrasings per question.
"""
import json
import sys

# We'll print results as JSON for easy parsing
EVAL_CASES = [
    # ===== CATEGORY 1: GREETING =====
    {"category": "Greeting", "id": "G1", "query": "Hi", "expect": ["WIN", "franchise", "job"], "history": []},
    {"category": "Greeting", "id": "G2", "query": "Hello there!", "expect": ["WIN", "franchise", "job"], "history": []},
    {"category": "Greeting", "id": "G3", "query": "Hey", "expect": ["WIN", "franchise", "job"], "history": []},

    # ===== CATEGORY 2: JOB VS FRANCHISE GATING =====
    {"category": "Job Gating", "id": "JG1", "query": "I am looking for a job",
     "expect": ["not providing employment", "local WIN"],
     "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Welcome to WIN..."}]},
    {"category": "Job Gating", "id": "JG2", "query": "I want to get hired as a home inspector",
     "expect": ["not providing employment", "local WIN"],
     "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Welcome to WIN..."}]},
    {"category": "Job Gating", "id": "JG3", "query": "I want to start a franchise business",
     "expect": ["franchise", "WIN"],
     "history": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Welcome to WIN..."}]},

    # ===== CATEGORY 3: LEAD GATING (Name + Email required) =====
    {"category": "Lead Gating", "id": "LG1", "query": "How much does it cost to start?",
     "expect": ["name", "email"],
     "history": []},
    {"category": "Lead Gating", "id": "LG2", "query": "What is the investment required?",
     "expect": ["name", "email"],
     "history": []},

    # ===== CATEGORY 4: LEAD UNLOCK (After providing name + email) =====
    {"category": "Lead Unlock", "id": "LU1",
     "query": "My name is John Doe and my email is johndoe@gmail.com. Tell me the cost.",
     "expect": ["41,200", "49,800"],
     "history": [
         {"role": "user", "content": "How much does it cost?"},
         {"role": "assistant", "content": "Could I get your name and email first?"}
     ]},

    # ===== CATEGORY 5: FORMAT VALIDATION =====
    {"category": "Format Validation", "id": "FV1",
     "query": "I'm John and my email is johndoe",
     "expect": ["email", "format", "@"],
     "history": [
         {"role": "user", "content": "How much?"},
         {"role": "assistant", "content": "Could I get your name and email?"}
     ]},
    {"category": "Format Validation", "id": "FV2",
     "query": "My number is 473298221",
     "expect": ["+1", "format"],
     "history": [
         {"role": "user", "content": "John Doe, johndoe@gmail.com"},
         {"role": "assistant", "content": "Thanks John! The investment is..."}
     ]},

    # ===== CATEGORY 6: INVESTMENT BREAKDOWN TABLE =====
    {"category": "Investment Breakdown", "id": "IB1",
     "query": "Can you give me a detailed breakdown of the investment?",
     "expect": ["Franchise Fee", "Equipment", "Marketing", "Insurance", "Additional"],
     "history": [
         {"role": "user", "content": "I am Jane Doe and my email is jane@test.com"},
         {"role": "assistant", "content": "Thank you Jane!"}
     ]},
    {"category": "Investment Breakdown", "id": "IB2",
     "query": "What are all the costs involved in starting a WIN franchise?",
     "expect": ["18,900", "21,000", "41,200", "49,800"],
     "history": [
         {"role": "user", "content": "I am Jane Doe and my email is jane@test.com"},
         {"role": "assistant", "content": "Thank you Jane!"}
     ]},
    {"category": "Investment Breakdown", "id": "IB3",
     "query": "How much money do I need to invest to open a WIN franchise?",
     "expect": ["41,200", "49,800"],
     "history": [
         {"role": "user", "content": "I am Jane Doe and my email is jane@test.com"},
         {"role": "assistant", "content": "Thank you Jane!"}
     ]},

    # ===== CATEGORY 7: STATE-SPECIFIC LICENSING =====
    {"category": "State Licensing", "id": "SL1",
     "query": "What are the licensing requirements in California?",
     "expect": ["California"],
     "history": []},
    {"category": "State Licensing", "id": "SL2",
     "query": "How do I become a home inspector in Alabama?",
     "expect": ["Alabama", "120 hours"],
     "history": []},
    {"category": "State Licensing", "id": "SL3",
     "query": "State specific licensing requirements for Texas",
     "expect": ["Texas", "TREC"],
     "history": []},
    {"category": "State Licensing", "id": "SL4",
     "query": "What training do I need in Florida to be a home inspector?",
     "expect": ["Florida"],
     "history": []},

    # ===== CATEGORY 8: COMPETITOR COMPARISON =====
    {"category": "Competitor Comparison", "id": "CC1",
     "query": "How is WIN better than Pillar to Post?",
     "expect": ["#1", "35+"],
     "not_expect": ["Pillar to Post"],
     "history": []},
    {"category": "Competitor Comparison", "id": "CC2",
     "query": "Why should I choose WIN over other franchises?",
     "expect": ["WIN"],
     "history": []},
    {"category": "Competitor Comparison", "id": "CC3",
     "query": "What makes WIN different from AmeriSpec?",
     "expect": ["WIN"],
     "not_expect": ["AmeriSpec"],
     "history": []},

    # ===== CATEGORY 9: PROCESS / STEPS =====
    {"category": "Process Steps", "id": "PS1",
     "query": "What are the steps to become a WIN franchise owner?",
     "expect": ["step", "training"],
     "history": []},
    {"category": "Process Steps", "id": "PS2",
     "query": "How do I join WIN Home Inspection?",
     "expect": ["WIN"],
     "history": []},
    {"category": "Process Steps", "id": "PS3",
     "query": "What is the process to start a WIN franchise?",
     "expect": ["franchise"],
     "history": []},

    # ===== CATEGORY 10: SPAM / PRIVACY REASSURANCE =====
    {"category": "Spam Reassurance", "id": "SR1",
     "query": "Will I get spam calls if I give you my number?",
     "expect": ["privacy", "spam"],
     "history": []},
    {"category": "Spam Reassurance", "id": "SR2",
     "query": "Are you going to send me promotional emails?",
     "expect": ["privacy", "reach out"],
     "history": []},

    # ===== CATEGORY 11: GENERAL WIN INFO =====
    {"category": "General Info", "id": "GI1",
     "query": "What does WIN Home Inspection do?",
     "expect": ["home inspection", "WIN"],
     "history": []},
    {"category": "General Info", "id": "GI2",
     "query": "Tell me about WIN Home Inspection franchise",
     "expect": ["franchise", "WIN"],
     "history": []},
    {"category": "General Info", "id": "GI3",
     "query": "Is WIN a good franchise to own?",
     "expect": ["WIN"],
     "history": []},

    # ===== CATEGORY 12: VETERAN BENEFITS =====
    {"category": "Veteran Benefits", "id": "VB1",
     "query": "Do you offer any discounts for veterans?",
     "expect": ["10%", "veteran"],
     "history": []},
    {"category": "Veteran Benefits", "id": "VB2",
     "query": "Is WIN good for military veterans?",
     "expect": ["veteran"],
     "history": []},

    # ===== CATEGORY 13: TRAINING =====
    {"category": "Training", "id": "TR1",
     "query": "What kind of training does WIN provide?",
     "expect": ["training", "35"],
     "history": []},
    {"category": "Training", "id": "TR2",
     "query": "Do I need prior experience to start a WIN franchise?",
     "expect": ["experience"],
     "history": []},
]

def run_eval():
    from app.orchestrator import Orchestrator
    orc = Orchestrator()
    
    results = []
    for case in EVAL_CASES:
        cid = case["id"]
        query = case["query"]
        history = case.get("history", [])
        expect = case.get("expect", [])
        not_expect = case.get("not_expect", [])
        
        try:
            res = orc.run(query, history=history)
            answer = res.get("answer", "")
        except Exception as e:
            answer = f"ERROR: {e}"
        
        # Check expected keywords
        hits = [kw for kw in expect if kw.lower() in answer.lower()]
        misses = [kw for kw in expect if kw.lower() not in answer.lower()]
        
        # Check not-expected keywords (should NOT appear)
        bad_hits = [kw for kw in not_expect if kw.lower() in answer.lower()]
        
        passed = len(misses) == 0 and len(bad_hits) == 0
        
        result = {
            "id": cid,
            "category": case["category"],
            "query": query,
            "passed": passed,
            "expected_found": hits,
            "expected_missing": misses,
            "unwanted_found": bad_hits,
            "answer_preview": answer[:300],
        }
        results.append(result)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} [{cid}] {case['category']}: {query[:60]}")
        if misses:
            print(f"   Missing: {misses}")
        if bad_hits:
            print(f"   Unwanted: {bad_hits}")
    
    # Summary
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed
    
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY: {passed}/{total} passed ({failed} failed)")
    print(f"{'='*60}")
    
    # Print failures
    if failed > 0:
        print("\nFAILED CASES:")
        for r in results:
            if not r["passed"]:
                print(f"\n  [{r['id']}] {r['category']}: {r['query']}")
                print(f"  Missing keywords: {r['expected_missing']}")
                print(f"  Unwanted keywords: {r['unwanted_found']}")
                print(f"  Answer: {r['answer_preview'][:200]}...")
    
    # Dump full results as JSON
    print("\n\n--- FULL RESULTS JSON ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_eval()
