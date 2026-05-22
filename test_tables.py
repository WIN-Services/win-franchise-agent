import requests
import json

URL = "http://localhost:9000/chat"
queries = ["What is the investment breakdown?", "Show me the startup cost breakdown."]

for q in queries:
    res = requests.post(URL, json={"query": q}, timeout=60)
    print(f"Q: {q}\nA: {res.json().get('answer')}\n")
