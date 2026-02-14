import json, time, requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434", timeout=180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def chat(self, model, messages, temperature=0.2, seed=42):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "seed": seed},
        }
        t0 = time.time()
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        raw = r.json()
        return {
            "content": raw.get("message", {}).get("content", ""),
            "raw": raw,
            "latency_s": time.time() - t0,
        }

    @staticmethod
    def extract_json(text: str):
        t = text.strip()
        if t.startswith("```"):
            t = t.strip("`")
            lines = t.splitlines()
            if lines and lines[0].strip().lower() == "json":
                t = "\n".join(lines[1:]).strip()
        first = min([i for i in [t.find("{"), t.find("[")] if i != -1], default=-1)
        last = max(t.rfind("}"), t.rfind("]"))
        if first != -1 and last != -1 and last > first:
            t = t[first:last+1]
        return json.loads(t)
