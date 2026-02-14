import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.llm.ollama_client import OllamaClient


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
OUTPUTS_DIR = ROOT / "outputs"


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def main():
    commits_path = DATA_DIR / "commits.csv"
    commits = pd.read_csv(commits_path)

    msg_col = "message" if "message" in commits.columns else "commit_message"
    commits[msg_col] = commits[msg_col].fillna("").astype(str)

    system_t = load_text(PROMPTS_DIR / "commit_label_system.txt")
    user_t = load_text(PROMPTS_DIR / "commit_label_user.txt")

    models = ["phi3:mini", "llama3.1:8b", "mistral:7b"]

    client = OllamaClient()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    for model in models:
        out_path = OUTPUTS_DIR / f"commit_labels__{model.replace(':','_')}.jsonl"
        n_written = 0

        with out_path.open("w", encoding="utf-8") as f:
            for msg in tqdm(commits[msg_col].tolist(), desc=f"Labeling with {model}"):
                msg = msg.strip()
                if not msg:
                    continue

                resp = client.chat(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_t},
                        {"role": "user", "content": render(user_t, message=msg)},
                    ],
                    temperature=0.2,
                    seed=42,
                )

                parsed, err = None, None
                try:
                    parsed = client.extract_json(resp["content"])
                except Exception as e:
                    err = str(e)

                record = {
                    "model": model,
                    "input_message": msg,
                    "parsed": parsed,
                    "raw_text": resp["content"],
                    "latency_s": resp["latency_s"],
                    "error": err,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

        print(f"Wrote {n_written} records to {out_path}")


if __name__ == "__main__":
    main()
