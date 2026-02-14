import json
from pathlib import Path

import pandas as pd

from src.llm.ollama_client import OllamaClient


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROMPTS_DIR = ROOT / "prompts"
OUTPUTS_DIR = ROOT / "outputs"

MODELS = ["phi3:mini", "llama3.1:8b", "mistral:7b"]


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render(template: str, **kwargs) -> str:
    for k, v in kwargs.items():
        template = template.replace("{{" + k + "}}", str(v))
    return template


def run_analysis(client, name, system_prompt, user_prompt, models=MODELS):
    """Run a single analysis with all models."""
    results = []

    for model in models:
        print(f"  Running with {model}...", end=" ", flush=True)
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            seed=42,
        )

        parsed, err = None, None
        try:
            parsed = client.extract_json(resp["content"])
        except Exception as e:
            err = str(e)

        record = {
            "analysis": name,
            "model": model,
            "parsed": parsed,
            "raw_text": resp["content"],
            "latency_s": resp["latency_s"],
            "error": err,
        }
        results.append(record)
        status = "OK" if err is None else "FAIL"
        print(f"{resp['latency_s']:.1f}s [{status}]")

    return results


def main():
    # Load data
    repos = pd.read_csv(DATA_DIR / "repos.csv")
    readmes = pd.read_csv(DATA_DIR / "readmes.csv")
    languages = pd.read_csv(DATA_DIR / "languages.csv")
    commits = pd.read_csv(DATA_DIR / "commits.csv")

    client = OllamaClient()
    all_results = []
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # ANALYSIS 2: Skill Extraction
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS 2: Skill Extraction from Repos")
    print("="*60)

    system_t = load_text(PROMPTS_DIR / "skills_system.txt")
    user_template = load_text(PROMPTS_DIR / "skills_user.txt")

    for _, repo in repos.iterrows():
        print(f"\nRepo: {repo['repo_name']}")
        readme_row = readmes[readmes["repo_name"] == repo["repo_name"]]
        readme_content = readme_row["readme_content"].iloc[0] if len(readme_row) > 0 else "No README"

        # Truncate long readmes
        if len(readme_content) > 1500:
            readme_content = readme_content[:1500] + "..."

        user_t = render(
            user_template,
            repo_name=repo["repo_name"],
            description=repo["description"] or "No description",
            language=repo["language"] or "Unknown",
            readme=readme_content
        )

        results = run_analysis(client, f"skills__{repo['repo_name']}", system_t, user_t)
        all_results.extend(results)

    # =========================================================================
    # ANALYSIS 3: Documentation Quality Assessment
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS 3: Documentation Quality Assessment")
    print("="*60)

    system_t = load_text(PROMPTS_DIR / "readme_quality_system.txt")
    user_template = load_text(PROMPTS_DIR / "readme_quality_user.txt")

    for _, readme in readmes.iterrows():
        print(f"\nRepo: {readme['repo_name']}")
        readme_content = readme["readme_content"]
        if len(readme_content) > 1500:
            readme_content = readme_content[:1500] + "..."

        user_t = render(
            user_template,
            repo_name=readme["repo_name"],
            readme=readme_content
        )

        results = run_analysis(client, f"readme_quality__{readme['repo_name']}", system_t, user_t)
        all_results.extend(results)

    # =========================================================================
    # ANALYSIS 4: Topic Clustering
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS 4: Topic Clustering of Projects")
    print("="*60)

    # Prepare repos summary with more detail
    repos_detail = []
    for _, r in repos.iterrows():
        readme_row = readmes[readmes["repo_name"] == r["repo_name"]]
        readme_preview = ""
        if len(readme_row) > 0:
            readme_preview = readme_row["readme_content"].iloc[0][:200]

        repos_detail.append(
            f"- {r['repo_name']}: {r['description'] or 'No description'}\n"
            f"  Language: {r['language'] or 'Unknown'}, Created: {r['created_at'][:10]}\n"
            f"  README preview: {readme_preview}..."
        )
    repos_text = "\n".join(repos_detail)

    system_t = load_text(PROMPTS_DIR / "topic_clustering_system.txt")
    user_t = render(load_text(PROMPTS_DIR / "topic_clustering_user.txt"), repos=repos_text)

    print("\nClustering all repos...")
    results = run_analysis(client, "topic_clustering", system_t, user_t)
    all_results.extend(results)

    # =========================================================================
    # ANALYSIS 5: What Should You Build Next
    # =========================================================================
    print("\n" + "="*60)
    print("ANALYSIS 5: What Should You Build Next")
    print("="*60)

    # Prepare summary data
    repos_summary = "\n".join([
        f"- {r['repo_name']}: {r['description'] or 'No description'} ({r['language'] or 'Unknown'})"
        for _, r in repos.iterrows()
    ])

    lang_summary = languages.groupby("language")["bytes"].sum().sort_values(ascending=False)
    lang_text = ", ".join([f"{lang} ({bytes/1000:.0f}KB)" for lang, bytes in lang_summary.head(5).items()])

    recent_commits = commits.sort_values("date", ascending=False).head(5)
    activity_text = "\n".join([
        f"- {c['date'][:10]}: {c['message'][:50]}..." if len(str(c['message'])) > 50 else f"- {c['date'][:10]}: {c['message']}"
        for _, c in recent_commits.iterrows()
    ])

    system_t = load_text(PROMPTS_DIR / "recommendations_system.txt")
    user_t = render(
        load_text(PROMPTS_DIR / "recommendations_user.txt"),
        repos=repos_summary,
        languages=lang_text,
        activity=activity_text
    )

    print("\nGenerating recommendations...")
    results = run_analysis(client, "recommendations", system_t, user_t)
    all_results.extend(results)

    # =========================================================================
    # Save Results
    # =========================================================================
    output_path = OUTPUTS_DIR / "llm_analyses.jsonl"
    with output_path.open("w", encoding="utf-8") as f:
        for record in all_results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"COMPLETE: Saved {len(all_results)} records to {output_path}")
    print("="*60)


if __name__ == "__main__":
    main()
