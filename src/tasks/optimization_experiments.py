"""
LLM Optimization Experiments

This script runs experiments to optimize LLM parameters:
1. Temperature comparison (0.1, 0.3, 0.5, 0.7)
2. Model speed vs quality trade-off analysis
3. Prompt length impact on latency
"""

import json
import time
from pathlib import Path

from src.llm.ollama_client import OllamaClient

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = ROOT / "outputs"

MODELS = ["phi3:mini", "llama3.1:8b", "mistral:7b"]
TEMPERATURES = [0.1, 0.3, 0.5, 0.7]

# Test prompt - commit classification
TEST_MESSAGE = "fix: resolve authentication bug in login flow"
SYSTEM_PROMPT = """You are a commit message analyzer. Return JSON with:
- type: one of [feature, bugfix, refactor, docs, test, chore]
- tone: one of [urgent, casual, technical, neutral]
- confidence: 1-10
Return ONLY valid JSON."""

USER_PROMPT = f'Analyze: "{TEST_MESSAGE}"'


def run_temperature_experiment(client: OllamaClient, model: str):
    """Compare different temperature settings."""
    results = []

    for temp in TEMPERATURES:
        successes = 0
        total_latency = 0
        runs = 3  # Multiple runs per temperature

        for _ in range(runs):
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=temp,
                seed=None  # No seed to test variance
            )

            total_latency += resp["latency_s"]

            try:
                parsed = client.extract_json(resp["content"])
                if "type" in parsed and "tone" in parsed:
                    successes += 1
            except:
                pass

        results.append({
            "model": model,
            "temperature": temp,
            "success_rate": successes / runs,
            "avg_latency": total_latency / runs
        })
        print(f"  temp={temp}: {successes}/{runs} success, {total_latency/runs:.2f}s avg")

    return results


def run_model_comparison(client: OllamaClient):
    """Compare models on speed vs quality."""
    results = []

    for model in MODELS:
        print(f"\nTesting {model}...")

        latencies = []
        successes = 0
        runs = 5

        for i in range(runs):
            resp = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT}
                ],
                temperature=0.3,
                seed=42
            )

            latencies.append(resp["latency_s"])

            try:
                parsed = client.extract_json(resp["content"])
                if "type" in parsed and "tone" in parsed:
                    successes += 1
            except:
                pass

        results.append({
            "model": model,
            "avg_latency": sum(latencies) / len(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "success_rate": successes / runs,
            "quality_score": successes / runs * 100,
            "speed_score": 100 / (sum(latencies) / len(latencies)),  # inverse latency
        })

    return results


def run_prompt_length_experiment(client: OllamaClient):
    """Test how prompt length affects latency."""
    results = []

    prompts = [
        ("minimal", "Classify: fix bug"),
        ("short", f'Analyze commit: "{TEST_MESSAGE}"'),
        ("medium", f'Analyze this commit message and classify it: "{TEST_MESSAGE}". Return type and tone.'),
        ("long", f'You are analyzing git commit messages. Given the following commit message: "{TEST_MESSAGE}", please classify it by determining the type of change (feature, bugfix, refactor, etc.) and the tone of the message (urgent, casual, technical, neutral). Return your analysis.')
    ]

    model = "llama3.1:8b"

    for name, prompt in prompts:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            seed=42
        )

        results.append({
            "prompt_type": name,
            "prompt_length": len(prompt),
            "latency_s": resp["latency_s"]
        })
        print(f"  {name} ({len(prompt)} chars): {resp['latency_s']:.2f}s")

    return results


def main():
    client = OllamaClient()
    all_results = {
        "temperature_experiments": [],
        "model_comparison": [],
        "prompt_length": []
    }

    # 1. Temperature experiments
    print("=" * 60)
    print("EXPERIMENT 1: Temperature Optimization")
    print("=" * 60)

    for model in MODELS:
        print(f"\n{model}:")
        results = run_temperature_experiment(client, model)
        all_results["temperature_experiments"].extend(results)

    # 2. Model comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Model Speed vs Quality")
    print("=" * 60)

    all_results["model_comparison"] = run_model_comparison(client)

    # Print summary
    print("\nModel Comparison Summary:")
    print("-" * 50)
    for r in all_results["model_comparison"]:
        print(f"{r['model']:15} | Latency: {r['avg_latency']:.2f}s | Success: {r['success_rate']*100:.0f}%")

    # 3. Prompt length experiment
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Prompt Length Impact")
    print("=" * 60)

    all_results["prompt_length"] = run_prompt_length_experiment(client)

    # Save results
    output_path = OUTPUTS_DIR / "optimization_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print optimization recommendations
    print("\n" + "=" * 60)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 60)

    # Best temperature
    temp_results = all_results["temperature_experiments"]
    best_temp = max(temp_results, key=lambda x: x["success_rate"])
    print(f"\n1. Optimal Temperature: {best_temp['temperature']}")
    print(f"   - Highest success rate with structured output")

    # Best model
    model_results = all_results["model_comparison"]
    best_quality = max(model_results, key=lambda x: x["success_rate"])
    best_speed = min(model_results, key=lambda x: x["avg_latency"])

    print(f"\n2. Best Quality Model: {best_quality['model']}")
    print(f"   - {best_quality['success_rate']*100:.0f}% success rate")

    print(f"\n3. Fastest Model: {best_speed['model']}")
    print(f"   - {best_speed['avg_latency']:.2f}s average latency")

    # Prompt length finding
    prompt_results = all_results["prompt_length"]
    print(f"\n4. Prompt Length Impact:")
    for r in prompt_results:
        print(f"   - {r['prompt_type']}: {r['latency_s']:.2f}s")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
