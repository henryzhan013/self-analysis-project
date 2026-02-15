# GitHub Self-Analysis with Local LLMs

Analyzing my GitHub activity using local LLMs (Ollama) to extract insights about coding patterns and skills.

## Setup Instructions

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed

### Installation

```bash
# Clone repo
git clone https://github.com/henryzhan013/self-analysis-project.git
cd self-analysis-project

# Install dependencies
pip install pandas numpy matplotlib scikit-learn jupyter streamlit plotly requests

# Pull required models
ollama pull phi3:mini
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull nomic-embed-text
```

## How to Run Local LLM

```bash
# 1. Start Ollama server
ollama serve

# 2. Run analyses
python -m src.tasks.run_commit_labels      # Commit sentiment
python -m src.tasks.run_all_analyses       # All other analyses
python -m src.tasks.optimization_experiments  # Optimization tests

# 3. Launch dashboard
streamlit run app.py
```

## Key Findings (Top 5 Insights)

1. **Primary Languages**: Rust (62.6%) and Python (36.7%) dominate my codebase
2. **Coding Pattern**: Most active around 4-6 PM late afternoons
3. **Commit Style**: Short, minimal messages (avg 10 words) - could improve documentation
4. **Project Themes**: 3 clusters - Competitive Programming, Data Science, Web Apps
5. **Best Portfolio Project**: roadtrip-planner (full-stack with ML-powered search)

## Model Comparison

| Model | Avg Latency | Parse Success | Best For |
|-------|-------------|---------------|----------|
| phi3:mini | ~4s | ~85% | Quick, simple tasks |
| llama3.1:8b | ~2.5s | 100% | **Production use** |
| mistral:7b | ~2.1s | 100% | Speed-critical tasks |

**Optimization**: Temperature 0.3 gives best structured output reliability.

## Project Structure

```
├── app.py              # Streamlit dashboard
├── notebooks/          # Analysis notebooks (01-06)
├── src/
│   ├── llm/           # Ollama client
│   ├── rag/           # RAG system
│   └── tasks/         # Analysis scripts
├── outputs/           # Results (JSON, PNG)
├── prompts/           # LLM prompt templates
├── Prompts.md         # Prompt documentation
└── Technical_Report.md
```
