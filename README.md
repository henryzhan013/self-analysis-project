# GitHub Self-Analysis Project

Analyzing my own GitHub activity using local LLMs to uncover patterns, extract insights, and build a personal code assistant.

## Project Overview

This project demonstrates the use of **local Large Language Models (LLMs)** for analyzing GitHub repository data. It combines traditional data science techniques with modern LLM-powered analysis to provide comprehensive insights into coding patterns, skills, and recommendations.

### Key Features

- **Multi-Model Comparison**: Compare outputs from 3 local LLMs (phi3:mini, llama3.1:8b, mistral:7b)
- **5 LLM-Powered Analyses**: Sentiment analysis, skill extraction, documentation quality, topic clustering, and project recommendations
- **Traditional Data Science**: 6+ visualizations and 2 ML techniques (K-Means clustering, time series forecasting)
- **RAG System**: Retrieval-Augmented Generation for querying your own code
- **Interactive Dashboard**: Streamlit app for exploring all analyses

## Results Summary

### Activity Profile
- **7 repositories** analyzed
- **23 commits** across all repos
- **Primary languages**: Rust (62.6%), Python (36.7%)
- **Most active**: Sundays at 22:00 Central Time

### LLM Analysis Insights
- **Commit types**: Mix of feature additions, bug fixes, and chores
- **Top skills identified**: Rust, Python, data analysis, competitive programming
- **README quality**: Average score 6.2/10 across repositories
- **Recommended next projects**: Based on skill gaps and portfolio balance

### Model Performance
| Model | Avg Latency | Parse Success |
|-------|-------------|---------------|
| phi3:mini | ~4s | ~85% |
| llama3.1:8b | ~10s | ~95% |
| mistral:7b | ~7s | ~90% |

## Project Structure

```
self_analysis_project/
├── app.py                    # Streamlit dashboard
├── Prompts.md               # Prompt engineering documentation
├── README.md                # This file
│
├── data/                    # Raw GitHub data
│   ├── repos.csv
│   ├── commits.csv
│   ├── languages.csv
│   ├── readmes.csv
│   └── events.csv
│
├── notebooks/               # Analysis notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_commit_labels_old.ipynb
│   ├── 03_commit_sentiment_analysis.ipynb
│   ├── 04_llm_analyses.ipynb
│   ├── 05_traditional_data_science.ipynb
│   └── 06_rag_code_assistant.ipynb
│
├── outputs/                 # Generated outputs
│   ├── commit_labels.jsonl
│   ├── llm_analyses.jsonl
│   ├── rag_index/
│   └── *.png (visualizations)
│
├── prompts/                 # LLM prompt templates
│   ├── sentiment_system.txt
│   ├── sentiment_user.txt
│   ├── skills_system.txt
│   ├── skills_user.txt
│   └── ...
│
└── src/                     # Source code
    ├── llm/
    │   └── ollama_client.py
    ├── rag/
    │   └── code_rag.py
    └── tasks/
        ├── run_commit_labels.py
        └── run_all_analyses.py
```

## Installation

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running

### Setup

1. Clone the repository:
```bash
git clone <repo-url>
cd self_analysis_project
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib scikit-learn jupyter streamlit plotly requests
```

3. Pull required Ollama models:
```bash
ollama pull phi3:mini
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull nomic-embed-text
```

## Usage

### Run the Dashboard
```bash
streamlit run app.py
```

### Run Analyses

1. **Commit Labeling** (Analysis 1):
```bash
python -m src.tasks.run_commit_labels
```

2. **All Other Analyses** (Analyses 2-5):
```bash
python -m src.tasks.run_all_analyses
```

### Explore Notebooks

```bash
jupyter notebook notebooks/
```

Notebooks are numbered in order:
1. Data exploration and sanity checks
2. Original commit labeling (2 models)
3. Commit sentiment analysis with model comparison
4. LLM analyses (skills, README quality, clustering, recommendations)
5. Traditional data science (visualizations + ML)
6. RAG code assistant demo

## LLM Analyses

### 1. Commit Sentiment Analysis
Classifies commit messages by **type** (feature, bugfix, refactor, etc.) and **tone** (urgent, casual, technical, etc.).

### 2. Skill Extraction
Identifies technical skills demonstrated in each repository based on code, documentation, and metadata.

### 3. README Quality Assessment
Scores documentation quality (1-10) and checks for key sections (description, installation, usage, contributing).

### 4. Topic Clustering
Groups repositories into thematic clusters to identify focus areas and expertise.

### 5. Project Recommendations
Suggests what to build next based on current skills and portfolio gaps.

## Advanced: RAG System

The project includes a **Retrieval-Augmented Generation (RAG)** system that allows natural language queries about your repositories:

```python
from src.rag.code_rag import CodeRAG

rag = CodeRAG()
rag.load("outputs/rag_index")

result = rag.query("What programming languages do I use most?")
print(result["answer"])
```

**Components:**
- **Embeddings**: nomic-embed-text (768 dimensions)
- **Vector Store**: NumPy-based cosine similarity
- **Generation**: llama3.1:8b

## Traditional Data Science

### Visualizations
1. Commit activity timeline
2. Language distribution (pie + bar)
3. Activity heatmap (day × hour)
4. Repository comparison
5. Commit message length analysis
6. Contribution timeline by repo

### Machine Learning
1. **K-Means Clustering**: Groups commits by characteristics (message length, time, day)
2. **Time Series Forecasting**: Linear regression to predict future commit activity

## Technologies Used

- **LLMs**: Ollama (phi3:mini, llama3.1:8b, mistral:7b)
- **Embeddings**: nomic-embed-text
- **Data Analysis**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, plotly
- **Dashboard**: Streamlit
- **Data Source**: GitHub API

## Prompt Engineering

See [Prompts.md](Prompts.md) for detailed documentation of all prompts, including:
- System and user prompt templates
- Design rationale for each prompt
- Best practices applied
- Model comparison findings

## Future Improvements

- [ ] Add more granular code chunking for RAG
- [ ] Implement hybrid search (semantic + keyword)
- [ ] Add few-shot examples to prompts
- [ ] Deploy dashboard to Streamlit Cloud
- [ ] Add CI/CD for automated analysis updates

## License

MIT License
