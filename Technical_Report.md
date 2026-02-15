# Technical Report: GitHub Self-Analysis Using Local LLMs

**Author:** Data Science Intern Project
**Date:** February 2026
**Version:** 1.0

---

## Executive Summary

This project demonstrates the application of local Large Language Models (LLMs) to analyze personal GitHub activity. By combining traditional data science techniques with modern LLM-powered analysis, we extracted meaningful insights about coding patterns, technical skills, and project recommendations.

**Key Achievements:**
- Compared 3 local LLM models across 5 different analysis tasks
- Achieved 85-95% JSON parsing success rate across models
- Built a RAG system for natural language code queries
- Created an interactive Streamlit dashboard for exploration
- Generated 6+ visualizations and applied 2 ML techniques

---

## 1. Introduction

### 1.1 Project Objectives

1. Collect and analyze personal GitHub data (repositories, commits, languages)
2. Apply local LLMs to extract insights not available through traditional analysis
3. Compare multiple LLM models on structured output tasks
4. Build a retrieval-augmented generation (RAG) system for code queries
5. Create visualizations and apply machine learning techniques
6. Develop an interactive dashboard for exploring results

### 1.2 Data Overview

| Dataset | Records | Description |
|---------|---------|-------------|
| Repositories | 7 | Public repos with metadata |
| Commits | 23 | Commit messages and timestamps |
| Languages | 17 | Language breakdown per repo |
| READMEs | 5 | Documentation content |
| Events | 32 | Recent GitHub activity |

---

## 2. Methodology

### 2.1 LLM Infrastructure

We used **Ollama** to run local LLM models, ensuring data privacy and eliminating API costs.

**Models Evaluated:**
| Model | Parameters | Strengths |
|-------|------------|-----------|
| phi3:mini | 3.8B | Fast inference, good for simple tasks |
| llama3.1:8b | 8B | Best output quality, most reliable JSON |
| mistral:7b | 7B | Balanced speed and quality |

### 2.2 Prompt Engineering Strategy

All prompts followed a consistent pattern:

1. **System Message**: Define role and output format
2. **User Message**: Provide context and specific request
3. **JSON Output**: Request structured responses for reliable parsing

**Key Techniques Applied:**
- Role assignment (e.g., "You are a technical recruiter")
- Output bounding (e.g., "max 10 skills")
- Example outputs for format clarity
- Minimal instruction sets to reduce latency

### 2.3 Analysis Pipeline

```
GitHub Data → Preprocessing → LLM Analysis → JSON Parsing → Visualization
                                   ↓
                            Model Comparison
```

---

## 3. LLM Analyses

### 3.1 Analysis 1: Commit Sentiment Classification

**Objective:** Classify commit messages by type and tone.

**Categories:**
- **Type**: feature, bugfix, refactor, docs, test, chore, style, perf
- **Tone**: urgent, casual, technical, descriptive, minimal

**Results:**
- Most commits classified as "chore" or "feature" type
- Predominant tone: "minimal" (short commit messages)
- Model agreement: ~70% across all three models

### 3.2 Analysis 2: Skill Extraction

**Objective:** Identify technical skills from repository content.

**Top Skills Identified:**
1. Rust programming
2. Python data analysis
3. Competitive programming / algorithms
4. Data processing (Polars, DataFrames)
5. Web development (React, FastAPI)

**Complexity Ratings:**
- polars (fork): 9/10 - Large-scale data processing library
- roadtrip-planner: 7/10 - Full-stack application
- nfl-bdb-2024: 6/10 - Data analysis project

### 3.3 Analysis 3: README Quality Assessment

**Objective:** Evaluate documentation quality.

**Scoring Criteria:**
- Has description (what the project does)
- Has installation instructions
- Has usage examples
- Has contributing guidelines

**Results:**
| Repository | Score | Missing Elements |
|------------|-------|------------------|
| polars | 9/10 | None |
| roadtrip-planner | 7/10 | Contributing guidelines |
| nfl-bdb-2024 | 6/10 | Installation, usage |
| leetcode-solutions | 4/10 | Most sections |
| roadtripplanner | 5/10 | Usage examples |

**Average Score:** 6.2/10

### 3.4 Analysis 4: Topic Clustering

**Objective:** Group repositories into thematic clusters.

**Clusters Identified:**

| Cluster | Theme | Repositories |
|---------|-------|--------------|
| 1 | Competitive Programming | Competitive-Programming, leetcode-solutions |
| 2 | Data Science | nfl-bdb-2024, polars |
| 3 | Web Applications | roadtrip-planner, roadtripplanner |
| 4 | Miscellaneous | questions |

### 3.5 Analysis 5: Project Recommendations

**Objective:** Suggest next projects based on portfolio gaps.

**Top Recommendations:**

1. **ML Pipeline Project**
   - Build an end-to-end ML pipeline
   - Skills: MLOps, Docker, CI/CD
   - Complements data analysis experience

2. **Open Source Contribution**
   - Contribute to a major project
   - Skills: Collaboration, code review
   - Builds professional credibility

3. **API-First Application**
   - Build a public API service
   - Skills: API design, documentation
   - Demonstrates backend expertise

---

## 4. Model Comparison

### 4.1 Performance Metrics

| Metric | phi3:mini | llama3.1:8b | mistral:7b |
|--------|-----------|-------------|------------|
| Avg Latency | 4.2s | 9.8s | 6.5s |
| Median Latency | 3.8s | 8.5s | 5.9s |
| Parse Success | 85.7% | 95.2% | 90.5% |
| Total Samples | 21 | 21 | 21 |

### 4.2 Quality Assessment

**phi3:mini:**
- Fastest inference
- Occasionally produces malformed JSON
- Good for simple classification tasks

**llama3.1:8b:**
- Best output quality
- Most reliable JSON formatting
- Recommended for production use

**mistral:7b:**
- Good balance of speed and quality
- Occasionally verbose outputs
- Solid alternative to llama3.1

### 4.3 Agreement Analysis

Cross-model agreement on commit classification:
- Type agreement: 71.4%
- Tone agreement: 66.7%
- Full agreement (both): 52.4%

### 4.4 Optimization Experiments

We conducted systematic experiments to optimize LLM parameters:

**Temperature Optimization:**

| Temperature | phi3:mini | llama3.1:8b | mistral:7b |
|-------------|-----------|-------------|------------|
| 0.1 | 100% | 100% | 100% |
| 0.3 | 100% | 100% | 100% |
| 0.5 | 67% | 100% | 100% |
| 0.7 | 67% | 100% | 100% |

**Finding:** Lower temperatures (0.1-0.3) produce more reliable JSON output. Selected **0.3** as optimal.

**Prompt Length Impact:**

| Prompt Type | Length | Latency |
|-------------|--------|---------|
| Minimal | 17 chars | 4.06s |
| Short | 63 chars | 1.78s |
| Medium | 115 chars | 1.75s |
| Long | 298 chars | 2.12s |

**Finding:** Very short prompts increase latency due to ambiguity. Medium-length prompts are optimal.

**Optimization Decisions Applied:**
1. Temperature = 0.3 for structured outputs
2. Seed = 42 for reproducibility
3. llama3.1:8b as default model
4. Medium-length prompts with clear instructions

---

## 5. Traditional Data Science

### 5.1 Visualizations Created

1. **Commit Activity Timeline** - Bar chart showing commits over time
2. **Language Distribution** - Pie chart of code by language (Rust 62.6%, Python 36.7%)
3. **Activity Heatmap** - Day × hour heatmap (most active: Sunday 22:00 CT)
4. **Repository Comparison** - Commits and size by repo
5. **Message Length Analysis** - Histogram of commit message lengths
6. **Contribution Timeline** - Scatter plot of commits by repo over time

### 5.2 Machine Learning Techniques

**K-Means Clustering:**
- Clustered commits into 3 groups based on message length, time, and day
- Cluster 0: Detailed commits (avg 330 chars)
- Cluster 1: Standard commits (avg 34 chars)
- Cluster 2: Minimal commits (avg 10 chars)

**Time Series Forecasting:**
- Applied linear regression to weekly commit counts
- Trend: +0.006 commits/week (slightly increasing)
- R² Score: 0.05 (high variance, sparse data)
- Forecast: ~5 commits in next 8 weeks

---

## 6. RAG System

### 6.1 Architecture

```
Query → Embed (nomic-embed-text) → Vector Search → Context Retrieval → LLM Generation
```

**Components:**
- **Embedding Model:** nomic-embed-text (768 dimensions)
- **Vector Store:** NumPy-based cosine similarity
- **Generation Model:** llama3.1:8b

### 6.2 Indexing Strategy

Documents indexed:
- Repository overviews (7 chunks)
- README content (5 chunks)
- Commit message history (5 chunks)

**Total:** 17 chunks indexed

### 6.3 Performance

| Metric | Value |
|--------|-------|
| Avg Query Latency | 8.5s |
| Avg Similarity Score | 0.55 |
| Index Size | 17 chunks |
| Vector Dimensions | 768 |

### 6.4 Example Queries

**Q:** "What is the polars project about?"
**A:** "The polars project is an analytical query engine for DataFrames, designed to be fast and expressive. It supports Python, Rust, Node.js, R, and SQL..."

**Q:** "Which project would be most impressive to show an employer?"
**A:** "The roadtrip-planner project would be most impressive as it demonstrates full-stack development with React frontend, FastAPI backend, and ML-powered search..."

---

## 7. Dashboard

### 7.1 Features

The Streamlit dashboard provides 5 interactive pages:

1. **Overview** - Key metrics, language distribution, activity timeline
2. **Commit Analysis** - Heatmap, message statistics, recent commits
3. **LLM Analyses** - All 5 analyses with model selection
4. **Model Comparison** - Latency and success rate visualizations
5. **RAG Demo** - Interactive query interface

### 7.2 Technologies

- **Framework:** Streamlit 1.54.0
- **Visualization:** Plotly Express
- **Data:** pandas DataFrames
- **Caching:** Streamlit cache decorators

---

## 8. Conclusions

### 8.1 Key Findings

1. **LLM Viability:** Local LLMs can effectively analyze code repositories with 85-95% reliability
2. **Model Selection:** llama3.1:8b offers best quality; phi3:mini best for speed
3. **Prompt Engineering:** Structured JSON requests with bounded outputs work well
4. **RAG Utility:** Semantic search over code metadata enables useful Q&A

### 8.2 Limitations

- Small dataset (23 commits) limits statistical significance
- No actual source code analysis (only metadata and READMEs)
- RAG system uses coarse document chunking
- Single-user analysis may not generalize

### 8.3 Future Work

1. **Code-Level Analysis:** Embed actual source code for deeper insights
2. **Hybrid Search:** Combine semantic and keyword search in RAG
3. **Fine-Tuning:** Fine-tune a model on personal coding style
4. **Automation:** Set up CI/CD for continuous analysis updates
5. **Multi-User:** Extend to analyze team or organization patterns

---

## Appendices

### A. Environment

- **Python:** 3.11+
- **Ollama:** Latest
- **Key Libraries:** pandas, numpy, scikit-learn, matplotlib, plotly, streamlit

### B. File Inventory

| File | Description |
|------|-------------|
| app.py | Streamlit dashboard |
| Prompts.md | Prompt documentation |
| README.md | Project overview |
| src/rag/code_rag.py | RAG system implementation |
| src/llm/ollama_client.py | Ollama API wrapper |
| notebooks/*.ipynb | Analysis notebooks (6 total) |
| outputs/*.jsonl | LLM analysis results |
| outputs/*.png | Generated visualizations |

### C. Reproducibility

To reproduce all analyses:

```bash
# 1. Start Ollama
ollama serve

# 2. Pull models
ollama pull phi3:mini llama3.1:8b mistral:7b nomic-embed-text

# 3. Run analyses
python -m src.tasks.run_commit_labels
python -m src.tasks.run_all_analyses

# 4. Execute notebooks
jupyter nbconvert --execute notebooks/*.ipynb

# 5. Launch dashboard
streamlit run app.py
```

---

*End of Technical Report*
