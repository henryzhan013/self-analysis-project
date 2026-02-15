# Prompt Engineering Documentation

This document describes all prompts used in the GitHub Self-Analysis project for LLM-powered analyses.

## Overview

We use **3 local LLM models** via Ollama:
- `phi3:mini` - Microsoft's compact model (3.8B parameters)
- `llama3.1:8b` - Meta's latest Llama model (8B parameters)
- `mistral:7b` - Mistral AI's efficient model (7B parameters)

All prompts follow a **system/user message pattern** and request **structured JSON output** for reliable parsing.

---

## Analysis 1: Commit Sentiment Analysis

**Purpose:** Classify commit messages by type (feature, bugfix, refactor, etc.) and tone (urgent, casual, technical, etc.)

### System Prompt
```
You are a commit message analyzer. Given a commit message, classify it and return JSON.

Return ONLY valid JSON with these fields:
- type: one of [feature, bugfix, refactor, docs, test, chore, style, perf]
- tone: one of [urgent, casual, technical, descriptive, minimal]
- confidence: 1-10 rating of your classification confidence

Example output:
{"type": "bugfix", "tone": "urgent", "confidence": 8}
```

### User Prompt Template
```
Analyze this commit message:

"{{message}}"
```

### Design Rationale
- **Constrained vocabulary**: Limits `type` and `tone` to predefined categories for consistent analysis
- **Confidence score**: Allows filtering low-confidence predictions
- **Example output**: Demonstrates exact JSON format expected
- **Minimal instructions**: Keeps prompt short to reduce latency

---

## Analysis 2: Skill Extraction

**Purpose:** Extract technical skills demonstrated by each repository based on description, language, and README content.

### System Prompt
```
You are a technical recruiter analyzing GitHub repositories to identify skills.

Analyze the repository and return JSON with:
- skills: array of specific technical skills demonstrated (max 10)
- complexity: 1-10 rating of project complexity
- summary: one sentence describing what this project demonstrates

Return ONLY valid JSON.
```

### User Prompt Template
```
Analyze this repository:

Repository: {{repo_name}}
Description: {{description}}
Primary Language: {{language}}

README:
{{readme}}
```

### Design Rationale
- **Role assignment**: "technical recruiter" frames the analysis from an employer's perspective
- **Bounded output**: Limits skills to 10 items to prevent verbose responses
- **Multiple data sources**: Combines repo metadata with README for richer context
- **Practical focus**: Complexity rating helps prioritize which projects to highlight

---

## Analysis 3: README Documentation Quality

**Purpose:** Evaluate the quality of repository documentation.

### System Prompt
```
You are a documentation quality assessor. Evaluate the README and return JSON with:

- overall_score: 1-10 rating
- has_description: boolean - does it explain what the project does?
- has_installation: boolean - are there setup instructions?
- has_usage: boolean - are there usage examples?
- has_contributing: boolean - are there contribution guidelines?
- suggestions: array of 1-3 specific improvements

Return ONLY valid JSON.
```

### User Prompt Template
```
Evaluate this README:

Repository: {{repo_name}}

README Content:
{{readme}}
```

### Design Rationale
- **Checklist approach**: Boolean flags for common README sections provide actionable insights
- **Numeric score**: Overall score enables cross-repo comparison
- **Improvement suggestions**: Makes the analysis actionable, not just evaluative
- **Bounded suggestions**: Limits to 3 to keep output focused

---

## Analysis 4: Topic Clustering

**Purpose:** Group repositories into thematic clusters to identify focus areas.

### System Prompt
```
You are a portfolio analyst. Given a list of repositories, group them into 2-4 thematic clusters.

Return JSON with:
- clusters: array of cluster objects, each with:
  - theme: short name for the cluster (2-4 words)
  - repos: array of repository names in this cluster
  - description: one sentence explaining the theme

Return ONLY valid JSON.
```

### User Prompt Template
```
Group these repositories into thematic clusters:

{{repos}}
```

### Design Rationale
- **Flexible clustering**: 2-4 clusters adapts to portfolio size
- **Theme naming**: Short names make clusters easy to reference
- **Portfolio perspective**: Frames analysis for career/professional context
- **Comprehensive input**: Includes repo descriptions, languages, and README previews

---

## Analysis 5: Project Recommendations

**Purpose:** Suggest what projects to build next based on current portfolio and skills.

### System Prompt
```
You are a career advisor for software developers. Based on the developer's current projects and skills, suggest what they should build next.

Return JSON with:
- recommendations: array of 3 project ideas, each with:
  - name: short project name
  - description: 2-3 sentences explaining the project
  - skills_developed: array of new skills this would demonstrate
  - why: one sentence on why this complements their portfolio

Return ONLY valid JSON.
```

### User Prompt Template
```
Based on this developer's profile, what should they build next?

Current Repositories:
{{repos}}

Top Languages: {{languages}}

Recent Activity:
{{activity}}
```

### Design Rationale
- **Career focus**: Advisor role emphasizes professional development
- **Actionable output**: Specific project ideas, not vague suggestions
- **Skill gaps**: `skills_developed` identifies growth opportunities
- **Portfolio fit**: `why` field explains strategic value of each suggestion

---

## RAG System Prompts

### Query Prompt Template
```
Based on the following context about GitHub repositories, answer the question.

CONTEXT:
{{context}}

QUESTION: {{question}}

Provide a clear, concise answer based only on the information in the context.
If the context doesn't contain enough information to answer the question, say so.
```

### Design Rationale
- **Grounded responses**: Explicitly instructs model to use only provided context
- **Honesty**: Asks model to acknowledge when information is insufficient
- **Flexible questions**: No constraints on question format

---

## Prompt Engineering Best Practices Used

### 1. Structured Output
All prompts request JSON output with specific field names. This enables:
- Reliable parsing with `json.loads()`
- Consistent data structures across models
- Easy aggregation and comparison

### 2. Role Assignment
Each prompt assigns a specific role (analyzer, recruiter, assessor, advisor). This:
- Focuses the model's perspective
- Improves response quality
- Reduces irrelevant information

### 3. Bounded Outputs
Prompts specify limits (e.g., "max 10 skills", "2-4 clusters", "1-3 suggestions"). This:
- Prevents verbose responses
- Reduces token usage and latency
- Ensures comparable outputs across runs

### 4. Example Outputs
Where helpful, prompts include example JSON. This:
- Demonstrates exact format expected
- Reduces parsing errors
- Clarifies ambiguous field names

### 5. Minimal Instructions
Prompts are kept concise to:
- Reduce input token count
- Lower latency
- Avoid confusing the model with excessive constraints

---

## Model Comparison Findings

| Metric | phi3:mini | llama3.1:8b | mistral:7b |
|--------|-----------|-------------|------------|
| Avg Latency | ~3-5s | ~8-12s | ~5-8s |
| Parse Success | ~85% | ~95% | ~90% |
| Output Quality | Good | Best | Good |

**Observations:**
- `llama3.1:8b` produces the most reliable JSON and highest quality analysis
- `phi3:mini` is fastest but occasionally produces malformed JSON
- `mistral:7b` balances speed and quality well

---

## Optimization Experiments

We ran systematic experiments to optimize LLM parameters:

### Temperature Optimization

| Temperature | phi3:mini | llama3.1:8b | mistral:7b |
|-------------|-----------|-------------|------------|
| 0.1 | 100% success | 100% success | 100% success |
| 0.3 | 100% success | 100% success | 100% success |
| 0.5 | 67% success | 100% success | 100% success |
| 0.7 | 67% success | 100% success | 100% success |

**Finding:** Lower temperatures (0.1-0.3) produce more reliable structured JSON output. We selected **temperature=0.3** as optimal - reliable parsing while allowing some variation.

### Model Speed vs Quality Trade-off

| Model | Avg Latency | Success Rate | Recommendation |
|-------|-------------|--------------|----------------|
| phi3:mini | ~4s | ~85% | Use for speed-critical, simple tasks |
| llama3.1:8b | ~2.5s | 100% | **Best overall choice** |
| mistral:7b | ~2.1s | 100% | Fastest with high quality |

**Finding:** mistral:7b offers the best speed, but llama3.1:8b provides most consistent quality. For production, we recommend llama3.1:8b.

### Prompt Length Impact

| Prompt Type | Length | Latency |
|-------------|--------|---------|
| Minimal | 17 chars | 4.06s |
| Short | 63 chars | 1.78s |
| Medium | 115 chars | 1.75s |
| Long | 298 chars | 2.12s |

**Finding:** Very short prompts actually increase latency (model struggles with ambiguity). Medium-length prompts with clear instructions are optimal.

### Optimization Decisions Applied

1. **Temperature = 0.3** - Balances reliability and variation
2. **Seed = 42** - Ensures reproducibility
3. **llama3.1:8b as default** - Best quality for structured outputs
4. **Medium-length prompts** - Clear instructions without verbosity

---

## Future Improvements

1. **Few-shot examples**: Add 2-3 examples per prompt for more consistent outputs
2. **Chain-of-thought**: For complex analyses, ask model to reason before answering
3. **Output validation**: Implement JSON schema validation with retry on failure
4. **Prompt versioning**: Track prompt versions to measure improvement over time
