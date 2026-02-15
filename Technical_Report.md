# Technical Report: GitHub Self-Analysis with Local LLMs

## 1. Local LLM Experience and Challenges

### Setup Experience
I used **Ollama** to run local LLMs on my MacBook. Setup was straightforward:
- Install Ollama (one command)
- Pull models: `ollama pull llama3.1:8b`
- Access via REST API at `localhost:11434`

### Models Used
| Model | Size | Use Case |
|-------|------|----------|
| phi3:mini | 3.8B | Fast inference, simple tasks |
| llama3.1:8b | 8B | Best quality, production use |
| mistral:7b | 7B | Balanced speed/quality |
| nomic-embed-text | 274MB | Embeddings for RAG |

### Challenges Faced

1. **JSON Parsing Reliability**
   - Problem: Models sometimes output malformed JSON or extra text
   - Solution: Lower temperature (0.3), explicit "Return ONLY valid JSON" instructions, robust parsing with fallbacks

2. **Prompt Engineering for Structured Output**
   - Problem: Models interpret prompts differently
   - Solution: Bounded outputs ("max 10 skills"), example outputs in prompts, consistent system message format

3. **Speed vs Quality Trade-off**
   - Problem: Larger models = better output but slower
   - Solution: Use phi3:mini for simple classification, llama3.1:8b for complex analysis

4. **Context Window Limits**
   - Problem: Large READMEs exceed context limits
   - Solution: Truncate to 3000 characters, chunk documents for RAG

---

## 2. Cost Analysis: Local vs Paid APIs

| Factor | Local (Ollama) | OpenAI API | Claude API |
|--------|----------------|------------|------------|
| **Setup Cost** | $0 | $0 | $0 |
| **Per-query Cost** | $0 | ~$0.002-0.06 | ~$0.003-0.08 |
| **69 analyses** | **$0** | ~$1-4 | ~$2-5 |
| **Privacy** | Data stays local | Data sent to cloud | Data sent to cloud |
| **Speed** | 2-10s/query | <1s/query | <1s/query |
| **Quality** | Good (8B models) | Excellent | Excellent |

**My Choice**: Local LLMs for this project because:
- Zero cost for experimentation and iteration
- Full data privacy (analyzing my own GitHub data)
- Educational value (understanding model behavior)
- Trade-off: Acceptable for batch processing, not real-time apps

---

## 3. Top Insights About Myself

### Coding Patterns
1. **Afternoon Coder**: Peak commit time is 4-6 PM - I'm most productive in late afternoon
2. **Minimal Documenter**: Average commit message is 10 words. I should write more descriptive commits

### Technical Profile
3. **Systems + Data**: Rust (62.6%) for performance, Python (36.7%) for data analysis - I bridge low-level and high-level programming
4. **Competitive Background**: Multiple repos for competitive programming - strong algorithmic foundations

### Portfolio Gaps
5. **Limited Web Presence**: Only 1-2 full-stack projects. Recommendation: Build more user-facing applications to demonstrate end-to-end skills

---

## 4. What I'd Do With More Time

1. **Ground truth evaluation** - manually label all commits and measure how well the LLM classifications actually match
2. **Embed actual code** - the RAG only indexes READMEs right now, indexing source files would make it way more useful
3. **Cache LLM responses** - re-running analyses is slow, storing results would speed up iteration
4. **CI/CD integration** - auto-analyze commits on push and generate weekly activity reports

