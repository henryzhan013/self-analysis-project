"""
GitHub Self-Analysis Dashboard

A Streamlit dashboard showcasing LLM-powered analysis of GitHub activity.
"""

import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# Page config
st.set_page_config(
    page_title="GitHub Self-Analysis",
    page_icon="📊",
    layout="wide"
)

# Paths
DATA = Path("data")
OUTPUTS = Path("outputs")

# Load data
@st.cache_data
def load_data():
    repos = pd.read_csv(DATA / "repos.csv")
    commits = pd.read_csv(DATA / "commits.csv")
    languages = pd.read_csv(DATA / "languages.csv")
    readmes = pd.read_csv(DATA / "readmes.csv")

    commits["date"] = pd.to_datetime(commits["date"]).dt.tz_convert("America/Chicago")
    commits["hour"] = commits["date"].dt.hour
    commits["day_of_week"] = commits["date"].dt.dayofweek

    return repos, commits, languages, readmes

@st.cache_data
def load_llm_results():
    results = []

    # Load commit labels from all model-specific files
    for labels_path in OUTPUTS.glob("commit_labels__*.jsonl"):
        with open(labels_path) as f:
            for line in f:
                results.append({"source": "commit_labels", **json.loads(line)})

    # Load other analyses
    analyses_path = OUTPUTS / "llm_analyses.jsonl"
    if analyses_path.exists():
        with open(analyses_path) as f:
            for line in f:
                results.append({"source": "analyses", **json.loads(line)})

    return results

repos, commits, languages, readmes = load_data()
llm_results = load_llm_results()

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Overview", "Commit Analysis", "LLM Analyses", "Model Comparison", "RAG Demo"]
)

# ============================================================================
# OVERVIEW PAGE
# ============================================================================
if page == "Overview":
    st.title("GitHub Self-Analysis Dashboard")
    st.markdown("*Analyzing my GitHub activity using local LLMs*")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Repositories", len(repos))
    with col2:
        st.metric("Total Commits", len(commits))
    with col3:
        total_kb = repos["size_kb"].sum()
        st.metric("Total Code", f"{total_kb/1024:.1f} MB")
    with col4:
        st.metric("Languages", languages["language"].nunique())

    st.divider()

    # Language distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Language Distribution")
        lang_totals = languages.groupby("language")["bytes"].sum().sort_values(ascending=False)
        lang_totals = lang_totals.drop("Jupyter Notebook", errors="ignore")

        fig = px.pie(
            values=lang_totals.head(6).values,
            names=lang_totals.head(6).index,
            hole=0.4
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Repository Overview")
        repo_display = repos[["repo_name", "description", "language", "size_kb", "is_fork"]].copy()
        repo_display["size_kb"] = repo_display["size_kb"].apply(lambda x: f"{x:,}")
        st.dataframe(repo_display, use_container_width=True, hide_index=True)

    # Activity timeline
    st.subheader("Commit Activity Timeline")
    daily = commits.groupby(commits["date"].dt.date).size().reset_index()
    daily.columns = ["date", "commits"]

    fig = px.bar(daily, x="date", y="commits", color_discrete_sequence=["#3498db"])
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Commits",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# COMMIT ANALYSIS PAGE
# ============================================================================
elif page == "Commit Analysis":
    st.title("Commit Analysis")

    # Activity heatmap
    st.subheader("Activity Heatmap (Central Time)")

    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_data = commits.groupby(["day_of_week", "hour"]).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.reindex(columns=range(24), fill_value=0)
    heatmap_data = heatmap_data.reindex(range(7), fill_value=0)

    fig = px.imshow(
        heatmap_data.values,
        labels=dict(x="Hour", y="Day", color="Commits"),
        x=[f"{h}:00" for h in range(24)],
        y=day_names,
        color_continuous_scale="YlOrRd"
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    # Commit messages
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Message Length Distribution")
        commits["msg_length"] = commits["message"].fillna("").str.len()
        fig = px.histogram(commits, x="msg_length", nbins=20, color_discrete_sequence=["#9b59b6"])
        fig.update_layout(xaxis_title="Characters", yaxis_title="Count", height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Commits by Repository")
        repo_counts = commits.groupby("repo_name").size().sort_values(ascending=True)
        fig = px.bar(
            x=repo_counts.values,
            y=repo_counts.index,
            orientation='h',
            color_discrete_sequence=["#2ecc71"]
        )
        fig.update_layout(xaxis_title="Commits", yaxis_title="", height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Recent commits table
    st.subheader("Recent Commits")
    recent = commits.sort_values("date", ascending=False).head(10)[["date", "repo_name", "message"]]
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d %H:%M")
    st.dataframe(recent, use_container_width=True, hide_index=True)

# ============================================================================
# LLM ANALYSES PAGE
# ============================================================================
elif page == "LLM Analyses":
    st.title("LLM-Powered Analyses")

    analysis_type = st.selectbox(
        "Select Analysis",
        ["Commit Sentiment", "Skill Extraction", "README Quality", "Topic Clustering", "Recommendations"]
    )

    if analysis_type == "Commit Sentiment":
        st.subheader("Commit Sentiment Analysis")
        st.markdown("*Classifying commit messages by type and tone using local LLMs*")

        # Get commit label results
        commit_results = [r for r in llm_results if r["source"] == "commit_labels" and r.get("parsed")]

        if commit_results:
            df = pd.DataFrame(commit_results)

            # Extract type and tone
            def safe_get(parsed, key):
                if not isinstance(parsed, dict):
                    return None
                val = parsed.get(key)
                if isinstance(val, list):
                    return val[0] if val else None
                return val

            df["type"] = df["parsed"].apply(lambda x: safe_get(x, "type"))
            df["tone"] = df["parsed"].apply(lambda x: safe_get(x, "tone"))

            col1, col2 = st.columns(2)

            with col1:
                type_counts = df["type"].value_counts()
                fig = px.pie(values=type_counts.values, names=type_counts.index, title="Commit Types")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                tone_counts = df["tone"].value_counts()
                fig = px.pie(values=tone_counts.values, names=tone_counts.index, title="Commit Tones")
                st.plotly_chart(fig, use_container_width=True)

            # Show by model
            st.subheader("Results by Model")
            model_counts = df.groupby("model")["type"].count()
            st.write(f"Total results: {len(df)} ({', '.join([f'{m}: {c}' for m, c in model_counts.items()])})")
        else:
            st.warning("No commit sentiment data available.")

    elif analysis_type == "Skill Extraction":
        st.subheader("Skills Extracted from Repositories")

        skill_results = [r for r in llm_results if "skills__" in r.get("analysis", "") and r.get("parsed")]

        if skill_results:
            # Group by repo and show best result (llama3.1:8b preferred)
            repos_shown = set()
            for result in sorted(skill_results, key=lambda x: (x["analysis"], x["model"] != "llama3.1:8b")):
                repo_name = result["analysis"].replace("skills__", "")
                if repo_name in repos_shown:
                    continue
                repos_shown.add(repo_name)

                parsed = result["parsed"]

                if isinstance(parsed, dict):
                    with st.expander(f"**{repo_name}** ({result['model']})", expanded=True):
                        # Handle different field names
                        languages = parsed.get("programming_languages", [])
                        domains = parsed.get("domains", [])
                        skill_level = parsed.get("skill_level", [])
                        confidence = parsed.get("confidence", "N/A")

                        if languages:
                            st.write("**Languages:**", ", ".join(languages) if isinstance(languages, list) else languages)
                        if domains:
                            st.write("**Domains:**", ", ".join(domains) if isinstance(domains, list) else domains)
                        if skill_level:
                            st.write("**Skill Level:**", ", ".join(skill_level) if isinstance(skill_level, list) else skill_level)
                        st.write(f"**Confidence:** {confidence}")
        else:
            st.warning("No skill extraction data available.")

    elif analysis_type == "README Quality":
        st.subheader("README Documentation Quality")

        readme_results = [r for r in llm_results if "readme_quality__" in r.get("analysis", "") and r.get("parsed")]

        if readme_results:
            scores = []
            for r in readme_results:
                if isinstance(r["parsed"], dict) and "overall_score" in r["parsed"]:
                    scores.append({
                        "repo": r["analysis"].replace("readme_quality__", ""),
                        "model": r["model"],
                        "score": r["parsed"]["overall_score"]
                    })

            if scores:
                scores_df = pd.DataFrame(scores)

                fig = px.bar(
                    scores_df,
                    x="repo",
                    y="score",
                    color="model",
                    barmode="group",
                    title="README Quality Scores by Model"
                )
                fig.add_hline(y=5, line_dash="dash", line_color="red", annotation_text="Passing (5/10)")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No README quality data available.")

    elif analysis_type == "Topic Clustering":
        st.subheader("Project Topic Clustering")
        st.markdown("*Grouping repositories into thematic clusters*")

        cluster_results = [r for r in llm_results if r.get("analysis") == "topic_clustering" and r.get("parsed")]

        if cluster_results:
            # Show results from each model in tabs
            tabs = st.tabs([r["model"] for r in cluster_results])

            for tab, result in zip(tabs, cluster_results):
                with tab:
                    parsed = result["parsed"]
                    clusters = []

                    if isinstance(parsed, dict):
                        clusters = parsed.get("clusters", [])
                    elif isinstance(parsed, list):
                        clusters = parsed

                    for cluster in clusters:
                        if isinstance(cluster, dict):
                            theme = cluster.get("cluster_name") or cluster.get("theme", "Unknown")
                            repos_list = cluster.get("repos", [])
                            desc = cluster.get("description", "")

                            st.markdown(f"### {theme}")
                            st.write(f"**Repositories:** {', '.join(repos_list)}")
                            if desc:
                                st.write(f"*{desc}*")
                            st.divider()
        else:
            st.warning("No topic clustering data available.")

    elif analysis_type == "Recommendations":
        st.subheader("What Should You Build Next?")

        rec_results = [r for r in llm_results if r.get("analysis") == "recommendations" and r.get("parsed")]

        if rec_results:
            for result in rec_results:
                with st.expander(f"**{result['model']}** recommendations", expanded=True):
                    parsed = result["parsed"]
                    if isinstance(parsed, dict):
                        # Handle different field names
                        recs = parsed.get("recommendations") or parsed.get("next_project_ideas") or parsed.get("projects", [])
                        for i, rec in enumerate(recs[:3], 1):
                            if isinstance(rec, dict):
                                name = rec.get("name") or rec.get("title", "Project")
                                st.markdown(f"**{i}. {name}**")
                                desc = rec.get("description", "")
                                why = rec.get("why") or rec.get("why_good_fit", "")
                                if desc:
                                    st.write(desc)
                                if why:
                                    st.caption(f"Why: {why}")
                            elif isinstance(rec, str):
                                st.markdown(f"**{i}.** {rec}")
        else:
            st.warning("No recommendations data available.")

# ============================================================================
# MODEL COMPARISON PAGE
# ============================================================================
elif page == "Model Comparison":
    st.title("LLM Model Comparison")
    st.markdown("*Comparing phi3:mini, llama3.1:8b, and mistral:7b*")

    # Optimization findings
    st.subheader("Optimization Findings")

    opt_path = OUTPUTS / "optimization_results.json"
    if opt_path.exists():
        with open(opt_path) as f:
            opt_results = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Temperature Optimization**")
            temp_data = opt_results.get("temperature_experiments", [])
            if temp_data:
                temp_df = pd.DataFrame(temp_data)
                temp_df["success_pct"] = temp_df["success_rate"] * 100

                # Use grouped bar chart instead of lines to avoid overlap
                fig = px.bar(
                    temp_df,
                    x="temperature",
                    y="success_pct",
                    color="model",
                    barmode="group",
                    title="JSON Parse Success by Temperature"
                )
                fig.update_yaxes(title="Success Rate (%)", range=[0, 105])
                fig.update_xaxes(title="Temperature")
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Prompt Length Impact**")
            prompt_data = opt_results.get("prompt_length", [])
            if prompt_data:
                prompt_df = pd.DataFrame(prompt_data)
                fig = px.bar(prompt_df, x="prompt_type", y="latency_s",
                            title="Latency by Prompt Length",
                            color_discrete_sequence=["#2ecc71"])
                fig.update_xaxes(title="")
                fig.update_yaxes(title="Latency (s)")
                st.plotly_chart(fig, use_container_width=True)

        # Recommendations box
        st.info("""
        **Optimization Recommendations:**
        - **Temperature**: 0.3 (reliable JSON output while allowing variation)
        - **Best Model**: llama3.1:8b (100% success rate, good speed)
        - **Fastest Model**: mistral:7b (2.1s average)
        - **Prompt Length**: Medium-length prompts perform best
        """)

    st.divider()

    # Latency comparison
    st.subheader("Response Latency")

    latency_data = []
    for r in llm_results:
        if "latency_s" in r:
            latency_data.append({
                "model": r["model"],
                "latency_s": r["latency_s"],
                "source": r.get("source", "unknown")
            })

    if latency_data:
        lat_df = pd.DataFrame(latency_data)

        fig = px.box(lat_df, x="model", y="latency_s", color="model",
                     title="Response Latency by Model")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Stats table
        stats = lat_df.groupby("model")["latency_s"].agg(["mean", "median", "std", "count"])
        stats.columns = ["Mean (s)", "Median (s)", "Std Dev", "Samples"]
        st.dataframe(stats.round(2), use_container_width=True)

    # Parse success rate
    st.subheader("JSON Parse Success Rate")

    success_data = []
    for r in llm_results:
        success_data.append({
            "model": r.get("model", "unknown"),
            "success": 1 if r.get("parsed") is not None else 0
        })

    if success_data:
        success_df = pd.DataFrame(success_data)
        rates = success_df.groupby("model")["success"].mean() * 100

        fig = px.bar(x=rates.index, y=rates.values, color=rates.index,
                     title="Parse Success Rate (%)")
        fig.update_layout(height=350, showlegend=False, yaxis_range=[0, 100])
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RAG DEMO PAGE
# ============================================================================
elif page == "RAG Demo":
    st.title("RAG Code Assistant")
    st.markdown("*Ask questions about your repositories using retrieval-augmented generation*")

    # Check if RAG index exists
    rag_index_path = OUTPUTS / "rag_index"

    if not rag_index_path.exists():
        st.warning("RAG index not found. Please run notebook 06 to build the index first.")
    else:
        st.info("RAG system uses `nomic-embed-text` for embeddings and `llama3.1:8b` for generation.")
        st.warning("Note: RAG requires Ollama to be running locally.")

        # Load RAG system
        @st.cache_resource
        def load_rag():
            from src.rag.code_rag import CodeRAG
            rag = CodeRAG()
            rag.load(rag_index_path)
            return rag

        try:
            rag = load_rag()

            st.success(f"Loaded RAG index with {len(rag.vector_store.documents)} chunks")

            # Query input
            query = st.text_input("Ask a question about your code:", placeholder="e.g., What programming languages do I use?")

            if query:
                with st.spinner("Searching and generating answer..."):
                    result = rag.query(query)

                st.subheader("Answer")
                st.write(result["answer"])

                st.subheader("Sources")
                for source in result["sources"]:
                    st.write(f"- **{source['repo']}** ({source['type']}) - similarity: {source['similarity']:.3f}")

                st.caption(f"Latency: {result['latency_s']}s")

            # Example queries
            st.subheader("Example Queries")
            examples = [
                "What is the polars project about?",
                "What programming languages do I know?",
                "Which project would be most impressive to show an employer?",
                "What kind of commits have I been making?"
            ]

            cols = st.columns(2)
            for i, ex in enumerate(examples):
                with cols[i % 2]:
                    if st.button(ex, key=f"ex_{i}"):
                        st.session_state["query"] = ex
                        st.rerun()

        except Exception as e:
            st.error(f"Error loading RAG system: {e}")
            st.code(str(e))

# Footer
st.sidebar.divider()
st.sidebar.caption("Built with Streamlit + Local LLMs (Ollama)")
st.sidebar.caption("Models: phi3:mini, llama3.1:8b, mistral:7b")
