"""
Simple RAG (Retrieval-Augmented Generation) system for querying your own code.

This system:
1. Chunks and embeds README content and repo metadata
2. Stores embeddings in a simple numpy-based vector store
3. Retrieves relevant context for natural language queries
4. Generates answers using a local LLM
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple

from src.llm.ollama_client import OllamaClient


class SimpleVectorStore:
    """Simple in-memory vector store using numpy."""

    def __init__(self):
        self.embeddings: np.ndarray = None
        self.documents: List[Dict] = []

    def add(self, embedding: List[float], document: Dict):
        """Add a document with its embedding."""
        emb_array = np.array(embedding).reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = emb_array
        else:
            self.embeddings = np.vstack([self.embeddings, emb_array])
        self.documents.append(document)

    def search(self, query_embedding: List[float], top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Search for most similar documents using cosine similarity."""
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query = np.array(query_embedding)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query)
        similarities = np.dot(self.embeddings, query) / (norms + 1e-10)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.documents[idx], float(similarities[idx])))

        return results

    def save(self, path: Path):
        """Save the vector store to disk."""
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f, indent=2)

    def load(self, path: Path):
        """Load the vector store from disk."""
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "documents.json", "r") as f:
            self.documents = json.load(f)


class CodeRAG:
    """RAG system for querying code repositories."""

    def __init__(
        self,
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434"
    ):
        self.client = OllamaClient(base_url=base_url)
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.vector_store = SimpleVectorStore()

    def _embed(self, text: str) -> List[float]:
        """Embed text using the shared OllamaClient."""
        result = self.client.embed(self.embedding_model, text)
        return result["embedding"]

    def index_repositories(self, repos_df, readmes_df, languages_df, commits_df):
        """Index repository data for retrieval."""
        print("Indexing repositories...")

        for _, repo in repos_df.iterrows():
            repo_name = repo["repo_name"]

            # Get README
            readme_row = readmes_df[readmes_df["repo_name"] == repo_name]
            readme_content = readme_row["readme_content"].iloc[0] if len(readme_row) > 0 else ""

            # Get languages
            repo_langs = languages_df[languages_df["repo_name"] == repo_name]
            lang_list = repo_langs["language"].tolist() if len(repo_langs) > 0 else []

            # Get commits
            repo_commits = commits_df[commits_df["repo_name"] == repo_name]
            commit_messages = repo_commits["message"].tolist() if len(repo_commits) > 0 else []

            # Create document chunks
            chunks = self._create_chunks(repo, readme_content, lang_list, commit_messages)

            # Embed and store each chunk
            for chunk in chunks:
                print(f"  Embedding: {chunk['type']} for {repo_name}...")
                embedding = self._embed(chunk["content"])
                self.vector_store.add(embedding, chunk)

        print(f"Indexed {len(self.vector_store.documents)} chunks from {len(repos_df)} repositories.")

    def _create_chunks(self, repo, readme_content: str, languages: List[str], commits: List[str]) -> List[Dict]:
        """Create document chunks from repository data."""
        chunks = []
        repo_name = repo["repo_name"]

        # Chunk 1: Repository overview
        overview = f"""Repository: {repo_name}
Description: {repo['description'] or 'No description'}
Primary Language: {repo['language'] or 'Unknown'}
All Languages: {', '.join(languages) if languages else 'Unknown'}
Created: {repo['created_at']}
Size: {repo['size_kb']} KB
Is Fork: {repo['is_fork']}"""

        chunks.append({
            "type": "overview",
            "repo_name": repo_name,
            "content": overview
        })

        # Chunk 2: README content (truncated if too long)
        if readme_content and len(readme_content.strip()) > 0:
            readme_text = readme_content[:3000] if len(readme_content) > 3000 else readme_content
            chunks.append({
                "type": "readme",
                "repo_name": repo_name,
                "content": f"README for {repo_name}:\n{readme_text}"
            })

        # Chunk 3: Recent commits
        if commits:
            commit_text = "\n".join([f"- {msg}" for msg in commits[:10]])
            chunks.append({
                "type": "commits",
                "repo_name": repo_name,
                "content": f"Recent commits for {repo_name}:\n{commit_text}"
            })

        return chunks

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query the RAG system with a natural language question."""
        # Embed the question
        query_embedding = self._embed(question)

        # Retrieve relevant chunks
        results = self.vector_store.search(query_embedding, top_k=top_k)

        # Build context from retrieved chunks
        context_parts = []
        sources = []
        for doc, score in results:
            context_parts.append(doc["content"])
            sources.append({
                "repo": doc["repo_name"],
                "type": doc["type"],
                "similarity": round(score, 3)
            })

        context = "\n\n---\n\n".join(context_parts)

        # Generate answer using LLM via shared client
        prompt = f"""Based on the following context about GitHub repositories, answer the question.

CONTEXT:
{context}

QUESTION: {question}

Provide a clear, concise answer based only on the information in the context. If the context doesn't contain enough information to answer the question, say so."""

        response = self.client.chat(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions about code repositories based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return {
            "question": question,
            "answer": response["content"],
            "sources": sources,
            "latency_s": response["latency_s"]
        }

    def save(self, path: Path):
        """Save the RAG index to disk."""
        path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save(path)
        print(f"Saved RAG index to {path}")

    def load(self, path: Path):
        """Load the RAG index from disk."""
        self.vector_store.load(path)
        print(f"Loaded RAG index from {path} ({len(self.vector_store.documents)} chunks)")
