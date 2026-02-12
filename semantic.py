# semantic.py
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download NLTK punkt if not already installed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Load SBERT model (small, efficient)
_model = SentenceTransformer("all-MiniLM-L6-v2")

def split_sentences(text: str) -> List[str]:
    """Split text into sentences using NLTK."""
    return [s.strip() for s in sent_tokenize(text) if s.strip()]

def semantic_compress(
    text: str,
    threshold: float = 0.85
) -> Tuple[str, np.ndarray, List[str], List[str]]:
    """
    Compress text by removing semantically similar (redundant) sentences.

    Returns:
      compressed_text: str
      similarity_matrix: np.ndarray
      kept_sentences: list[str]
      removed_sentences: list[str]
    """
    sentences = split_sentences(text)
    if not sentences:
        return "", np.zeros((0, 0)), [], []

    embeddings = _model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    sim_matrix = cosine_similarity(embeddings)

    visited = [False] * len(sentences)
    kept, removed = [], []

    for i in range(len(sentences)):
        if visited[i]:
            continue
        kept.append(sentences[i])
        for j in range(i + 1, len(sentences)):
            if sim_matrix[i, j] >= threshold:
                visited[j] = True
                removed.append(sentences[j])

    compressed_text = "\n".join(kept)
    return compressed_text, sim_matrix, kept, removed

def byte_size(text: str, encoding: str = "utf-8") -> int:
    """Return text size in bytes."""
    return len(text.encode(encoding))
