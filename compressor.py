from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# Load SBERT model
_model = SentenceTransformer("all-MiniLM-L6-v2")

# Simple regex-based sentence tokenizer (avoids NLTK dependency)
def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

# SBERT semantic compression with representative sentence per cluster
def semantic_compress(text, threshold=0.75):
    sentences = simple_sent_tokenize(text)
    if len(sentences) <= 1:
        return text

    # Create embeddings
    embeddings = _model.encode(sentences)

    # Similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    visited = [False] * len(sentences)
    compressed = []

    for i in range(len(sentences)):
        if visited[i]:
            continue

        # Create a cluster for this sentence
        cluster = [i]
        visited[i] = True

        for j in range(i + 1, len(sentences)):
            if sim_matrix[i][j] > threshold:
                cluster.append(j)
                visited[j] = True

        # Find representative sentence for this cluster
        cluster_embeddings = embeddings[cluster]
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find the sentence closest to the centroid
        distances = cosine_similarity([centroid], cluster_embeddings)[0]
        best_idx = cluster[np.argmax(distances)]
        compressed.append(sentences[best_idx])

    return "\n".join(compressed)
