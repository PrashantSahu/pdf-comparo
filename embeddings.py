"""
Embedding utilities for document and logo embeddings.
"""

from typing import Optional

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from pdf_extraction import chunk_text, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

# Try to import FAISS, but fall back to numpy if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# Default models
DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_CLIP_MODEL = "clip-ViT-B-32"


def embed_document(
    text: str,
    model: SentenceTransformer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> Optional[np.ndarray]:
    """
    Create embedding for a document using chunking and mean pooling.

    Args:
        text: Document text.
        model: SentenceTransformer model for encoding.
        chunk_size: Number of characters per chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        Document embedding as numpy array, or None if text is empty.
    """
    if not text.strip():
        return None

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        return None

    # Embed all chunks
    chunk_embeddings = model.encode(chunks, show_progress_bar=False)

    # Mean pooling over chunks
    document_embedding = np.mean(chunk_embeddings, axis=0)

    # Normalize for cosine similarity
    document_embedding = document_embedding / np.linalg.norm(document_embedding)

    return document_embedding


def embed_logos(
    logos: list[Image.Image],
    clip_model: SentenceTransformer,
) -> Optional[np.ndarray]:
    """
    Create embedding for logos using CLIP.

    Args:
        logos: List of PIL Image objects.
        clip_model: CLIP model for encoding images.

    Returns:
        Combined logo embedding as numpy array, or None if no logos.
    """
    if not logos:
        return None

    try:
        # Encode all logos
        logo_embeddings = clip_model.encode(logos, show_progress_bar=False)

        # Mean pooling if multiple logos
        combined_embedding = np.mean(logo_embeddings, axis=0)

        # Normalize
        combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

        return combined_embedding

    except Exception as e:
        print(f"Error embedding logos: {e}")
        return None


def compute_similarities_numpy(
    local_embeddings: np.ndarray,
    remote_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute similarities using numpy (cosine similarity for normalized vectors).

    Args:
        local_embeddings: Embeddings for local forms.
        remote_embeddings: Embeddings for remote forms.
        top_k: Number of top matches to return.

    Returns:
        Tuple of (similarities, indices) arrays.
    """
    # Compute dot product (equivalent to cosine similarity for normalized vectors)
    similarity_matrix = np.dot(local_embeddings, remote_embeddings.T)

    # Get top-k indices and similarities for each local form
    k = min(top_k, remote_embeddings.shape[0])
    indices = np.argsort(-similarity_matrix, axis=1)[:, :k]
    similarities = np.array([
        similarity_matrix[i, indices[i]] for i in range(len(local_embeddings))
    ])

    return similarities, indices


def compute_similarities_faiss(
    local_embeddings: np.ndarray,
    remote_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute similarities using FAISS (faster for large datasets).

    Args:
        local_embeddings: Embeddings for local forms.
        remote_embeddings: Embeddings for remote forms.
        top_k: Number of top matches to return.

    Returns:
        Tuple of (similarities, indices) arrays.
    """
    if not FAISS_AVAILABLE:
        raise ImportError("FAISS is not available. Install with: pip install faiss-cpu")

    dimension = remote_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(remote_embeddings)
    similarities, indices = index.search(local_embeddings, top_k)

    return similarities, indices

