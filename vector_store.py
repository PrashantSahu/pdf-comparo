"""
ChromaDB vector store for persistent storage of PDF form embeddings.
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from tqdm import tqdm

from pdf_extraction import (
    extract_text_from_pdf,
    extract_logos_from_pdf,
    get_pdf_files,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from embeddings import (
    embed_document,
    embed_logos,
    DEFAULT_MODEL,
    DEFAULT_CLIP_MODEL,
)


# Default paths
DEFAULT_CHROMA_PATH = "./chroma_db"
TEXT_COLLECTION_NAME = "remote_forms_text"
LOGO_COLLECTION_NAME = "remote_forms_logo"


class VectorStore:
    """ChromaDB-based vector store for PDF form embeddings."""

    def __init__(
        self,
        persist_path: str = DEFAULT_CHROMA_PATH,
        text_collection_name: str = TEXT_COLLECTION_NAME,
        logo_collection_name: str = LOGO_COLLECTION_NAME,
    ):
        """
        Initialize the vector store.

        Args:
            persist_path: Path to persist ChromaDB data.
            text_collection_name: Name of the text embeddings collection.
            logo_collection_name: Name of the logo embeddings collection.
        """
        self.persist_path = persist_path
        self.text_collection_name = text_collection_name
        self.logo_collection_name = logo_collection_name

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collections
        self.text_collection = self.client.get_or_create_collection(
            name=text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logo_collection = self.client.get_or_create_collection(
            name=logo_collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def get_collection_count(self) -> tuple[int, int]:
        """Get the number of documents in each collection."""
        return self.text_collection.count(), self.logo_collection.count()

    def is_indexed(self) -> bool:
        """Check if remote forms have been indexed."""
        text_count, logo_count = self.get_collection_count()
        return text_count > 0 and logo_count > 0

    def clear_collections(self) -> None:
        """Clear all collections (useful for re-indexing)."""
        self.client.delete_collection(self.text_collection_name)
        self.client.delete_collection(self.logo_collection_name)
        # Recreate empty collections
        self.text_collection = self.client.get_or_create_collection(
            name=self.text_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.logo_collection = self.client.get_or_create_collection(
            name=self.logo_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Collections cleared.")

    def get_indexed_ids(self) -> set[str]:
        """Get the set of already indexed document IDs."""
        existing = self.text_collection.get(include=[])
        return set(existing["ids"])

    def build_index(
        self,
        pdf_dir: str,
        text_model,
        clip_model,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        batch_size: int = 100,
    ) -> tuple[int, int]:
        """
        Build and persist embeddings for all PDFs in a directory.
        Uses upsert behavior: skips already indexed documents.

        Args:
            pdf_dir: Directory containing PDF files.
            text_model: SentenceTransformer model for text embeddings.
            clip_model: CLIP model for logo embeddings.
            chunk_size: Characters per chunk for text processing.
            chunk_overlap: Overlap between chunks.
            batch_size: Number of documents to add in each batch.

        Returns:
            Tuple of (new documents indexed, total documents skipped).
        """
        pdf_files = get_pdf_files(pdf_dir)
        print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

        if len(pdf_files) == 0:
            print("No PDF files found.")
            return 0, 0

        # Get already indexed documents
        existing_ids = self.get_indexed_ids()
        print(f"Already indexed: {len(existing_ids)} documents")

        # Process PDFs and collect embeddings
        ids = []
        text_embeddings = []
        logo_embeddings = []
        metadatas = []
        skipped = 0

        clip_dim = clip_model.get_sentence_embedding_dimension()

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            filename = pdf_path.name

            # Skip if already indexed
            if filename in existing_ids:
                skipped += 1
                continue

            # Extract and embed text
            text = extract_text_from_pdf(str(pdf_path))
            text_embedding = embed_document(text, text_model, chunk_size, chunk_overlap)

            if text_embedding is None:
                print(f"  Skipping {filename}: no text extracted")
                continue

            # Extract and embed logos
            logos = extract_logos_from_pdf(str(pdf_path))
            logo_embedding = embed_logos(logos, clip_model)
            if logo_embedding is None:
                logo_embedding = np.zeros(clip_dim)

            ids.append(filename)
            text_embeddings.append(text_embedding.tolist())
            logo_embeddings.append(logo_embedding.tolist())
            metadatas.append({
                "filename": filename,
                "text_preview": text[:200] if text else "",
            })

        # Add to collections in batches
        total = len(ids)
        if total == 0:
            print(f"\nNo new documents to index. Skipped {skipped} already indexed.")
            return 0, skipped

        print(f"\nAdding {total} new documents to ChromaDB...")

        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            batch_ids = ids[i:end]
            batch_text_emb = text_embeddings[i:end]
            batch_logo_emb = logo_embeddings[i:end]
            batch_meta = metadatas[i:end]

            self.text_collection.add(
                ids=batch_ids,
                embeddings=batch_text_emb,
                metadatas=batch_meta,
            )
            self.logo_collection.add(
                ids=batch_ids,
                embeddings=batch_logo_emb,
                metadatas=batch_meta,
            )
            print(f"  Added batch {i//batch_size + 1}: {len(batch_ids)} documents")

        print(f"\nIndexing complete. New: {total}, Skipped: {skipped}")
        return total, skipped

    def query(
        self,
        text_embeddings: np.ndarray,
        logo_embeddings: np.ndarray,
        top_k: int = 5,
        text_weight: float = 0.7,
    ) -> list[dict]:
        """
        Query the vector store for similar documents.

        Args:
            text_embeddings: Text embeddings for query documents (N x D).
            logo_embeddings: Logo embeddings for query documents (N x D).
            top_k: Number of top matches to return per query.
            text_weight: Weight for text similarity (logo_weight = 1 - text_weight).

        Returns:
            List of match results for each query document.
        """
        logo_weight = 1.0 - text_weight
        results = []

        for i in range(len(text_embeddings)):
            # Query text collection
            text_results = self.text_collection.query(
                query_embeddings=[text_embeddings[i].tolist()],
                n_results=min(top_k * 2, self.text_collection.count()),  # Get more for re-ranking
                include=["embeddings", "metadatas", "distances"],
            )

            # Query logo collection
            logo_results = self.logo_collection.query(
                query_embeddings=[logo_embeddings[i].tolist()],
                n_results=min(top_k * 2, self.logo_collection.count()),
                include=["embeddings", "metadatas", "distances"],
            )

            # Combine results and compute combined similarity
            # ChromaDB returns distances, convert to similarity (1 - distance for cosine)
            text_scores = {}
            for j, doc_id in enumerate(text_results["ids"][0]):
                # ChromaDB cosine distance = 1 - cosine_similarity
                text_sim = 1 - text_results["distances"][0][j]
                text_scores[doc_id] = text_sim

            logo_scores = {}
            for j, doc_id in enumerate(logo_results["ids"][0]):
                logo_sim = 1 - logo_results["distances"][0][j]
                logo_scores[doc_id] = logo_sim

            # Combine scores for all unique documents
            all_docs = set(text_scores.keys()) | set(logo_scores.keys())
            combined_scores = []

            for doc_id in all_docs:
                text_sim = text_scores.get(doc_id, 0.0)
                logo_sim = logo_scores.get(doc_id, 0.0)
                combined_sim = (text_weight * text_sim) + (logo_weight * logo_sim)

                combined_scores.append({
                    "remote_form": doc_id,
                    "combined_similarity": combined_sim,
                    "text_similarity": text_sim,
                    "logo_similarity": logo_sim,
                })

            # Sort by combined similarity
            combined_scores.sort(key=lambda x: x["combined_similarity"], reverse=True)
            top_matches = combined_scores[:top_k]

            results.append({
                "top_matches": top_matches,
                "best_match": top_matches[0]["remote_form"] if top_matches else None,
                "best_similarity": top_matches[0]["combined_similarity"] if top_matches else 0.0,
                "best_text_similarity": top_matches[0]["text_similarity"] if top_matches else 0.0,
                "best_logo_similarity": top_matches[0]["logo_similarity"] if top_matches else 0.0,
            })

        return results

    def get_all_embeddings(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Retrieve all embeddings from the store (for compatibility with existing code).

        Returns:
            Tuple of (text_embeddings, logo_embeddings, filenames).
        """
        text_data = self.text_collection.get(include=["embeddings"])
        logo_data = self.logo_collection.get(include=["embeddings"])

        text_emb = np.array(text_data["embeddings"]).astype("float32")
        logo_emb = np.array(logo_data["embeddings"]).astype("float32")
        filenames = text_data["ids"]

        return text_emb, logo_emb, filenames

