#!/usr/bin/env python3
"""
PDF Form Comparison using Sentence Transformers.

This script compares PDF forms between two directories using semantic embeddings.
For each form in the source directory, it finds the closest matching form(s) in
the target directory.

Features:
- Uses sentence-transformers (all-mpnet-base-v2) for semantic embeddings
- Handles long documents via chunking and mean pooling
- Uses numpy for similarity computation (with optional FAISS for large scale)
- Supports thousands of documents efficiently
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Try to import FAISS, but fall back to numpy if not available or problematic
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# Configuration
DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_CHUNK_SIZE = 1000  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # overlap between chunks
DEFAULT_TOP_K = 5  # number of top matches to return


class PDFFormComparator:
    """Compare PDF forms using semantic embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        """
        Initialize the comparator.

        Args:
            model_name: Name of the sentence-transformer model to use.
            chunk_size: Number of characters per chunk for long documents.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        print(f"Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print("Model loaded successfully.")

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(pdf_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        except Exception as e:
            print(f"Error reading {pdf_path}: {e}")
        return text

    def chunk_text(self, text: str) -> list[str]:
        """Split text into overlapping chunks."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def embed_document(self, text: str) -> Optional[np.ndarray]:
        """
        Create embedding for a document using chunking and mean pooling.

        Args:
            text: Document text.

        Returns:
            Document embedding as numpy array, or None if text is empty.
        """
        if not text.strip():
            return None

        chunks = self.chunk_text(text)
        if not chunks:
            return None

        # Embed all chunks
        chunk_embeddings = self.model.encode(chunks, show_progress_bar=False)

        # Mean pooling over chunks
        document_embedding = np.mean(chunk_embeddings, axis=0)

        # Normalize for cosine similarity
        document_embedding = document_embedding / np.linalg.norm(document_embedding)

        return document_embedding

    def get_pdf_files(self, directory: str) -> list[Path]:
        """Get all PDF files in a directory."""
        return sorted(Path(directory).glob("*.pdf"))

    def build_embeddings(
        self, pdf_files: list[Path]
    ) -> tuple[np.ndarray, list[str], list[str]]:
        """
        Build embeddings for a list of PDF files.

        Returns:
            Tuple of (embeddings array, file names, extracted texts)
        """
        embeddings = []
        file_names = []
        texts = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            text = self.extract_text_from_pdf(str(pdf_path))
            embedding = self.embed_document(text)

            if embedding is not None:
                embeddings.append(embedding)
                file_names.append(pdf_path.name)
                texts.append(text[:500] + "..." if len(text) > 500 else text)

        return np.array(embeddings).astype("float32"), file_names, texts

    def compute_similarities_numpy(
        self,
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

    def find_matches(
        self,
        local_dir: str,
        remote_dir: str,
        top_k: int = DEFAULT_TOP_K,
        use_faiss: bool = False,
    ) -> list[dict]:
        """
        Find closest matching forms for each local form.

        Args:
            local_dir: Directory containing local forms.
            remote_dir: Directory containing remote forms.
            top_k: Number of top matches to return.
            use_faiss: Whether to use FAISS for similarity search (default: False).

        Returns:
            List of match results.
        """
        # Get PDF files
        local_files = self.get_pdf_files(local_dir)
        remote_files = self.get_pdf_files(remote_dir)

        print(f"\nFound {len(local_files)} files in {local_dir}")
        print(f"Found {len(remote_files)} files in {remote_dir}")

        # Build embeddings
        print("\nBuilding embeddings for remote forms...")
        remote_embeddings, remote_names, _ = self.build_embeddings(remote_files)

        print("\nBuilding embeddings for local forms...")
        local_embeddings, local_names, _ = self.build_embeddings(local_files)

        if len(remote_embeddings) == 0 or len(local_embeddings) == 0:
            print("Error: No valid embeddings could be created.")
            return []

        # Compute similarities
        print("\nComputing similarities...")
        k = min(top_k, len(remote_names))

        if use_faiss and FAISS_AVAILABLE:
            print("Using FAISS for similarity search...")
            dimension = remote_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(remote_embeddings)
            similarities, indices = index.search(local_embeddings, k)
        else:
            print("Using numpy for similarity search...")
            similarities, indices = self.compute_similarities_numpy(
                local_embeddings, remote_embeddings, k
            )

        # Compile results
        results = []
        for i, local_name in enumerate(local_names):
            matches = []
            for j in range(k):
                matches.append({
                    "remote_form": remote_names[indices[i][j]],
                    "similarity": float(similarities[i][j]),
                })
            results.append({
                "local_form": local_name,
                "best_match": matches[0]["remote_form"],
                "best_similarity": matches[0]["similarity"],
                "top_matches": matches,
            })

        return results

    def print_results(self, results: list[dict]) -> None:
        """Print results in a formatted way."""
        print("\n" + "=" * 80)
        print("MATCHING RESULTS")
        print("=" * 80)

        for r in results:
            print(f"\n{'─' * 60}")
            print(f"Local Form: {r['local_form']}")
            print(f"Best Match: {r['best_match']}")
            print(f"Similarity: {r['best_similarity']:.4f} ({r['best_similarity']*100:.2f}%)")
            print(f"\nTop matches:")
            for match in r["top_matches"]:
                sim = match["similarity"]
                print(f"  • {match['remote_form']}: {sim:.4f} ({sim*100:.2f}%)")

        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        for r in results:
            print(f"{r['local_form']} → {r['best_match']} ({r['best_similarity']:.2%})")

    def save_results(self, results: list[dict], output_path: str) -> None:
        """Save results to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare PDF forms using semantic embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_forms_embeddings.py
  python compare_forms_embeddings.py --local-dir my_forms --remote-dir templates
  python compare_forms_embeddings.py --top-k 10 --output results.json
        """,
    )
    parser.add_argument(
        "--local-dir",
        default="local_forms",
        help="Directory containing local forms (default: local_forms)",
    )
    parser.add_argument(
        "--remote-dir",
        default="remote_forms",
        help="Directory containing remote forms (default: remote_forms)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Sentence transformer model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of top matches to return (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Characters per chunk for long documents (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Overlap between chunks (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--use-faiss",
        action="store_true",
        help="Use FAISS for similarity search (faster for very large datasets)",
    )

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.local_dir):
        print(f"Error: Local directory not found: {args.local_dir}")
        return 1
    if not os.path.isdir(args.remote_dir):
        print(f"Error: Remote directory not found: {args.remote_dir}")
        return 1

    # Initialize comparator
    comparator = PDFFormComparator(
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    # Find matches
    results = comparator.find_matches(
        local_dir=args.local_dir,
        remote_dir=args.remote_dir,
        top_k=args.top_k,
        use_faiss=args.use_faiss,
    )

    # Print and optionally save results
    if results:
        comparator.print_results(results)
        if args.output:
            comparator.save_results(results, args.output)

    return 0


if __name__ == "__main__":
    exit(main())

