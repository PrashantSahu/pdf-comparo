"""
PDF Form Comparator class for comparing PDF forms using semantic embeddings.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from pdf_extraction import (
    extract_text_from_pdf,
    extract_logos_from_pdf,
    get_pdf_files,
    PYMUPDF_AVAILABLE,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
)
from embeddings import (
    embed_document,
    embed_logos,
    compute_similarities_numpy,
    compute_similarities_faiss,
    FAISS_AVAILABLE,
    DEFAULT_MODEL,
    DEFAULT_CLIP_MODEL,
)


DEFAULT_TOP_K = 5  # number of top matches to return
DEFAULT_LOGO_WEIGHT = 0.3  # weight for logo similarity (0.0 to 1.0)


class PDFFormComparator:
    """Compare PDF forms using semantic embeddings and logo comparison."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        clip_model_name: str = DEFAULT_CLIP_MODEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        logo_weight: float = DEFAULT_LOGO_WEIGHT,
        use_logo_comparison: bool = True,
    ):
        """
        Initialize the comparator.

        Args:
            model_name: Name of the sentence-transformer model for text.
            clip_model_name: Name of the CLIP model for logo/image comparison.
            chunk_size: Number of characters per chunk for long documents.
            chunk_overlap: Number of overlapping characters between chunks.
            logo_weight: Weight for logo similarity (0.0 to 1.0).
            use_logo_comparison: Whether to use logo/image comparison.
        """
        print(f"Loading text model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logo_weight = logo_weight
        self.use_logo_comparison = use_logo_comparison and PYMUPDF_AVAILABLE

        if self.use_logo_comparison:
            print(f"Loading CLIP model: {clip_model_name}...")
            self.clip_model = SentenceTransformer(clip_model_name)
            print("CLIP model loaded successfully.")
        elif use_logo_comparison and not PYMUPDF_AVAILABLE:
            print("Warning: pymupdf not available. Logo comparison disabled.")
            print("Install with: pip install pymupdf")

        print("Models loaded successfully.")

    def build_embeddings(
        self, pdf_files: list[Path], include_logos: bool = True
    ) -> tuple[np.ndarray, Optional[np.ndarray], list[str], list[str]]:
        """
        Build text and logo embeddings for a list of PDF files.

        Args:
            pdf_files: List of PDF file paths.
            include_logos: Whether to extract and embed logos.

        Returns:
            Tuple of (text embeddings, logo embeddings, file names, extracted texts)
        """
        text_embeddings = []
        logo_embeddings = []
        file_names = []
        texts = []

        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            text = extract_text_from_pdf(str(pdf_path))
            text_embedding = embed_document(
                text, self.model, self.chunk_size, self.chunk_overlap
            )

            if text_embedding is not None:
                text_embeddings.append(text_embedding)
                file_names.append(pdf_path.name)
                texts.append(text[:500] + "..." if len(text) > 500 else text)

                # Extract and embed logos if enabled
                if include_logos and self.use_logo_comparison:
                    logos = extract_logos_from_pdf(str(pdf_path))
                    logo_embedding = embed_logos(logos, self.clip_model)
                    if logo_embedding is not None:
                        logo_embeddings.append(logo_embedding)
                    else:
                        # Use zero vector as placeholder
                        logo_embeddings.append(
                            np.zeros(self.clip_model.get_sentence_embedding_dimension())
                        )

        text_arr = np.array(text_embeddings).astype("float32")
        logo_arr = np.array(logo_embeddings).astype("float32") if logo_embeddings else None

        return text_arr, logo_arr, file_names, texts

    def find_matches(
        self,
        local_dir: str,
        remote_dir: str,
        top_k: int = DEFAULT_TOP_K,
        use_faiss: bool = False,
    ) -> list[dict]:
        """
        Find closest matching forms for each local form.

        Combines text similarity with logo similarity (if enabled) using weighted average.

        Args:
            local_dir: Directory containing local forms.
            remote_dir: Directory containing remote forms.
            top_k: Number of top matches to return.
            use_faiss: Whether to use FAISS for similarity search.

        Returns:
            List of match results.
        """
        # Get PDF files
        local_files = get_pdf_files(local_dir)
        remote_files = get_pdf_files(remote_dir)

        print(f"\nFound {len(local_files)} files in {local_dir}")
        print(f"Found {len(remote_files)} files in {remote_dir}")

        # Build embeddings
        print("\nBuilding embeddings for remote forms...")
        remote_text_emb, remote_logo_emb, remote_names, _ = self.build_embeddings(remote_files)

        print("\nBuilding embeddings for local forms...")
        local_text_emb, local_logo_emb, local_names, _ = self.build_embeddings(local_files)

        if len(remote_text_emb) == 0 or len(local_text_emb) == 0:
            print("Error: No valid embeddings could be created.")
            return []

        # Compute text similarities
        print("\nComputing text similarities...")
        text_weight = 1.0 - self.logo_weight

        if use_faiss and FAISS_AVAILABLE:
            print("Using FAISS for similarity search...")
            text_similarities, text_indices = compute_similarities_faiss(
                local_text_emb, remote_text_emb, len(remote_names)
            )
        else:
            print("Using numpy for similarity search...")
            text_similarities, text_indices = compute_similarities_numpy(
                local_text_emb, remote_text_emb, len(remote_names)
            )

        # Compute logo similarities if available
        use_logos = (
            self.use_logo_comparison
            and local_logo_emb is not None
            and remote_logo_emb is not None
            and len(local_logo_emb) > 0
            and len(remote_logo_emb) > 0
        )

        if use_logos:
            print("Computing logo similarities...")
            logo_similarity_matrix = np.dot(local_logo_emb, remote_logo_emb.T)
        else:
            logo_similarity_matrix = None
            text_weight = 1.0  # Use only text if no logos

        # Combine similarities and rank
        print("\nRanking matches...")
        k = min(top_k, len(remote_names))
        results = []

        for i, local_name in enumerate(local_names):
            match_scores = []

            for j in range(len(remote_names)):
                text_sim = float(np.dot(local_text_emb[i], remote_text_emb[j]))

                if use_logos and logo_similarity_matrix is not None:
                    logo_sim = float(logo_similarity_matrix[i, j])
                    combined_sim = (text_weight * text_sim) + (self.logo_weight * logo_sim)
                else:
                    logo_sim = None
                    combined_sim = text_sim

                match_scores.append({
                    "remote_form": remote_names[j],
                    "combined_similarity": combined_sim,
                    "text_similarity": text_sim,
                    "logo_similarity": logo_sim,
                })

            # Sort by combined similarity
            match_scores.sort(key=lambda x: x["combined_similarity"], reverse=True)
            top_matches = match_scores[:k]

            results.append({
                "local_form": local_name,
                "best_match": top_matches[0]["remote_form"],
                "best_similarity": top_matches[0]["combined_similarity"],
                "best_text_similarity": top_matches[0]["text_similarity"],
                "best_logo_similarity": top_matches[0]["logo_similarity"],
                "top_matches": top_matches,
            })

        return results

    def print_results(self, results: list[dict]) -> None:
        """Print results in a formatted way."""
        print("\n" + "=" * 80)
        print("MATCHING RESULTS")
        if self.use_logo_comparison:
            print(f"(Text weight: {1-self.logo_weight:.0%}, Logo weight: {self.logo_weight:.0%})")
        print("=" * 80)

        for r in results:
            print(f"\n{'─' * 60}")
            print(f"Local Form: {r['local_form']}")
            print(f"Best Match: {r['best_match']}")
            print(f"Combined Similarity: {r['best_similarity']:.4f} ({r['best_similarity']*100:.2f}%)")
            if r.get('best_text_similarity') is not None:
                print(f"  Text Similarity:   {r['best_text_similarity']:.4f} ({r['best_text_similarity']*100:.2f}%)")
            if r.get('best_logo_similarity') is not None:
                print(f"  Logo Similarity:   {r['best_logo_similarity']:.4f} ({r['best_logo_similarity']*100:.2f}%)")
            print(f"\nTop matches:")
            for match in r["top_matches"]:
                combined = match["combined_similarity"]
                text_sim = match.get("text_similarity")
                logo_sim = match.get("logo_similarity")
                if logo_sim is not None:
                    print(f"  • {match['remote_form']}: {combined:.4f} (text: {text_sim:.4f}, logo: {logo_sim:.4f})")
                else:
                    print(f"  • {match['remote_form']}: {combined:.4f} ({combined*100:.2f}%)")

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

