#!/usr/bin/env python3
"""
PDF Form Comparison using Sentence Transformers and CLIP.

This script compares PDF forms between two directories using semantic embeddings.
For each form in the source directory, it finds the closest matching form(s) in
the target directory.

Features:
- Uses sentence-transformers (all-mpnet-base-v2) for semantic text embeddings
- Uses CLIP for logo/image embeddings to match forms by visual branding
- Handles long documents via chunking and mean pooling
- Uses numpy for similarity computation (with optional FAISS for large scale)
- Supports thousands of documents efficiently
- Combines text and logo similarity with configurable weights
"""

import os
import argparse
import json
from pathlib import Path
from typing import Optional
import io

import numpy as np
import PyPDF2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from PIL import Image

# Try to import pymupdf for image extraction
try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Try to import FAISS, but fall back to numpy if not available or problematic
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


# Configuration
DEFAULT_MODEL = "all-mpnet-base-v2"
DEFAULT_CLIP_MODEL = "clip-ViT-B-32"
DEFAULT_CHUNK_SIZE = 1000  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # overlap between chunks
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
            logo_weight: Weight for logo similarity (0.0 to 1.0). Text weight = 1 - logo_weight.
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

    def extract_logos_from_pdf(self, pdf_path: str, max_images: int = 5) -> list[Image.Image]:
        """
        Extract potential logo images from the first page of a PDF.

        Focuses on images in the header area (top 20% of the page) which are
        typically where logos appear.

        Args:
            pdf_path: Path to the PDF file.
            max_images: Maximum number of images to extract.

        Returns:
            List of PIL Image objects.
        """
        if not PYMUPDF_AVAILABLE:
            return []

        logos = []
        try:
            doc = fitz.open(pdf_path)
            if len(doc) == 0:
                return []

            # Only look at first page for logos
            page = doc[0]
            page_height = page.rect.height
            header_threshold = page_height * 0.25  # Top 25% of page

            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list[:max_images * 2]):
                try:
                    xref = img_info[0]

                    # Get image bbox to check if it's in header area
                    img_rects = page.get_image_rects(xref)
                    if img_rects:
                        img_rect = img_rects[0]
                        # Check if image is in the header area
                        if img_rect.y0 > header_threshold:
                            continue  # Skip images not in header

                    # Extract image
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(image_bytes))

                    # Convert to RGB if necessary
                    if image.mode != "RGB":
                        image = image.convert("RGB")

                    # Filter out very small or very large images (likely not logos)
                    width, height = image.size
                    if width < 20 or height < 20:
                        continue
                    if width > 1000 and height > 1000:
                        continue

                    logos.append(image)

                    if len(logos) >= max_images:
                        break

                except Exception:
                    continue

            doc.close()

        except Exception as e:
            print(f"Error extracting logos from {pdf_path}: {e}")

        return logos

    def embed_logos(self, logos: list[Image.Image]) -> Optional[np.ndarray]:
        """
        Create embedding for logos using CLIP.

        Args:
            logos: List of PIL Image objects.

        Returns:
            Combined logo embedding as numpy array, or None if no logos.
        """
        if not logos or not self.use_logo_comparison:
            return None

        try:
            # Encode all logos
            logo_embeddings = self.clip_model.encode(logos, show_progress_bar=False)

            # Mean pooling if multiple logos
            combined_embedding = np.mean(logo_embeddings, axis=0)

            # Normalize
            combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

            return combined_embedding

        except Exception as e:
            print(f"Error embedding logos: {e}")
            return None

    def get_pdf_files(self, directory: str) -> list[Path]:
        """Get all PDF files in a directory."""
        return sorted(Path(directory).glob("*.pdf"))

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
            text = self.extract_text_from_pdf(str(pdf_path))
            text_embedding = self.embed_document(text)

            if text_embedding is not None:
                text_embeddings.append(text_embedding)
                file_names.append(pdf_path.name)
                texts.append(text[:500] + "..." if len(text) > 500 else text)

                # Extract and embed logos if enabled
                if include_logos and self.use_logo_comparison:
                    logos = self.extract_logos_from_pdf(str(pdf_path))
                    logo_embedding = self.embed_logos(logos)
                    if logo_embedding is not None:
                        logo_embeddings.append(logo_embedding)
                    else:
                        # Use zero vector as placeholder
                        logo_embeddings.append(np.zeros(self.clip_model.get_sentence_embedding_dimension()))

        text_arr = np.array(text_embeddings).astype("float32")
        logo_arr = np.array(logo_embeddings).astype("float32") if logo_embeddings else None

        return text_arr, logo_arr, file_names, texts

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

        Combines text similarity with logo similarity (if enabled) using weighted average.

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
            dimension = remote_text_emb.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(remote_text_emb)
            text_similarities, text_indices = index.search(local_text_emb, len(remote_names))
        else:
            print("Using numpy for similarity search...")
            text_similarities, text_indices = self.compute_similarities_numpy(
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
                    # Combined score with logo weight
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare PDF forms using semantic embeddings and logo matching.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_forms_embeddings.py
  python compare_forms_embeddings.py --local-dir my_forms --remote-dir templates
  python compare_forms_embeddings.py --top-k 10 --output results.json
  python compare_forms_embeddings.py --logo-weight 0.4  # Give more weight to logo matching
  python compare_forms_embeddings.py --no-logo  # Disable logo comparison
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
        help=f"Sentence transformer model for text (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--clip-model",
        default=DEFAULT_CLIP_MODEL,
        help=f"CLIP model for logo comparison (default: {DEFAULT_CLIP_MODEL})",
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
        "--logo-weight",
        type=float,
        default=DEFAULT_LOGO_WEIGHT,
        help=f"Weight for logo similarity (0.0-1.0, default: {DEFAULT_LOGO_WEIGHT})",
    )
    parser.add_argument(
        "--no-logo",
        action="store_true",
        help="Disable logo comparison (text only)",
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
        clip_model_name=args.clip_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        logo_weight=args.logo_weight,
        use_logo_comparison=not args.no_logo,
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

