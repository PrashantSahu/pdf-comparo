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

from comparator import PDFFormComparator, DEFAULT_TOP_K, DEFAULT_LOGO_WEIGHT
from pdf_extraction import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from embeddings import DEFAULT_MODEL, DEFAULT_CLIP_MODEL


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

