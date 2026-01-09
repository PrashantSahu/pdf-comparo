#!/usr/bin/env python3
"""
Ingest PDF forms into ChromaDB vector store.

Usage:
    uv run python ingest.py [OPTIONS]

Options:
    --forms-dir PATH     Directory containing PDFs to ingest (default: remote_forms)
    --chroma-path PATH   Path to ChromaDB storage (default: ./chroma_db)
    --clear              Clear existing index before ingesting
"""

import argparse
import time

from sentence_transformers import SentenceTransformer

from embeddings import DEFAULT_MODEL, DEFAULT_CLIP_MODEL
from vector_store import VectorStore, DEFAULT_CHROMA_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Ingest PDF forms into ChromaDB vector store."
    )
    parser.add_argument(
        "--forms-dir",
        type=str,
        default="remote_forms",
        help="Directory containing PDFs to ingest (default: remote_forms)",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help=f"Path to ChromaDB storage (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing index before ingesting",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PDF Form Ingestion")
    print("=" * 60)
    print(f"Forms directory: {args.forms_dir}")
    print(f"ChromaDB path: {args.chroma_path}")
    print(f"Clear existing: {args.clear}")
    print("=" * 60)

    # Load models
    print(f"Loading text model: {DEFAULT_MODEL}...")
    text_model = SentenceTransformer(DEFAULT_MODEL)

    print(f"Loading CLIP model: {DEFAULT_CLIP_MODEL}...")
    clip_model = SentenceTransformer(DEFAULT_CLIP_MODEL)
    print("Models loaded successfully.\n")

    # Initialize vector store
    store = VectorStore(persist_path=args.chroma_path)

    # Clear if requested
    if args.clear:
        print("Clearing existing index...")
        store.clear_collections()

    # Build index
    start_time = time.time()
    new_count, skipped_count = store.build_index(
        pdf_dir=args.forms_dir,
        text_model=text_model,
        clip_model=clip_model,
    )
    elapsed = time.time() - start_time

    # Summary
    text_count, logo_count = store.get_collection_count()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"New documents indexed: {new_count}")
    print(f"Documents skipped (already indexed): {skipped_count}")
    print(f"Total documents in index: {text_count}")
    print(f"Time elapsed: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("=" * 60)


if __name__ == "__main__":
    main()

