#!/usr/bin/env python3
"""
Compare local PDF forms against indexed forms in ChromaDB.

Usage:
    uv run python compare.py [OPTIONS]

Options:
    --local-dir PATH     Directory containing local PDFs to compare (default: local_forms)
    --chroma-path PATH   Path to ChromaDB storage (default: ./chroma_db)
    --top-k N            Number of top matches to return (default: 5)
    --output FILE        Save results to JSON file (optional)
"""

import argparse
import json
import time

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from embeddings import DEFAULT_MODEL, DEFAULT_CLIP_MODEL, embed_document, embed_logos
from pdf_extraction import extract_text_from_pdf, extract_logos_from_pdf, get_pdf_files
from vector_store import VectorStore, DEFAULT_CHROMA_PATH


def main():
    parser = argparse.ArgumentParser(
        description="Compare local PDF forms against indexed forms in ChromaDB."
    )
    parser.add_argument(
        "--local-dir",
        type=str,
        default="local_forms",
        help="Directory containing local PDFs to compare (default: local_forms)",
    )
    parser.add_argument(
        "--chroma-path",
        type=str,
        default=DEFAULT_CHROMA_PATH,
        help=f"Path to ChromaDB storage (default: {DEFAULT_CHROMA_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top matches to return (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file (optional)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PDF Form Comparison")
    print("=" * 60)
    print(f"Local directory: {args.local_dir}")
    print(f"ChromaDB path: {args.chroma_path}")
    print(f"Top K matches: {args.top_k}")
    print("=" * 60)

    # Load models
    print(f"Loading text model: {DEFAULT_MODEL}...")
    text_model = SentenceTransformer(DEFAULT_MODEL)

    print(f"Loading CLIP model: {DEFAULT_CLIP_MODEL}...")
    clip_model = SentenceTransformer(DEFAULT_CLIP_MODEL)
    print("Models loaded successfully.\n")

    # Load vector store
    store = VectorStore(persist_path=args.chroma_path)

    if not store.is_indexed():
        print("ERROR: No indexed forms found. Run ingest.py first.")
        return

    text_count, _ = store.get_collection_count()
    print(f"Loaded index with {text_count} forms.\n")

    # Get local forms
    local_files = get_pdf_files(args.local_dir)
    print(f"Found {len(local_files)} local forms to compare.\n")

    if len(local_files) == 0:
        print("No local forms found.")
        return

    # Build embeddings for local forms
    print("Building embeddings for local forms...")
    local_filenames = []
    local_text_embeddings = []
    local_logo_embeddings = []

    clip_dim = clip_model.get_sentence_embedding_dimension()

    for pdf_path in tqdm(local_files, desc="Processing PDFs"):
        filename = pdf_path.name

        text = extract_text_from_pdf(str(pdf_path))
        text_emb = embed_document(text, text_model)

        if text_emb is None:
            print(f"  Skipping {filename}: no text extracted")
            continue

        logos = extract_logos_from_pdf(str(pdf_path))
        logo_emb = embed_logos(logos, clip_model)
        if logo_emb is None:
            logo_emb = np.zeros(clip_dim)

        local_filenames.append(filename)
        local_text_embeddings.append(text_emb)
        local_logo_embeddings.append(logo_emb)

    # Query ChromaDB
    print("\nQuerying index for matches...")
    start_time = time.time()

    results = store.query(
        text_embeddings=np.array(local_text_embeddings),
        logo_embeddings=np.array(local_logo_embeddings),
        top_k=args.top_k,
    )

    elapsed = time.time() - start_time

    # Display results
    print("\n" + "=" * 80)
    print("MATCHING RESULTS")
    print("(Text weight: 70%, Logo weight: 30%)")
    print("=" * 80)

    output_results = []

    for i, (filename, result) in enumerate(zip(local_filenames, results)):
        print(f"\n{'─' * 60}")
        print(f"Local Form: {filename}")
        print(f"Best Match: {result['best_match']}")
        print(f"Combined Similarity: {result['best_similarity']:.4f} ({result['best_similarity']*100:.2f}%)")
        print(f"  Text Similarity:   {result['best_text_similarity']:.4f} ({result['best_text_similarity']*100:.2f}%)")
        print(f"  Logo Similarity:   {result['best_logo_similarity']:.4f} ({result['best_logo_similarity']*100:.2f}%)")

        print(f"\nTop {args.top_k} matches:")
        for match in result['top_matches']:
            print(f"  • {match['remote_form']}: {match['combined_similarity']:.4f} "
                  f"(text: {match['text_similarity']:.4f}, logo: {match['logo_similarity']:.4f})")

        output_results.append({
            "local_form": filename,
            "best_match": result['best_match'],
            "best_similarity": result['best_similarity'],
            "top_matches": result['top_matches'],
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in output_results:
        print(f"{r['local_form']} → {r['best_match']} ({r['best_similarity']*100:.2f}%)")

    print(f"\n{'=' * 60}")
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Forms compared: {len(local_filenames)}")
    print(f"Time elapsed: {elapsed:.2f} seconds")
    print("=" * 60)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()

