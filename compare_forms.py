#!/usr/bin/env python3
"""
Script to compare PDF forms between local_forms and remote_forms directories.
For each form in local_forms, find the closest matching form in remote_forms.
"""

import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import numpy as np


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text


def get_pdf_files(directory: str) -> list:
    """Get all PDF files in a directory."""
    return list(Path(directory).glob("*.pdf"))


def compute_similarity(text1: str, text2: str, vectorizer: TfidfVectorizer) -> float:
    """Compute cosine similarity between two texts."""
    if not text1.strip() or not text2.strip():
        return 0.0
    try:
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return similarity
    except Exception as e:
        print(f"Error computing similarity: {e}")
        return 0.0


def find_closest_matches(builder_dir: str, library_dir: str):
    """Find closest matching forms for each builder form."""
    builder_files = get_pdf_files(builder_dir)
    library_files = get_pdf_files(library_dir)
    
    print(f"Found {len(builder_files)} files in local_forms")
    print(f"Found {len(library_files)} files in remote_forms")
    print("-" * 80)
    
    # Extract text from all PDFs
    print("\nExtracting text from PDFs...")
    builder_texts = {}
    for pdf_path in builder_files:
        text = extract_text_from_pdf(str(pdf_path))
        builder_texts[pdf_path.name] = text
        print(f"  Builder: {pdf_path.name} - {len(text)} chars extracted")
    
    library_texts = {}
    for pdf_path in library_files:
        text = extract_text_from_pdf(str(pdf_path))
        library_texts[pdf_path.name] = text
        print(f"  Library: {pdf_path.name} - {len(text)} chars extracted")
    
    print("\n" + "=" * 80)
    print("MATCHING RESULTS")
    print("=" * 80)
    
    # For each builder form, find the closest library form
    results = []
    for builder_name, builder_text in builder_texts.items():
        best_match = None
        best_similarity = -1
        all_similarities = []
        
        for library_name, library_text in library_texts.items():
            vectorizer = TfidfVectorizer(stop_words='english')
            similarity = compute_similarity(builder_text, library_text, vectorizer)
            all_similarities.append((library_name, similarity))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = library_name
        
        # Sort by similarity for detailed view
        all_similarities.sort(key=lambda x: x[1], reverse=True)
        
        results.append({
            'builder': builder_name,
            'best_match': best_match,
            'similarity': best_similarity,
            'all_similarities': all_similarities
        })
        
        print(f"\n{'='*60}")
        print(f"Builder Form: {builder_name}")
        print(f"Best Match:   {best_match}")
        print(f"Similarity:   {best_similarity:.4f} ({best_similarity*100:.2f}%)")
        print(f"\nAll matches (sorted by similarity):")
        for lib_name, sim in all_similarities[:5]:  # Top 5 matches
            print(f"  - {lib_name}: {sim:.4f} ({sim*100:.2f}%)")
    
    return results


if __name__ == "__main__":
    builder_dir = "local_forms"
    library_dir = "remote_forms"
    
    results = find_closest_matches(builder_dir, library_dir)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for r in results:
        print(f"{r['builder']} -> {r['best_match']} (similarity: {r['similarity']:.2%})")

