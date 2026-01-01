#!/usr/bin/env python3
"""
PDF Form Comparison using Sentence Transformers and CLIP.

Compares PDF forms in local_forms/ against remote_forms/ using semantic
text embeddings and logo/image comparison.
"""

import os

from comparator import PDFFormComparator

LOCAL_DIR = "local_forms"
REMOTE_DIR = "remote_forms"


def main():
    """Main entry point."""
    # Validate directories
    if not os.path.isdir(LOCAL_DIR):
        print(f"Error: Local directory not found: {LOCAL_DIR}")
        return 1
    if not os.path.isdir(REMOTE_DIR):
        print(f"Error: Remote directory not found: {REMOTE_DIR}")
        return 1

    # Initialize comparator and find matches
    comparator = PDFFormComparator()
    results = comparator.find_matches(local_dir=LOCAL_DIR, remote_dir=REMOTE_DIR)

    # Print results
    if results:
        comparator.print_results(results)

    return 0


if __name__ == "__main__":
    exit(main())

