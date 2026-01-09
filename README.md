# PDF Form Comparison

A Python tool for comparing PDF forms using semantic embeddings and logo matching. For each form in the `local_forms/` directory, it finds the closest matching form(s) in the `remote_forms/` directory.

## Features

- **Semantic Text Matching**: Uses sentence-transformers (`all-mpnet-base-v2`) for meaningful text comparison
- **Logo/Brand Matching**: Uses CLIP to compare logos and visual branding in forms
- **Weighted Scoring**: Combines text (70%) and logo (30%) similarity scores
- **Handles Long Documents**: Automatic chunking with mean pooling for documents of any length
- **Persistent Vector Storage**: Uses ChromaDB for local storage, enabling efficient batch processing

## Requirements

- Python 3.9+
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer

## Quick Start

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone the repository

```bash
git clone https://github.com/PrashantSahu/pdf-comparo.git
cd pdf-comparo
```

### 3. Set up the virtual environment

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate

# Install dependencies
uv sync
```

### 4. Add your PDF files

- Place your reference/template PDFs in `remote_forms/`
- Place your source PDFs to compare in `local_forms/`

### 5. Ingest remote forms (one-time)

```bash
uv run python ingest.py
```

This creates a persistent ChromaDB index at `./chroma_db/`. Re-run when remote forms change - it will only index new forms.

### 6. Compare local forms

```bash
uv run python compare.py
```

This compares all local forms against the indexed remote forms.

## CLI Options

### ingest.py

```bash
uv run python ingest.py [OPTIONS]

Options:
  --forms-dir PATH     Directory containing PDFs to ingest (default: remote_forms)
  --chroma-path PATH   Path to ChromaDB storage (default: ./chroma_db)
  --clear              Clear existing index before ingesting
```

### compare.py

```bash
uv run python compare.py [OPTIONS]

Options:
  --local-dir PATH     Directory containing local PDFs to compare (default: local_forms)
  --chroma-path PATH   Path to ChromaDB storage (default: ./chroma_db)
  --top-k N            Number of top matches to return (default: 5)
  --output FILE        Save results to JSON file (optional)
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  INGEST (ingest.py) - Run once, or when remote forms change    │
│  ───────────────────────────────────────────────────────────    │
│  • Extract text and logos from remote PDFs                      │
│  • Generate embeddings (text + CLIP)                            │
│  • Store in ChromaDB (skips already indexed forms)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  COMPARE (compare.py) - Run for each batch of local forms      │
│  ───────────────────────────────────────────────────────────    │
│  • Load pre-built ChromaDB index                                │
│  • Embed local forms                                            │
│  • Query index for top-K matches                                │
│  • Return combined similarity scores                            │
└─────────────────────────────────────────────────────────────────┘
```

## Dependencies

- `PyPDF2` - PDF text extraction
- `pymupdf` - PDF image/logo extraction
- `Pillow` - Image processing
- `sentence-transformers` - Text and CLIP embeddings
- `chromadb` - Persistent vector storage
- `numpy` - Scientific computing
- `tqdm` - Progress bars
