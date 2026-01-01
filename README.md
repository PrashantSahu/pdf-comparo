# PDF Form Comparison

A Python tool for comparing PDF forms using semantic embeddings and logo matching. For each form in the `local_forms` directory, it finds the closest matching form(s) in the `remote_forms` directory.

## Features

- **Semantic Text Matching**: Uses sentence-transformers (`all-mpnet-base-v2`) for meaningful text comparison
- **Logo/Brand Matching**: Uses CLIP to compare logos and visual branding in forms
- **Weighted Scoring**: Combines text and logo similarity with configurable weights
- **Handles Long Documents**: Automatic chunking with mean pooling for documents of any length
- **Efficient Search**: Optional FAISS integration for large-scale comparisons
- **Flexible Output**: Console display and JSON export options

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

### 3. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:

- Create a virtual environment (`.venv/`)
- Install all dependencies using uv

### 4. Add your PDF files

- Place your source PDFs in the `local_forms/` directory
- Place your reference/template PDFs in the `remote_forms/` directory

### 5. Run the comparison

```bash
# Using uv run (no need to activate venv)
uv run python compare_forms_embeddings.py

# Or activate the venv first
source .venv/bin/activate
python compare_forms_embeddings.py
```

## Usage Options

```bash
# Basic usage (uses default directories)
python compare_forms_embeddings.py

# Specify custom directories
python compare_forms_embeddings.py --local-dir my_forms --remote-dir templates

# Get more matches and save to JSON
python compare_forms_embeddings.py --top-k 10 --output results.json

# Give more weight to logo matching (default is 0.3)
python compare_forms_embeddings.py --logo-weight 0.4

# Disable logo comparison (text only)
python compare_forms_embeddings.py --no-logo

# Use FAISS for faster search (recommended for large datasets)
python compare_forms_embeddings.py --use-faiss
```

### All Options

| Option | Default | Description |
| ------ | ------- | ----------- |
| `--local-dir` | `local_forms` | Directory containing source forms |
| `--remote-dir` | `remote_forms` | Directory containing reference forms |
| `--model` | `all-mpnet-base-v2` | Sentence transformer model for text |
| `--clip-model` | `clip-ViT-B-32` | CLIP model for logo comparison |
| `--top-k` | `5` | Number of top matches to return |
| `--chunk-size` | `1000` | Characters per chunk for long documents |
| `--chunk-overlap` | `200` | Overlap between chunks |
| `--logo-weight` | `0.3` | Weight for logo similarity (0.0-1.0) |
| `--no-logo` | False | Disable logo comparison |
| `--output`, `-o` | None | Output JSON file path |
| `--use-faiss` | False | Use FAISS for similarity search |

## Alternative: Simple TF-IDF Comparison

For a simpler TF-IDF based comparison (no ML models required):

```bash
python compare_forms.py
```

## Deactivate

When done, deactivate the virtual environment:

```bash
deactivate
```

## Dependencies

- `PyPDF2` - PDF text extraction
- `pymupdf` - PDF image/logo extraction
- `Pillow` - Image processing
- `sentence-transformers` - Text and CLIP embeddings
- `faiss-cpu` - Efficient similarity search
- `numpy` - Scientific computing
- `tqdm` - Progress bars
