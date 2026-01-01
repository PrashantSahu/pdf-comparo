# PDF Form Comparison

A Python tool for comparing PDF forms using semantic embeddings and logo matching. For each form in the `local_forms/` directory, it finds the closest matching form(s) in the `remote_forms/` directory.

## Features

- **Semantic Text Matching**: Uses sentence-transformers (`all-mpnet-base-v2`) for meaningful text comparison
- **Logo/Brand Matching**: Uses CLIP to compare logos and visual branding in forms
- **Weighted Scoring**: Combines text (70%) and logo (30%) similarity scores
- **Handles Long Documents**: Automatic chunking with mean pooling for documents of any length

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

### 3. Add your PDF files

- Place your source PDFs in the `local_forms/` directory
- Place your reference/template PDFs in the `remote_forms/` directory

### 4. Run the comparison

```bash
uv run python compare_forms_embeddings.py
```

This automatically creates a virtual environment and installs dependencies on first run.

## Dependencies

- `PyPDF2` - PDF text extraction
- `pymupdf` - PDF image/logo extraction
- `Pillow` - Image processing
- `sentence-transformers` - Text and CLIP embeddings
- `faiss-cpu` - Efficient similarity search
- `numpy` - Scientific computing
- `tqdm` - Progress bars
