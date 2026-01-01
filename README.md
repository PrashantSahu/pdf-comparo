# PDF Form Comparison

A Python tool for comparing PDF forms using semantic embeddings. For each form in the `local_forms` directory, it finds the closest matching form(s) in the `remote_forms` directory.

## Features

- **Semantic Matching**: Uses sentence-transformers (`all-mpnet-base-v2`) for meaningful text comparison
- **Handles Long Documents**: Automatic chunking with mean pooling for documents of any length
- **Efficient Search**: Optional FAISS integration for large-scale comparisons
- **Flexible Output**: Console display and JSON export options

## Requirements

- Python 3.8+

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/PrashantSahu/pdf-comparo.git
cd pdf-comparo
```

### 2. Run the setup script

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies

### 3. Add your PDF files

- Place your source PDFs in the `local_forms/` directory
- Place your reference/template PDFs in the `remote_forms/` directory

### 4. Activate the virtual environment

```bash
source venv/bin/activate
```

### 5. Run the comparison

```bash
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

# Use FAISS for faster search (recommended for large datasets)
python compare_forms_embeddings.py --use-faiss
```

### All Options

| Option | Default | Description |
|--------|---------|-------------|
| `--local-dir` | `local_forms` | Directory containing source forms |
| `--remote-dir` | `remote_forms` | Directory containing reference forms |
| `--model` | `all-mpnet-base-v2` | Sentence transformer model |
| `--top-k` | `5` | Number of top matches to return |
| `--chunk-size` | `1000` | Characters per chunk for long documents |
| `--chunk-overlap` | `200` | Overlap between chunks |
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
- `sentence-transformers` - Semantic embeddings
- `faiss-cpu` - Efficient similarity search
- `numpy` - Scientific computing
- `tqdm` - Progress bars

