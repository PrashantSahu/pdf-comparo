"""
PDF extraction utilities for text and logo/image extraction.
"""

import io
from pathlib import Path

import PyPDF2
from PIL import Image

# Try to import pymupdf for image extraction
try:
    import fitz  # pymupdf
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


# Configuration defaults
DEFAULT_CHUNK_SIZE = 1000  # characters per chunk
DEFAULT_CHUNK_OVERLAP = 200  # overlap between chunks


def extract_text_from_pdf(pdf_path: str) -> str:
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


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - chunk_overlap

    return chunks


def extract_logos_from_pdf(pdf_path: str, max_images: int = 5) -> list[Image.Image]:
    """
    Extract potential logo images from the first page of a PDF.

    Focuses on images in the header area (top 25% of the page) which are
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


def get_pdf_files(directory: str) -> list[Path]:
    """Get all PDF files in a directory."""
    return sorted(Path(directory).glob("*.pdf"))

