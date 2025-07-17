import fitz  # PyMuPDF
from pathlib import Path

def trim_pdf(input_path: Path, output_path: Path, max_pages: int = 50):
    doc = fitz.open(input_path)
    total_pages = doc.page_count
    pages_to_keep = min(max_pages, total_pages)

    new_doc = fitz.open()
    for i in range(pages_to_keep):
        new_doc.insert_pdf(doc, from_page=i, to_page=i)

    new_doc.save(output_path)
    new_doc.close()
    doc.close()
    print(f"Trimmed {pages_to_keep} pages from '{input_path.name}' saved to '{output_path.name}'")

def batch_trim_pdfs(input_folder: Path, output_folder: Path, max_pages: int = 50):
    output_folder.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_folder.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {input_folder}")
        return

    for pdf_file in pdf_files:
        output_file = output_folder / pdf_file.name
        trim_pdf(pdf_file, output_file, max_pages)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python batch_trim_pdfs.py /path/to/input_folder /path/to/output_folder")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    batch_trim_pdfs(input_dir, output_dir)
