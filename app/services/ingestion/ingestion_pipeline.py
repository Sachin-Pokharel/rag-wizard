# cli/ingest.py
import typer
from ..loaders.loader_registry import get_loader
from ..chunkers.chunker_registry import get_chunker

app = typer.Typer()

@app.command()
def ingest(
    folder_path: str,
    loader_type: str = "pymupdf",
    chunk_size: int = 500,
    chunk_overlap: int = 50
):
    docs = get_loader(loader_type)
    documents = docs.load(folder_path)
    typer.echo(f"✅ Done: {len(documents)} chunks processed.")
    chunker = get_chunker(chunker_type="recursive", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = chunker.chunk(documents)
    typer.echo(f"✅ Done: {len(documents)} chunks after chunker processed.")
    # embed and store

if __name__ == "__main__":
    app()
