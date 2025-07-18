from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pathlib import Path
from app.services.loaders.loader_registry import get_loader

router = APIRouter()
MAX_FILE_SIZE_MB = 20

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    loader_type: str = Query("pymupdf", description="Type of PDF loader to use")
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB} MB limit.")

    # Save file to data/uploaded_docs
    UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "uploaded_docs"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    save_path = UPLOAD_DIR / file.filename

    try:
        with open(save_path, "wb") as f:
            f.write(contents)

        loader = get_loader(loader_type)
        documents = loader.load(str(save_path))

        return JSONResponse(content={
            "filename": file.filename,
            "saved_as": str(save_path.relative_to(Path.cwd())),
            "loader": loader_type,
            "num_documents": len(documents),
            "preview": documents[0].page_content[:300] if documents else "",
            "metadata": documents[0].metadata if documents else {},
        })

    except Exception as e:
        if save_path.exists():
            save_path.unlink()
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
