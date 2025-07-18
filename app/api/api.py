from fastapi import APIRouter
from .endpoints import document_loader


api_router = APIRouter()
api_router.include_router(document_loader.router, tags=['document_loader'])