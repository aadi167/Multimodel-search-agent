"""
FastAPI Application — Multimodal Search Engine
REST API for text-to-image, image-to-image, and zero-shot classification.
"""

import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn

from src.api.schemas import (
    TextSearchRequest, HybridSearchRequest,
    SearchResponse, ClassifyResponse, IndexStatsResponse,
)
from src.model.search_engine import search_engine
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Multimodal Search Engine API...")
    search_engine.load()
    logger.info("API ready")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Multimodal AI Search Engine API",
    description="CLIP-powered vector search API for text, image, and hybrid queries.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "healthy", "engine_ready": search_engine.is_ready}


@app.get("/index/stats", response_model=IndexStatsResponse)
@app.get("/stats", response_model=IndexStatsResponse)
def index_stats():
    """Return statistics about the loaded image index."""
    return search_engine.get_stats()


# ── Text → Images ─────────────────────────────────────────────────────────
@app.post("/search/text", response_model=SearchResponse)
def search_text(req: TextSearchRequest):
    """
    Search images using a natural language description.

    Examples:
    - "a golden retriever playing fetch"
    - "city skyline at night with lights"
    - "person wearing red jacket in mountains"
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not ready. Build the index first.")

    try:
        result = search_engine.search_by_text(
            query=req.query,
            top_k=req.top_k,
            include_base64=req.include_base64,
        )
        return result
    except Exception as e:
        logger.error(f"Text search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Image → Images ─────────────────────────────────────────────────────────
@app.post("/search/image", response_model=SearchResponse)
async def search_by_image(
    file: UploadFile = File(..., description="Query image to find similar images"),
    top_k: int = Form(default=5),
    include_base64: bool = Form(default=False),
):
    """
    Upload an image to find visually similar images (reverse image search).
    Supports JPG, PNG, WebP formats.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not ready.")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = search_engine.search_by_image(
            image=image,
            top_k=top_k,
            include_base64=include_base64,
        )
        return result
    except Exception as e:
        logger.error(f"Image search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Zero-shot classification ────────────────────────────────────────────────
@app.post("/classify", response_model=ClassifyResponse)
async def classify_image(
    request: Request,
    file: UploadFile = File(...),
    labels: str = Form(default="cat,dog,car,person,food,building,nature",
                       description="Comma-separated category names"),
):
    """
    Classify an image into custom categories — no training needed.

    Example labels: "cat,dog,car,person"
    Works for any categories — CLIP has broad visual knowledge.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not ready.")

    form_data = await request.form()
    if "labels" in form_data and not str(form_data["labels"]).strip():
        raise HTTPException(status_code=400, detail="Provide at least one label.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        label_list = [l.strip() for l in labels.split(",") if l.strip()]

        if not label_list:
            raise HTTPException(status_code=400, detail="Provide at least one label.")

        result = search_engine.classify(image=image, labels=label_list)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Hybrid search ───────────────────────────────────────────────────────────
@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(
    file: UploadFile = File(..., description="Reference image"),
    text_query: str = Form(..., description="Text to refine the search"),
    top_k: int = Form(default=5),
    text_weight: float = Form(default=0.5, ge=0.0, le=1.0),
    include_base64: bool = Form(default=False),
):
    """
    Combine image + text for more precise search.

    Example: upload a shoe photo + query "blue leather"
    to find blue leather versions of that shoe style.
    Neither query alone would work as well.
    """
    if not search_engine.is_ready:
        raise HTTPException(status_code=503, detail="Search engine not ready.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        result = search_engine.hybrid_search(
            text_query=text_query,
            image=image,
            text_weight=text_weight,
            top_k=top_k,
            include_base64=include_base64,
        )
        return result
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=False)

