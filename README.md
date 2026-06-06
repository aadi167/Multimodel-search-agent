# Multimodal Search Engine — CLIP + FAISS

Search images using text descriptions or upload an image to find visually similar ones.
Built with OpenAI CLIP, Facebook FAISS, FastAPI, and Gradio.

## What it does

| Search Mode | Input | Output |
|---|---|---|
| Text → Images | "red sports car at night" | Top-K matching images |
| Image → Images | Upload a photo | Visually similar images |
| Zero-shot classify | Image + label list | Category scores, no training needed |

## Architecture

```
[User Query]
    │
    ├─ Text query ──→ [CLIP Text Encoder] ──→ 512-dim vector
    │                                              │
    └─ Image query ─→ [CLIP Image Encoder] ──→ 512-dim vector
                                                   │
                                          [FAISS Index Search]
                                                   │
                                          [Top-K Results + Scores]
                                                   │
                                          [FastAPI Response / Gradio UI]
```

## Folder Structure

```
multimodal-search/
├── src/
│   ├── api/         main.py (FastAPI), schemas.py
│   ├── model/       clip_encoder.py, search_engine.py, indexer.py
│   └── utils/       config.py, logger.py, helpers.py
├── frontend/        app.py (Gradio UI)
├── data/
│   └── images/      Put your images here
├── tests/           test_search.py
├── scripts/         build_index.py, download_sample_data.py
├── notebooks/       exploration.ipynb
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt

# 2. Add images to data/images/ folder
#    OR download sample dataset:
python scripts/download_sample_data.py

# 3. Build the FAISS index
python scripts/build_index.py

# 4. Start API server
set PYTHONPATH=.             # Windows
export PYTHONPATH=.          # Mac/Linux
uvicorn src.api.main:app --reload --port 8000

# 5. Launch Gradio UI (new terminal)
python frontend/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | /search/text | Search images by text description |
| POST | /search/image | Find similar images by uploading one |
| POST | /classify | Zero-shot classify an image |
| GET  | /index/stats | Index statistics |
| GET  | /health | Health check |

## Model Info

- Model: `openai/clip-vit-base-patch32`
- Embedding dims: 512
- Text context: 77 tokens
- Image size: 224×224 (auto-resized)
- Index type: FAISS IndexFlatIP (exact inner product)

## Interview Talking Points

- L2 normalization converts inner product to cosine similarity
- FAISS IndexFlatIP is exact search — no approximation, best for < 1M images
- Zero-shot classification works because image+text share the same embedding space
- Incremental indexing: `index.add()` supports adding new images without rebuilding
- For 10M+ images: swap to IndexIVFPQ (approximate, 100x faster, 32x less memory)
