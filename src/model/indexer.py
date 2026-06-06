"""
FAISS Indexer
Builds the image embedding index and saves it to disk.

Index types explained:
- IndexFlatIP:  Exact inner product search. Best for < 1M images. No approximation.
- IndexIVFFlat: Approximate. Clusters vectors. 100x faster, ~1% accuracy loss.
- IndexIVFPQ:   Approximate + compressed. 32x less memory. For 10M+ images.

This project uses IndexFlatIP (exact) since it's straightforward and accurate.
Swap to IndexIVFFlat for production scale.
"""

import json
import os
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
from typing import List, Tuple

from src.model.clip_encoder import encoder
from src.utils.config import settings
from src.utils.helpers import get_image_files, load_image
from src.utils.logger import logger


def build_index(
    images_folder: str = None,
    index_path: str = None,
    paths_file: str = None,
    batch_size: int = 32,
) -> Tuple[faiss.Index, List[str]]:
    """
    Encode all images in a folder and build a FAISS index.

    Steps:
    1. Find all images recursively
    2. Load + encode in batches using CLIP
    3. Build FAISS IndexFlatIP (exact inner product)
    4. Save index + image paths to disk

    Args:
        images_folder: folder containing images
        index_path: where to save index.faiss
        paths_file: where to save image_paths.json
        batch_size: images per CLIP batch (32 is safe for 8GB RAM)
    """
    images_folder = images_folder or settings.IMAGES_FOLDER
    index_path = index_path or settings.INDEX_PATH
    paths_file = paths_file or settings.IMAGE_PATHS_FILE

    os.makedirs(os.path.dirname(index_path) if os.path.dirname(index_path) else ".", exist_ok=True)

    if not encoder.is_ready:
        encoder.load()

    image_paths = get_image_files(images_folder)
    if not image_paths:
        raise ValueError(f"No images found in {images_folder}. Add .jpg/.png files and retry.")

    logger.info(f"Found {len(image_paths):,} images in {images_folder}")

    # Encode all images in batches
    all_embeddings = []
    valid_paths = []

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid = []

        for path in batch_paths:
            try:
                img = load_image(path)
                batch_images.append(img)
                batch_valid.append(path)
            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

        if not batch_images:
            continue

        try:
            embeddings = encoder.encode_images_batch(batch_images)
            all_embeddings.append(embeddings)
            valid_paths.extend(batch_valid)
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")

    if not all_embeddings:
        raise RuntimeError("No images could be encoded. Check image files.")

    # Stack into matrix: shape (N, 512)
    matrix = np.vstack(all_embeddings)
    logger.info(f"Embedding matrix: {matrix.shape}")

    # Build FAISS index
    # IndexFlatIP = exact inner product (= cosine similarity for L2-normalized vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    # Save to disk
    faiss.write_index(index, index_path)
    with open(paths_file, "w") as f:
        json.dump(valid_paths, f, indent=2)

    logger.info(f"Index saved: {index_path} ({index.ntotal:,} vectors)")
    logger.info(f"Paths saved: {paths_file}")

    return index, valid_paths


def load_index(
    index_path: str = None,
    paths_file: str = None,
) -> Tuple[faiss.Index, List[str]]:
    """Load a previously built FAISS index from disk."""
    index_path = index_path or settings.INDEX_PATH
    paths_file = paths_file or settings.IMAGE_PATHS_FILE

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Index not found at {index_path}. "
            "Run: python scripts/build_index.py"
        )

    index = faiss.read_index(index_path)
    with open(paths_file) as f:
        image_paths = json.load(f)

    logger.info(f"Loaded index: {index.ntotal:,} vectors from {index_path}")
    return index, image_paths


def get_index_stats(index: faiss.Index, image_paths: List[str]) -> dict:
    """Return stats about the loaded index."""
    return {
        "total_images": index.ntotal,
        "embedding_dim": index.d,
        "index_type": type(index).__name__,
        "index_path": settings.INDEX_PATH,
    }
