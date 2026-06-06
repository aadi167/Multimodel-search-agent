import os
import base64
import time
from pathlib import Path
from typing import List
from PIL import Image
import numpy as np

from src.utils.config import settings


def get_image_files(folder: str) -> List[str]:
    """Recursively find all supported image files in a folder."""
    image_files = []
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(settings.SUPPORTED_FORMATS):
                image_files.append(os.path.join(root, file))
    return sorted(image_files)


def load_image(path: str) -> Image.Image:
    """Load and convert image to RGB."""
    return Image.open(path).convert("RGB")


def image_to_base64(path: str) -> str:
    """Convert image file to base64 string for API responses."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_extension(path: str) -> str:
    ext = Path(path).suffix.lower()
    return "jpeg" if ext in (".jpg", ".jpeg") else ext.lstrip(".")


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """L2 normalize a vector. Required before FAISS inner product search."""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    return vec / (norm + 1e-10)


class Timer:
    """Simple context manager for timing operations."""
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed_ms = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = round((time.time() - self._start) * 1000, 2)
