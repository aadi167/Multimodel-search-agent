"""
CLIP Encoder
Loads the CLIP model once at startup and provides encode_image / encode_text methods.

Key design decisions:
- Singleton pattern: model loaded once, reused for all queries
- L2 normalization: makes inner product = cosine similarity in FAISS
- torch.no_grad(): disables gradient computation for faster inference
- device auto-detection: uses GPU if available, falls back to CPU
"""

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Union

from src.utils.config import settings
from src.utils.helpers import normalize_vector
from src.utils.logger import logger


class CLIPEncoder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return
        logger.info(f"Loading CLIP model: {settings.CLIP_MODEL_NAME}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        if self.device == "cpu":
            torch.set_num_threads(1)
            torch.set_grad_enabled(False)

        self.model = CLIPModel.from_pretrained(settings.CLIP_MODEL_NAME)
        self.processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info("CLIP model loaded successfully")

    @property
    def is_ready(self) -> bool:
        return self._loaded

    def encode_image(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Encode a single image to a 512-dim L2-normalized vector.

        Args:
            image: PIL Image or path string

        Returns:
            np.ndarray of shape (1, 512), float32, L2 normalized
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image must be a PIL Image or file path string")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs)
            features = outputs.pooler_output
            features = self.model.visual_projection(features)

        features = features.cpu().numpy().astype(np.float32)
        return normalize_vector(features)

    def encode_images_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(
                images=batch, return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
                features = outputs.pooler_output
                features = self.model.visual_projection(features)
            embeddings = features.cpu().numpy().astype(np.float32)
            all_embeddings.append(normalize_vector(embeddings))

        return np.vstack(all_embeddings)

    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            features = outputs.pooler_output
            features = self.model.text_projection(features)

        features = features.cpu().numpy().astype(np.float32)
        return normalize_vector(features)

    def encode_texts_batch(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.text_model(**inputs)
            features = outputs.pooler_output
            features = self.model.text_projection(features)

        features = features.cpu().numpy().astype(np.float32)
        return normalize_vector(features)

    def similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized vectors."""
        return float(np.dot(vec1.flatten(), vec2.flatten()))


# Global singleton
encoder = CLIPEncoder()
