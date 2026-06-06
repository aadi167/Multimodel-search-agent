"""
Multimodal Search Engine
Core search logic — text-to-image, image-to-image, zero-shot classification.

All three modes use the same FAISS index and CLIP encoder.
The magic is that CLIP maps images and text to the SAME embedding space,
so searching by text vs image is the same FAISS call with a different query vector.
"""

import time
from typing import List, Dict, Union
import numpy as np
import faiss
from PIL import Image

from src.model.clip_encoder import encoder
from src.model.indexer import load_index, get_index_stats
from src.utils.config import settings
from src.utils.helpers import image_to_base64, get_image_extension, Timer
from src.utils.logger import logger


class MultimodalSearchEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._ready = False
        return cls._instance

    def load(self):
        if self._ready:
            return

        # Load CLIP model
        encoder.load()

        # Load FAISS index
        self.index, self.image_paths = load_index()
        self.stats = get_index_stats(self.index, self.image_paths)
        self._ready = True

        logger.info(
            f"Search engine ready | "
            f"{self.index.ntotal:,} images | "
            f"device: {encoder.device}"
        )

    @property
    def is_ready(self) -> bool:
        return self._ready

    # ── Text → Images ──────────────────────────────────────────────────────
    def search_by_text(
        self,
        query: str,
        top_k: int = None,
        include_base64: bool = False,
    ) -> Dict:
        """
        Find images matching a text description.

        How it works:
        1. Encode text to 512-dim vector using CLIP text encoder
        2. Run FAISS inner product search (= cosine similarity for normalized vectors)
        3. Return top-K image paths + similarity scores

        Args:
            query: natural language description e.g. "a red car at sunset"
            top_k: number of results to return
            include_base64: include base64-encoded images in response (for UI)
        """
        top_k = top_k or settings.DEFAULT_TOP_K

        with Timer("encode_text") as t_enc:
            query_vec = encoder.encode_text(query)

        with Timer("faiss_search") as t_search:
            scores, indices = self.index.search(query_vec, top_k)

        results = self._format_results(scores[0], indices[0], include_base64)

        return {
            "query": query,
            "query_type": "text",
            "results": results,
            "total_results": len(results),
            "encode_ms": t_enc.elapsed_ms,
            "search_ms": t_search.elapsed_ms,
        }

    # ── Image → Images ─────────────────────────────────────────────────────
    def search_by_image(
        self,
        image: Union[Image.Image, str],
        top_k: int = None,
        include_base64: bool = False,
    ) -> Dict:
        """
        Find visually similar images (reverse image search).

        Same as text search but uses the image encoder instead.
        Since both encoders map to the same 512-dim space, the FAISS
        search call is identical.

        Args:
            image: PIL Image or file path
            top_k: number of results
            include_base64: include base64-encoded images in response
        """
        top_k = top_k or settings.DEFAULT_TOP_K

        with Timer("encode_image") as t_enc:
            query_vec = encoder.encode_image(image)

        with Timer("faiss_search") as t_search:
            scores, indices = self.index.search(query_vec, top_k)

        results = self._format_results(scores[0], indices[0], include_base64)

        return {
            "query_type": "image",
            "results": results,
            "total_results": len(results),
            "encode_ms": t_enc.elapsed_ms,
            "search_ms": t_search.elapsed_ms,
        }

    # ── Zero-shot classification ────────────────────────────────────────────
    def classify(
        self,
        image: Union[Image.Image, str],
        labels: List[str],
    ) -> Dict:
        """
        Classify an image into custom categories WITHOUT any training.

        How it works:
        1. Encode the image to a vector
        2. Encode each label as "a photo of {label}" to a vector
        3. Compute cosine similarity between image and each label vector
        4. Return sorted scores

        This works because CLIP was trained to align image-text pairs.
        "a photo of a cat" and a photo of a cat end up near each other
        in the shared embedding space.

        Args:
            image: PIL Image or file path
            labels: list of category names e.g. ["cat", "dog", "car"]
        """
        with Timer("encode") as t:
            img_vec = encoder.encode_image(image)
            # Encode all labels in one batch for efficiency
            label_texts = [f"a photo of {label}" for label in labels]
            label_vecs = encoder.encode_texts_batch(label_texts)

        scores = {}
        for label, label_vec in zip(labels, label_vecs):
            similarity = float(np.dot(img_vec.flatten(), label_vec.flatten()))
            scores[label] = round(similarity, 4)

        scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))
        top_label = list(scores.keys())[0]

        return {
            "scores": scores,
            "top_label": top_label,
            "confidence": scores[top_label],
            "encode_ms": t.elapsed_ms,
        }

    # ── Hybrid search ───────────────────────────────────────────────────────
    def hybrid_search(
        self,
        text_query: str,
        image: Union[Image.Image, str],
        text_weight: float = 0.5,
        top_k: int = None,
        include_base64: bool = False,
    ) -> Dict:
        """
        Combine text + image query vectors for more precise search.

        Example: upload a shoe image + type "blue" to find blue versions
        of that shoe style. Neither query alone would work as well.

        The combined vector is a weighted average of text and image vectors,
        re-normalized before searching FAISS.

        Args:
            text_query: text description to refine the search
            image: reference image to search from
            text_weight: 0.0 = pure image search, 1.0 = pure text search
        """
        top_k = top_k or settings.DEFAULT_TOP_K
        image_weight = 1.0 - text_weight

        with Timer("encode") as t:
            text_vec = encoder.encode_text(text_query)
            image_vec = encoder.encode_image(image)

        # Weighted combination + re-normalize
        combined = text_weight * text_vec + image_weight * image_vec
        norm = np.linalg.norm(combined, axis=-1, keepdims=True)
        combined = combined / (norm + 1e-10)

        with Timer("faiss_search") as t_search:
            scores, indices = self.index.search(combined, top_k)

        results = self._format_results(scores[0], indices[0], include_base64)

        return {
            "query": text_query,
            "query_type": "hybrid",
            "text_weight": text_weight,
            "image_weight": image_weight,
            "results": results,
            "total_results": len(results),
            "encode_ms": t.elapsed_ms,
            "search_ms": t_search.elapsed_ms,
        }

    # ── Internal helpers ────────────────────────────────────────────────────
    def _format_results(
        self,
        scores: np.ndarray,
        indices: np.ndarray,
        include_base64: bool,
    ) -> List[Dict]:
        results = []
        for rank, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0 or idx >= len(self.image_paths):
                continue

            path = self.image_paths[idx]
            result = {
                "rank": rank + 1,
                "path": path,
                "filename": path.split("/")[-1].split("\\")[-1],
                "score": round(float(score), 4),
            }

            if include_base64:
                try:
                    result["image_base64"] = image_to_base64(path)
                    result["image_format"] = get_image_extension(path)
                except Exception:
                    result["image_base64"] = None

            results.append(result)

        return results

    def get_stats(self) -> Dict:
        if not self._ready:
            return {"status": "not_loaded"}
        return {
            "status": "ready",
            "total_images": self.index.ntotal,
            "embedding_dim": self.index.d,
            "index_type": type(self.index).__name__,
            "model": settings.CLIP_MODEL_NAME,
            "device": encoder.device if encoder.is_ready else "unknown",
        }


# Global singleton
search_engine = MultimodalSearchEngine()
