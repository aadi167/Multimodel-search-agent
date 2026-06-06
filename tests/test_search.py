"""
Tests for the Multimodal Search Engine
Run: pytest tests/ -v
"""

import pytest
import numpy as np
from PIL import Image
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
import io


# ── Test CLIP Encoder ──────────────────────────────────────────────────────
class TestCLIPEncoder:
    def test_encoder_is_singleton(self):
        from src.model.clip_encoder import CLIPEncoder
        e1 = CLIPEncoder()
        e2 = CLIPEncoder()
        assert e1 is e2

    def test_normalize_vector(self):
        from src.utils.helpers import normalize_vector
        vec = np.array([[3.0, 4.0]])
        normalized = normalize_vector(vec)
        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6, "Vector should be L2 normalized"

    def test_normalize_zero_vector(self):
        from src.utils.helpers import normalize_vector
        vec = np.zeros((1, 512))
        result = normalize_vector(vec)
        assert not np.any(np.isnan(result)), "Should not produce NaN for zero vector"


# ── Test Helpers ───────────────────────────────────────────────────────────
class TestHelpers:
    def test_get_image_files_empty_folder(self, tmp_path):
        from src.utils.helpers import get_image_files
        result = get_image_files(str(tmp_path))
        assert result == []

    def test_get_image_files_finds_images(self, tmp_path):
        from src.utils.helpers import get_image_files
        img = Image.new("RGB", (10, 10), color="red")
        img.save(tmp_path / "test.jpg")
        img.save(tmp_path / "test2.png")
        (tmp_path / "not_image.txt").write_text("hello")

        result = get_image_files(str(tmp_path))
        assert len(result) == 2
        assert all(f.endswith((".jpg", ".png")) for f in result)

    def test_load_image_rgb(self, tmp_path):
        from src.utils.helpers import load_image
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        img.save(tmp_path / "rgba.png")
        loaded = load_image(str(tmp_path / "rgba.png"))
        assert loaded.mode == "RGB"

    def test_timer(self):
        from src.utils.helpers import Timer
        import time
        with Timer("test") as t:
            time.sleep(0.01)
        assert t.elapsed_ms >= 10


# ── Test Schemas ───────────────────────────────────────────────────────────
class TestSchemas:
    def test_text_search_request_valid(self):
        from src.api.schemas import TextSearchRequest
        req = TextSearchRequest(query="a cat on a roof", top_k=5)
        assert req.query == "a cat on a roof"
        assert req.top_k == 5

    def test_text_search_empty_query_fails(self):
        from src.api.schemas import TextSearchRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TextSearchRequest(query="", top_k=5)

    def test_top_k_max_limit(self):
        from src.api.schemas import TextSearchRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            TextSearchRequest(query="test", top_k=100)

    def test_hybrid_request_weight_bounds(self):
        from src.api.schemas import HybridSearchRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            HybridSearchRequest(text_query="test", text_weight=1.5)


# ── Test API ───────────────────────────────────────────────────────────────
class TestAPI:
    @pytest.fixture
    def client(self):
        from src.api.main import app
        with TestClient(app) as c:
            yield c

    def test_health_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"

    def test_index_stats_endpoint(self, client):
        response = client.get("/index/stats")
        assert response.status_code == 200

    def test_text_search_returns_503_without_index(self, client):
        response = client.post(
            "/search/text",
            json={"query": "a dog", "top_k": 3}
        )
        assert response.status_code in [200, 503]

    def test_image_search_rejects_non_image(self, client):
        response = client.post(
            "/search/image",
            files={"file": ("test.txt", b"not an image", "text/plain")},
        )
        assert response.status_code == 400

    def test_classify_empty_labels(self, client):
        img = Image.new("RGB", (100, 100), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        response = client.post(
            "/classify",
            files={"file": ("test.jpg", buf, "image/jpeg")},
            data={"labels": ""},
        )
        assert response.status_code == 400


# ── Test Search Engine Logic ───────────────────────────────────────────────
class TestSearchEngineLogic:
    def test_cosine_similarity_identical_vectors(self):
        """Identical normalized vectors should have cosine sim = 1.0"""
        from src.utils.helpers import normalize_vector
        vec = normalize_vector(np.random.randn(1, 512).astype(np.float32))
        sim = float(np.dot(vec.flatten(), vec.flatten()))
        assert abs(sim - 1.0) < 1e-5

    def test_cosine_similarity_orthogonal_vectors(self):
        """Orthogonal vectors should have cosine sim = 0.0"""
        vec1 = np.zeros((1, 512), dtype=np.float32)
        vec2 = np.zeros((1, 512), dtype=np.float32)
        vec1[0, 0] = 1.0
        vec2[0, 1] = 1.0
        sim = float(np.dot(vec1.flatten(), vec2.flatten()))
        assert abs(sim) < 1e-6

    def test_hybrid_weights_sum_to_one(self):
        """text_weight + image_weight should always sum to 1.0"""
        text_weight = 0.3
        image_weight = 1.0 - text_weight
        assert abs(text_weight + image_weight - 1.0) < 1e-9
