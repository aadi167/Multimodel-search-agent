import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # Model
    CLIP_MODEL_NAME: str = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", 512))

    # Index
    INDEX_PATH: str = os.getenv("INDEX_PATH", "data/index.faiss")
    IMAGE_PATHS_FILE: str = os.getenv("IMAGE_PATHS_FILE", "data/image_paths.json")
    IMAGES_FOLDER: str = os.getenv("IMAGES_FOLDER", "data/images")

    # API
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    MAX_RESULTS: int = int(os.getenv("MAX_RESULTS", 20))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", 5))

    # UI
    GRADIO_PORT: int = int(os.getenv("GRADIO_PORT", 7860))
    GRADIO_SHARE: bool = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Supported image formats
    SUPPORTED_FORMATS: tuple = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif")


settings = Settings()
