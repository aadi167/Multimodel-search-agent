"""
Build the FAISS image index.
Run this once after adding images to data/images/.
Run: python scripts/build_index.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.indexer import build_index
from src.utils.config import settings

if __name__ == "__main__":
    print("=" * 50)
    print("Multimodal Search Engine — Index Builder")
    print("=" * 50)
    print(f"Images folder : {settings.IMAGES_FOLDER}")
    print(f"Index output  : {settings.INDEX_PATH}")
    print(f"CLIP model    : {settings.CLIP_MODEL_NAME}")
    print()

    images_folder = settings.IMAGES_FOLDER
    if not os.path.exists(images_folder):
        print(f"ERROR: Images folder not found: {images_folder}")
        print("Create it and add images, or run:")
        print("  python scripts/download_sample_data.py")
        sys.exit(1)

    index, paths = build_index()

    print()
    print("=" * 50)
    print(f"Index built successfully!")
    print(f"Total images indexed: {index.ntotal:,}")
    print()
    print("Next steps:")
    print("  1. Start API:       uvicorn src.api.main:app --reload --port 8000")
    print("  2. Launch UI:       python frontend/app.py")
    print("  3. Open browser:    http://localhost:7860")
    print("=" * 50)
