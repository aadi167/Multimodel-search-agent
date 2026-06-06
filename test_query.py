import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
from src.model.search_engine import search_engine

# Load index
search_engine.load()

# Search
print("Search Results for 'a dog':")
results = search_engine.search_by_text("a dog", top_k=3)
for res in results["results"]:
    print(f"Path: {res['path']}, Score: {res['score']:.4f}")
