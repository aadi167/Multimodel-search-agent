"""
Online Image Scraper — Google / DDG / Unsplash / Wikimedia live image fetching.
Fast parallel downloading with ThreadPoolExecutor.
"""

import os
import io
import time
import requests
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from src.utils.logger import logger

UNSPLASH_KEY = "F5iycODjTNyxFZqdV9PtOxnucJ2iU2zTo6dBEnhZHf4"


def fetch_single_image(url: str, timeout: int = 5) -> Image.Image:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", "image/jpeg"):
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            return img
    except Exception:
        pass
    return None


def search_online_images(query: str, max_images: int = 6) -> List[Tuple[Image.Image, str]]:
    """
    Search and download live online images matching query.
    Returns list of (PIL.Image, caption) tuples.
    """
    if not query or not query.strip():
        return []

    images_with_captions = []

    # 1. Unsplash API search
    try:
        url = "https://api.unsplash.com/search/photos"
        params = {"query": query, "per_page": max_images, "orientation": "landscape"}
        headers = {"Authorization": f"Client-ID {UNSPLASH_KEY}"}
        res = requests.get(url, params=params, headers=headers, timeout=6)

        if res.status_code == 200:
            photos = res.json().get("results", [])
            image_urls = [(p["urls"]["small"], p.get("alt_description") or p.get("description") or query) for p in photos]

            with ThreadPoolExecutor(max_workers=8) as executor:
                future_map = {
                    executor.submit(fetch_single_image, img_url): caption
                    for img_url, caption in image_urls
                }
                rank = 1
                for future in as_completed(future_map):
                    caption = future_map[future]
                    img = future.result()
                    if img is not None:
                        images_with_captions.append((img, f"🌐 Online #{rank} | {caption[:40]}"))
                        rank += 1
                        if len(images_with_captions) >= max_images:
                            break
    except Exception as e:
        logger.error(f"Unsplash online search error: {e}")

    # 2. Fallback Wikimedia Commons if needed
    if len(images_with_captions) < max_images:
        try:
            wiki_url = "https://commons.wikimedia.org/w/api.php"
            params = {
                "action": "query",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": max_images,
                "prop": "imageinfo",
                "iiprop": "url",
                "format": "json"
            }
            headers = {"User-Agent": "MultimodalSearchApp/1.0"}
            res = requests.get(wiki_url, params=params, headers=headers, timeout=6)
            if res.status_code == 200:
                pages = res.json().get("query", {}).get("pages", {})
                urls = []
                for p in pages.values():
                    ii = p.get("imageinfo", [])
                    if ii and "url" in ii[0]:
                        urls.append(ii[0]["url"])

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(fetch_single_image, u) for u in urls]
                    rank = len(images_with_captions) + 1
                    for f in futures:
                        img = f.result()
                        if img is not None:
                            images_with_captions.append((img, f"🌐 Online #{rank} | Wikimedia"))
                            rank += 1
                            if len(images_with_captions) >= max_images:
                                break
        except Exception as e:
            logger.error(f"Wikimedia online search error: {e}")

    return images_with_captions[:max_images]
