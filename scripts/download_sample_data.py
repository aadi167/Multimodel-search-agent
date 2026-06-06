"""
Downloads images from Unsplash API by search keywords.
Free: 50 requests/hour | High quality | Categorized
"""

import os
import requests
from tqdm import tqdm

UNSPLASH_ACCESS_KEY = "F5iycODjTNyxFZqdV9PtOxnucJ2iU2zTo6dBEnhZHf4"

SEARCH_CATEGORIES = [
    "dog", "cat", "car", "food", "mountain",
    "beach", "city", "person", "sports", "nature",
    "fashion", "shoes", "laptop", "flower", "sunset"
]

IMAGES_PER_CATEGORY = 10


def download_from_unsplash():
    os.makedirs("data/images", exist_ok=True)
    total_downloaded = 0

    for category in SEARCH_CATEGORIES:
        print(f"\nDownloading: {category}")
        url = "https://api.unsplash.com/search/photos"
        params = {
            "query": category,
            "per_page": IMAGES_PER_CATEGORY,
            "orientation": "landscape",
        }
        headers = {"Authorization": f"Client-ID {UNSPLASH_ACCESS_KEY}"}

        try:
            response = requests.get(url, params=params,
                                    headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"  Error {response.status_code}: {response.text}")
                continue

            photos = response.json().get("results", [])
            for i, photo in enumerate(tqdm(photos, desc=f"  {category}")):
                filename = f"{category}_{i+1:02d}_{photo['id'][:6]}.jpg"
                save_path = os.path.join("data/images", filename)

                if os.path.exists(save_path):
                    total_downloaded += 1
                    continue

                img_url = photo["urls"]["small"]
                img_response = requests.get(img_url, timeout=15)

                if img_response.status_code == 200:
                    with open(save_path, "wb") as f:
                        f.write(img_response.content)
                    total_downloaded += 1

        except Exception as e:
            print(f"  Failed for {category}: {e}")

    total_files = len([f for f in os.listdir("data/images")
                       if f.endswith(".jpg")])
    print(f"\nDone! {total_files} images saved to data/images/")
    print("Next step: python scripts/build_index.py")


if __name__ == "__main__":
    download_from_unsplash()