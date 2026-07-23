"""
Gradio UI — Multimodal Search Engine
Full-featured interactive web interface with modern dark theme, speed optimizations,
and dynamic search source selector (Local Database vs Online Google/DDG).
Run with: python frontend/app.py
"""

import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from typing import List, Tuple, Dict
from PIL import Image
import gradio as gr

from src.model.search_engine import search_engine
from src.utils.google_scraper import search_online_images
from src.utils.config import settings
from src.utils.logger import logger


def load_engine():
    if not search_engine.is_ready:
        search_engine.load()


def text_search(query: str, top_k: int, source: str = "Local Database") -> Tuple[List, str]:
    """Handle text → image search with timing stats and dynamic source selection."""
    if not query or not query.strip():
        return [], "⚠️ Please enter a search query."

    t0 = time.perf_counter()
    try:
        if source == "Online (Google/DDG)":
            online_results = search_online_images(query=query.strip(), max_images=int(top_k))
            total_time = (time.perf_counter() - t0) * 1000
            info = (
                f"🌐 Source: Online (Google/DDG / Unsplash)  |  "
                f"⚡ Total Fetch Time: {total_time:.1f}ms  |  "
                f"📊 Downloaded Images: {len(online_results)}"
            )
            return online_results, info

        # Local Database FAISS search
        result = search_engine.search_by_text(query=query.strip(), top_k=int(top_k))
        total_time = (time.perf_counter() - t0) * 1000

        images = []
        for r in result["results"]:
            try:
                img = Image.open(r["path"])
                images.append((img, f"#{r['rank']} | Score: {r['score']:.4f} | {r['filename']}"))
            except Exception:
                pass

        info = (
            f"📁 Source: Local Vector Database  |  "
            f"⚡ Total Time: {total_time:.1f}ms  |  "
            f"🧠 CLIP Encode: {result['encode_ms']:.1f}ms  |  "
            f"🔍 FAISS Search: {result['search_ms']:.2f}ms  |  "
            f"📊 Results: {result['total_results']}"
        )
        return images, info
    except Exception as e:
        return [], f"❌ Error: {str(e)}"


# Full-sentence CLIP prompts → CLIP was trained on descriptive sentences, NOT single words.
# Using "a photo of X" format gives far more accurate classification than bare keywords.
# Each tuple is (search_query_for_unsplash, clip_prompt)
_CLIP_PROMPT_MAP = [
    # Medical & Symbols
    ("caduceus medical symbol",         "a photo of a caduceus medical symbol with snakes and a staff"),
    ("medical cross symbol",            "a red or blue medical cross symbol or healthcare logo"),
    ("hospital",                        "a photo of a hospital building or emergency room"),
    ("stethoscope doctor",              "a photo of a doctor holding a stethoscope"),
    ("dna science",                     "a photo of DNA double helix or scientific diagram"),
    # Logos, Icons, Symbols
    ("logo icon emblem",                "a flat icon logo or emblem on a white or dark background"),
    ("abstract symbol",                 "an abstract geometric symbol or badge illustration"),
    # Animals
    ("dog",                             "a photo of a dog"),
    ("cat",                             "a photo of a cat"),
    ("bird",                            "a photo of a bird flying or perched"),
    ("horse",                           "a photo of a horse in a field"),
    ("lion",                            "a photo of a lion in the wild"),
    ("elephant",                        "a photo of an elephant in nature"),
    ("butterfly",                       "a photo of a colorful butterfly"),
    ("fish underwater",                 "a photo of fish underwater"),
    ("bear",                            "a photo of a bear in the wild"),
    ("deer forest",                     "a photo of a deer in the forest"),
    ("wolf",                            "a photo of a wolf"),
    ("penguin",                         "a photo of penguins"),
    ("dolphin ocean",                   "a photo of dolphins jumping in the ocean"),
    # Vehicles
    ("car",                             "a photo of a car on the road"),
    ("airplane flying",                 "a photo of an airplane in the sky"),
    ("motorcycle",                      "a photo of a motorcycle"),
    ("train railway",                   "a photo of a train on railway tracks"),
    ("ship ocean",                      "a photo of a large ship on the ocean"),
    ("bicycle",                         "a photo of a bicycle"),
    ("helicopter",                      "a photo of a helicopter in the sky"),
    ("rocket launch",                   "a photo of a rocket launching into space"),
    # People
    ("portrait person",                 "a portrait photo of a person's face"),
    ("baby child",                      "a photo of a baby or young child"),
    ("wedding couple",                  "a photo of a wedding ceremony or couple"),
    ("athlete sports",                  "a photo of an athlete playing sports"),
    ("musician performing",             "a photo of a musician performing on stage"),
    ("soldier military",                "a photo of a soldier in military uniform"),
    ("yoga meditation",                 "a photo of someone doing yoga or meditation"),
    ("chef cooking",                    "a photo of someone cooking in a kitchen"),
    # Nature & Landscapes
    ("mountain landscape",              "a photo of a mountain landscape"),
    ("beach ocean sunset",              "a photo of a beach with waves and sunset"),
    ("forest trees",                    "a photo of a dense green forest"),
    ("desert sand dunes",               "a photo of a desert with sand dunes"),
    ("sunset sky",                      "a photo of a colorful sunset sky"),
    ("ocean waves",                     "a photo of ocean waves"),
    ("river waterfall",                 "a photo of a river or waterfall"),
    ("snow winter",                     "a photo of a snowy winter landscape"),
    ("volcano eruption",                "a photo of a volcano erupting"),
    ("aurora borealis",                 "a photo of the northern lights aurora borealis"),
    ("lightning storm",                 "a photo of lightning in a storm"),
    # Cities & Architecture
    ("city skyline night",              "a photo of a city skyline at night"),
    ("skyscraper building",             "a photo of tall skyscrapers and buildings"),
    ("bridge architecture",             "a photo of a bridge over water"),
    ("castle medieval",                 "a photo of a medieval castle"),
    ("church cathedral",                "a photo of a church or cathedral"),
    ("mosque temple",                   "a photo of a mosque or temple"),
    ("lighthouse coast",                "a photo of a lighthouse on the coast"),
    # Food
    ("pizza food",                      "a photo of a delicious pizza"),
    ("fresh fruit",                     "a photo of colorful fresh fruits"),
    ("coffee cafe",                     "a photo of a cup of coffee"),
    ("sushi japanese food",             "a photo of sushi"),
    ("cake dessert",                    "a photo of a cake or dessert"),
    ("burger fast food",                "a photo of a burger"),
    # Technology
    ("laptop computer",                 "a photo of a laptop or computer"),
    ("smartphone mobile",               "a photo of a smartphone"),
    ("robot technology",                "a photo of a robot or futuristic technology"),
    ("circuit board electronics",       "a photo of a circuit board or electronic components"),
    ("drone aerial",                    "a photo of a drone flying"),
    ("camera photography",              "a photo of a camera"),
    # Art & Creative
    ("painting artwork",                "a photo of a painting or artwork"),
    ("graffiti street art",             "a photo of graffiti or street art"),
    ("sculpture statue",                "a photo of a sculpture or statue"),
    ("abstract art colorful",           "a photo of abstract colorful art"),
    ("cartoon illustration",            "a cartoon or digital illustration"),
    # Space
    ("galaxy stars space",              "a photo of a galaxy or stars in space"),
    ("planet earth space",              "a photo of a planet from space"),
    ("nebula cosmos",                   "a photo of a colorful nebula in space"),
    ("astronaut space",                 "a photo of an astronaut in space"),
    # Plants
    ("flower garden",                   "a photo of colorful flowers in a garden"),
    ("tree forest nature",              "a photo of a large tree in nature"),
    ("mushroom fungi",                  "a photo of mushrooms"),
    ("cactus desert",                   "a photo of a cactus in the desert"),
    # Sports
    ("football soccer",                 "a photo of people playing football or soccer"),
    ("basketball game",                 "a photo of a basketball game"),
    ("tennis player",                   "a photo of a tennis player"),
    ("surfing waves",                   "a photo of someone surfing big waves"),
    ("golf course",                     "a photo of a golf course"),
    ("swimming pool",                   "a photo of a swimming pool"),
]

# Separate lists for classification (prompts) and query lookup
_CLIP_PROMPTS = [p for _, p in _CLIP_PROMPT_MAP]
_CLIP_QUERIES = [q for q, _ in _CLIP_PROMPT_MAP]


def _classify_with_prompts(search_engine_obj, image: Image.Image) -> tuple:
    """
    Classify image using full-sentence CLIP prompts for maximum accuracy.
    Returns (search_query, confidence_score, detected_description).
    """
    # Use the engine's encode_image + batch text encode directly
    from src.model.clip_encoder import encoder
    import numpy as np

    img_vec = encoder.encode_image(image)
    label_vecs = encoder.encode_texts_batch(_CLIP_PROMPTS)

    scores = {}
    for i, (query, prompt) in enumerate(_CLIP_PROMPT_MAP):
        similarity = float(np.dot(img_vec.flatten(), label_vecs[i].flatten()))
        scores[query] = (similarity, prompt)

    # Sort by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
    best_query, (best_score, best_prompt) = sorted_scores[0]
    return best_query, best_score, best_prompt


def image_search(query_image: Image.Image, top_k: int, source: str = "Local Database") -> Tuple[List, str]:
    """Handle image → image search with timing stats and dynamic source selection."""
    if query_image is None:
        return [], "⚠️ Please upload an image."

    t0 = time.perf_counter()
    try:
        if source == "Online (Google/DDG)":
            # Use full-sentence CLIP prompts for accurate image content detection.
            # CLIP was trained on descriptive sentences — this correctly identifies
            # logos, medical symbols, animals, etc. that single words would miss.
            query_tag, top_score, detected_prompt = _classify_with_prompts(search_engine, query_image)
            online_results = search_online_images(query=query_tag, max_images=int(top_k))
            total_time = (time.perf_counter() - t0) * 1000
            info = (
                f"🌐 Source: Online (Google/DDG)  |  "
                f"🏷️ Auto-Detected: '{query_tag}' ({top_score*100:.1f}% conf)  |  "
                f"⚡ Total Time: {total_time:.1f}ms  |  "
                f"📊 Results: {len(online_results)}"
            )
            return online_results, info

        # Local Database FAISS search
        result = search_engine.search_by_image(image=query_image, top_k=int(top_k))
        total_time = (time.perf_counter() - t0) * 1000

        images = []
        for r in result["results"]:
            try:
                img = Image.open(r["path"])
                images.append((img, f"#{r['rank']} | Score: {r['score']:.4f} | {r['filename']}"))
            except Exception:
                pass

        info = (
            f"📁 Source: Local Vector Database  |  "
            f"⚡ Total Time: {total_time:.1f}ms  |  "
            f"🧠 CLIP Encode: {result['encode_ms']:.1f}ms  |  "
            f"🔍 FAISS Search: {result['search_ms']:.2f}ms  |  "
            f"📊 Results: {result['total_results']}"
        )
        return images, info
    except Exception as e:
        return [], f"❌ Error: {str(e)}"


def hybrid_search_fn(
    query_image: Image.Image,
    text_query: str,
    top_k: int,
    text_weight: float,
    source: str = "Local Database",
) -> Tuple[List, str]:
    """Handle hybrid search with timing stats and dynamic source selection."""
    if query_image is None or not text_query or not text_query.strip():
        return [], "⚠️ Please upload an image AND enter a text query."

    t0 = time.perf_counter()
    try:
        if source == "Online (Google/DDG)":
            online_results = search_online_images(query=text_query.strip(), max_images=int(top_k))
            total_time = (time.perf_counter() - t0) * 1000
            info = (
                f"🌐 Source: Online (Google/DDG)  |  "
                f"⚡ Fetch Time: {total_time:.1f}ms  |  "
                f"📊 Results: {len(online_results)}"
            )
            return online_results, info

        # Local Database FAISS search
        result = search_engine.hybrid_search(
            text_query=text_query.strip(),
            image=query_image,
            text_weight=text_weight,
            top_k=int(top_k),
        )
        total_time = (time.perf_counter() - t0) * 1000

        images = []
        for r in result["results"]:
            try:
                img = Image.open(r["path"])
                images.append((img, f"#{r['rank']} | Score: {r['score']:.4f} | {r['filename']}"))
            except Exception:
                pass

        info = (
            f"📁 Source: Local Vector Database  |  "
            f"⚡ Total Time: {total_time:.1f}ms  |  "
            f"⚖️ Text Weight: {text_weight:.1f}  |  "
            f"🖼️ Image Weight: {result['image_weight']:.1f}  |  "
            f"🔍 FAISS Search: {result['search_ms']:.2f}ms"
        )
        return images, info
    except Exception as e:
        return [], f"❌ Error: {str(e)}"


def classify_fn(image: Image.Image, labels_str: str) -> str:
    """Handle zero-shot classification with progress bar output."""
    if image is None:
        return "⚠️ Please upload an image."
    if not labels_str or not labels_str.strip():
        return "⚠️ Please enter at least one category label."

    try:
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]
        result = search_engine.classify(image=image, labels=labels)

        output = f"🎯 TOP PREDICTION:  {result['top_label'].upper()}  (Confidence: {result['confidence'] * 100:.1f}%)\n"
        output += f"⏱️ CLIP Inference Time: {result['encode_ms']:.1f}ms\n"
        output += "=" * 55 + "\n\n"
        output += "CATEGORY BREAKDOWN & SIMILARITY SCORES:\n"
        output += "-" * 55 + "\n"

        for label, score in result["scores"].items():
            pct = score * 100
            bar_len = int(score * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            output += f"{label:<18} | {bar} | {pct:5.1f}%\n"

        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_index_info() -> str:
    stats = search_engine.get_stats()
    if stats.get("status") != "ready":
        return "⚠️ Index not loaded."
    return (
        f"⚡ Status: READY\n"
        f"🖼️ Indexed Images: {stats['total_images']:,}\n"
        f"📐 Embedding Dims: {stats['embedding_dim']}\n"
        f"🧠 Model: {stats['model']}\n"
        f"💻 Inference Device: {stats['device'].upper()}"
    )


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

/* Force dark mode color scheme & Plus Jakarta Sans typography globally */
* {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

:root {
  --block-label-background-fill: #1e293b !important;
  --block-label-text-color: #38bdf8 !important;
  --block-title-background-fill: #1e293b !important;
  --block-title-text-color: #38bdf8 !important;
  --body-text-color: #f8fafc !important;
  --subdued-text-color: #94a3b8 !important;
  --radio-circle: #f97316 !important;
}

:root, html, body, div, span, button, input, textarea, select {
    color-scheme: dark !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

body, html, .gradio-container {
    background-color: #0b0f19 !important;
    color: #f8fafc !important;
}

/* Card containers & blocks */
.block, .panel, .form, fieldset, div[class*="block"] {
    background-color: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4) !important;
}

/* Text Inputs, Textareas, and Sliders */
input, textarea, .gr-input, input[type="text"], input[type="number"] {
    background-color: #000000 !important;
    color: #ffffff !important;
    border: 1.5px solid #334155 !important;
    border-radius: 8px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
}

input:focus, textarea:focus {
    border-color: #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25) !important;
}

input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
    font-weight: 400 !important;
}

/* Block Labels & Headers — Force Dark Blue Pill & Bright Cyan Text */
.block-label, label.block-label, span.block-label, div.block-label, legend.block-label,
label[data-testid="block-label"], .block span.block-label, fieldset legend, .block label {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    border-radius: 6px !important;
    padding: 4px 10px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    border: 1px solid #334155 !important;
}

/* Force child elements of block labels (icons/spans) to be cyan on dark blue */
.block-label *, label.block-label *, span.block-label * {
    color: #38bdf8 !important;
    background-color: transparent !important;
}

/* Radio buttons container styling */
.gr-radio, fieldset[class*="radio"], div[class*="radio"], .radio-group {
    background-color: #111827 !important;
    border: 1px solid #1e293b !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
}

/* Radio option labels — High contrast white text */
.gr-radio label, fieldset label, label.gr-radio, label[class*="radio"],
.gr-radio span, div[class*="radio"] label span, label input[type="radio"] + span {
    background-color: transparent !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 14px !important;
}

input[type="radio"] {
    accent-color: #f97316 !important;
}

/* Tab Headers */
.tabs, div.tabs, .tab-nav, div.tab-nav {
    background-color: #0f172a !important;
    border-bottom: 2px solid #1e293b !important;
    padding: 6px 6px 0 6px !important;
}

button[role="tab"], .tab-nav button, button.tab-nav {
    color: #cbd5e1 !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    background-color: #1e293b !important;
    margin-right: 8px !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    border: 1px solid #334155 !important;
    border-bottom: none !important;
}

button[role="tab"][aria-selected="true"], .tab-nav button.selected, button.selected {
    color: #ffffff !important;
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    border-color: #f97316 !important;
    box-shadow: 0 -2px 10px rgba(249, 115, 22, 0.3) !important;
}

/* Action Buttons */
button.primary, button.lg.primary, .gr-button-primary {
    background: linear-gradient(135deg, #f97316 0%, #d97706 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    box-shadow: 0 4px 15px rgba(249, 115, 22, 0.4) !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #ea580c 0%, #b45309 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(249, 115, 22, 0.6) !important;
}

/* Secondary & Chip Buttons */
button.secondary, button.chip-btn {
    background-color: #1e293b !important;
    color: #38bdf8 !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 6px 14px !important;
}

button.secondary:hover, button.chip-btn:hover {
    background-color: #334155 !important;
    color: #ffffff !important;
    border-color: #38bdf8 !important;
}

/* Markdown Text & Typography */
p, span, h1, h2, h3, h4, .markdown {
    color: #f8fafc !important;
}

h1 {
    color: #fb923c !important;
    font-weight: 800 !important;
    font-size: 2.2rem !important;
    margin-bottom: 4px !important;
}

/* Examples Tables */
.examples, table, tr, td, th {
    background-color: #111827 !important;
    color: #f1f5f9 !important;
    border-color: #1e293b !important;
}

td, th {
    color: #ffffff !important;
    font-weight: 600 !important;
    padding: 10px !important;
}

/* Output Code & Pre blocks */
pre, code {
    background-color: #000000 !important;
    color: #38bdf8 !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}
"""


def create_app():
    load_engine()

    with gr.Blocks(title="⚡ Multimodal AI Search Engine") as demo:

        gr.Markdown("# ⚡ Multimodal AI Search Engine")
        gr.Markdown(
            "**Powered by OpenAI CLIP & FAISS Vector Indexing** — High-speed vector search for natural language text, reverse image search, hybrid queries, and zero-shot visual classification."
        )

        with gr.Row():
            with gr.Column(scale=3):
                pass
            with gr.Column(scale=1):
                index_info = gr.Textbox(
                    label="⚙️ Vector Database Status",
                    value=get_index_info,
                    interactive=False,
                    lines=5,
                )

        with gr.Tabs():

            # ── Tab 1: Text → Images ──────────────────────────────────────
            with gr.TabItem("💬 Text → Images"):
                gr.Markdown("### Search images using natural language descriptions")
                with gr.Row():
                    with gr.Column(scale=3):
                        text_input = gr.Textbox(
                            label="🔍 Search Query",
                            placeholder="e.g. a golden retriever playing in the park at sunset",
                            lines=2,
                        )
                        gr.Markdown("**⚡ Quick Sample Queries:**")
                        with gr.Row():
                            chip1 = gr.Button("🐕 Dog in snow", variant="secondary", elem_classes=["chip-btn"])
                            chip2 = gr.Button("🌆 City lights at night", variant="secondary", elem_classes=["chip-btn"])
                            chip3 = gr.Button("🏔️ Mountain landscape", variant="secondary", elem_classes=["chip-btn"])
                            chip4 = gr.Button("🏎️ Vintage sports car", variant="secondary", elem_classes=["chip-btn"])

                    with gr.Column(scale=2):
                        source_text = gr.Radio(
                            choices=["Local Database", "Online (Google/DDG)"],
                            value="Local Database",
                            label="🌐 Search Source",
                        )
                        top_k_text = gr.Slider(
                            minimum=1, maximum=20, value=6, step=1,
                            label="📊 Number of Results",
                        )
                        search_btn1 = gr.Button("🔍 Search Images", variant="primary")

                info_text = gr.Textbox(label="⏱️ Real-Time Performance & Vector Search Metrics", interactive=False)
                gallery_text = gr.Gallery(
                    label="🖼️ Matching Image Results",
                    columns=3,
                    height=520,
                    object_fit="contain",
                )

                # Quick chip event handlers
                chip1.click(lambda: ("a dog playing in the snow", 6, "Local Database"), outputs=[text_input, top_k_text, source_text]).then(text_search, inputs=[text_input, top_k_text, source_text], outputs=[gallery_text, info_text])
                chip2.click(lambda: ("city skyline at night with colorful lights", 6, "Local Database"), outputs=[text_input, top_k_text, source_text]).then(text_search, inputs=[text_input, top_k_text, source_text], outputs=[gallery_text, info_text])
                chip3.click(lambda: ("person hiking in mountains with backpack", 6, "Local Database"), outputs=[text_input, top_k_text, source_text]).then(text_search, inputs=[text_input, top_k_text, source_text], outputs=[gallery_text, info_text])
                chip4.click(lambda: ("vintage sports car on an empty road", 6, "Local Database"), outputs=[text_input, top_k_text, source_text]).then(text_search, inputs=[text_input, top_k_text, source_text], outputs=[gallery_text, info_text])

                search_btn1.click(
                    text_search,
                    inputs=[text_input, top_k_text, source_text],
                    outputs=[gallery_text, info_text],
                )

            # ── Tab 2: Image → Similar Images ──────────────────────────────
            with gr.TabItem("🖼️ Image → Visual Search"):
                gr.Markdown("### Upload an image to perform reverse vector search for visually similar photos")
                with gr.Row():
                    with gr.Column(scale=2):
                        image_input = gr.Image(
                            label="📷 Reference Query Image",
                            type="pil",
                            height=320,
                        )
                    with gr.Column(scale=2):
                        source_img = gr.Radio(
                            choices=["Local Database", "Online (Google/DDG)"],
                            value="Local Database",
                            label="🌐 Search Source",
                        )
                        top_k_img = gr.Slider(
                            minimum=1, maximum=20, value=6, step=1,
                            label="📊 Number of Results",
                        )
                        search_btn2 = gr.Button("⚡ Find Visually Similar", variant="primary")

                info_img = gr.Textbox(label="⏱️ Real-Time Performance & Vector Search Metrics", interactive=False)
                gallery_img = gr.Gallery(
                    label="🖼️ Visually Similar Results",
                    columns=3,
                    height=520,
                    object_fit="contain",
                )

                search_btn2.click(
                    image_search,
                    inputs=[image_input, top_k_img, source_img],
                    outputs=[gallery_img, info_img],
                )

            # ── Tab 3: Hybrid Search ──────────────────────────────────────
            with gr.TabItem("🔀 Hybrid Search"):
                gr.Markdown("### Combine reference image + text query refinement for dual-modal precision")
                with gr.Row():
                    with gr.Column(scale=2):
                        hybrid_image = gr.Image(
                            label="📷 Base Reference Image",
                            type="pil",
                            height=260,
                        )
                    with gr.Column(scale=2):
                        hybrid_text = gr.Textbox(
                            label="✍️ Refine Description with Text",
                            placeholder="e.g. blue leather texture or snowy background",
                            lines=2,
                        )
                        source_hybrid = gr.Radio(
                            choices=["Local Database", "Online (Google/DDG)"],
                            value="Local Database",
                            label="🌐 Search Source",
                        )
                        text_weight_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="⚖️ Text Weight (0.0 = pure image, 1.0 = pure text)",
                        )
                        top_k_hybrid = gr.Slider(
                            minimum=1, maximum=20, value=6, step=1,
                            label="📊 Number of Results",
                        )
                        search_btn3 = gr.Button("⚡ Execute Hybrid Search", variant="primary")

                info_hybrid = gr.Textbox(label="⏱️ Real-Time Performance & Hybrid Weight Metrics", interactive=False)
                gallery_hybrid = gr.Gallery(
                    label="🖼️ Hybrid Search Results",
                    columns=3,
                    height=520,
                    object_fit="contain",
                )

                search_btn3.click(
                    hybrid_search_fn,
                    inputs=[hybrid_image, hybrid_text, top_k_hybrid, text_weight_slider, source_hybrid],
                    outputs=[gallery_hybrid, info_hybrid],
                )

            # ── Tab 4: Zero-shot Classification ───────────────────────────
            with gr.TabItem("🔍 Zero-Shot Classify"):
                gr.Markdown("### Classify any image into arbitrary custom categories — zero training required")
                with gr.Row():
                    with gr.Column(scale=2):
                        classify_image_input = gr.Image(
                            label="📷 Target Image",
                            type="pil",
                            height=320,
                        )
                    with gr.Column(scale=2):
                        classify_labels = gr.Textbox(
                            label="🏷️ Categories (comma-separated)",
                            value="cat, dog, car, person, food, building, nature, sports",
                            lines=3,
                        )
                        classify_btn = gr.Button("🎯 Run Zero-Shot Classifier", variant="primary")
                        classify_output = gr.Textbox(
                            label="📊 Probability Breakdown & Confidence Scores",
                            lines=12,
                            interactive=False,
                        )

                classify_btn.click(
                    classify_fn,
                    inputs=[classify_image_input, classify_labels],
                    outputs=[classify_output],
                )

    return demo


if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 7860))
    is_cloud = "PORT" in os.environ or "RENDER" in os.environ
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(),
        share=not is_cloud,
        show_error=True,
    )

