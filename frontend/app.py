"""
Gradio UI — Multimodal Search Engine
Full-featured web interface with three search modes.
Run with: python frontend/app.py
"""

from pickle import TRUE
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from PIL import Image
from typing import List, Tuple

from src.model.search_engine import search_engine
from src.utils.config import settings
from src.utils.logger import logger


def load_engine():
    if not search_engine.is_ready:
        search_engine.load()


def text_search(query: str, top_k: int) -> Tuple[List, str]:
    """Handle text → image search from UI."""
    if not query.strip():
        return [], "Please enter a search query."

    try:
        result = search_engine.search_by_text(query=query, top_k=int(top_k))
        images = []
        for r in result["results"]:
            try:
                img = Image.open(r["path"])
                images.append((img, f"#{r['rank']} Score: {r['score']:.3f}"))
            except Exception:
                pass

        info = (
            f"Found {result['total_results']} results | "
            f"Encode: {result['encode_ms']}ms | "
            f"Search: {result['search_ms']}ms"
        )
        return images, info
    except Exception as e:
        return [], f"Error: {str(e)}"


def image_search(query_image: Image.Image, top_k: int) -> Tuple[List, str]:
    """Handle image → image search from UI."""
    if query_image is None:
        return [], "Please upload an image."

    try:
        result = search_engine.search_by_image(image=query_image, top_k=int(top_k))
        images = []
        for r in result["results"]:
            try:
                img = Image.open(r["path"])
                images.append((img, f"#{r['rank']} Score: {r['score']:.3f}"))
            except Exception:
                pass

        info = (
            f"Found {result['total_results']} results | "
            f"Encode: {result['encode_ms']}ms | "
            f"Search: {result['search_ms']}ms"
        )
        return images, info
    except Exception as e:
        return [], f"Error: {str(e)}"


def hybrid_search_fn(
    query_image: Image.Image,
    text_query: str,
    top_k: int,
    text_weight: float,
) -> Tuple[List, str]:
    """Handle hybrid image+text search from UI."""
    if query_image is None or not text_query.strip():
        return [], "Please upload an image AND enter a text query."

    try:
        result = search_engine.hybrid_search(
            text_query=text_query,
            image=query_image,
            text_weight=text_weight,
            top_k=int(top_k),
        )
        images = []
        for r in result["results"]:
            try:
                img = Image.open(r["path"])
                images.append((img, f"#{r['rank']} Score: {r['score']:.3f}"))
            except Exception:
                pass

        info = (
            f"Found {result['total_results']} results | "
            f"Text weight: {text_weight:.1f} | Image weight: {result['image_weight']:.1f} | "
            f"Search: {result['search_ms']}ms"
        )
        return images, info
    except Exception as e:
        return [], f"Error: {str(e)}"


def classify_fn(image: Image.Image, labels_str: str) -> str:
    """Handle zero-shot classification from UI."""
    if image is None:
        return "Please upload an image."
    if not labels_str.strip():
        return "Please enter at least one label."

    try:
        labels = [l.strip() for l in labels_str.split(",") if l.strip()]
        result = search_engine.classify(image=image, labels=labels)

        output = f"Top prediction: **{result['top_label']}** (confidence: {result['confidence']:.3f})\n\n"
        output += "All scores:\n"
        for label, score in result["scores"].items():
            bar = "█" * int(score * 40)
            output += f"{label:<20} {score:.4f}  {bar}\n"
        return output
    except Exception as e:
        return f"Error: {str(e)}"


def get_index_info() -> str:
    stats = search_engine.get_stats()
    if stats.get("status") != "ready":
        return "Index not loaded. Run: python scripts/build_index.py"
    return (
        f"Model: {stats['model']}\n"
        f"Images indexed: {stats['total_images']:,}\n"
        f"Embedding dims: {stats['embedding_dim']}\n"
        f"Index type: {stats['index_type']}\n"
        f"Device: {stats['device']}"
    )


# ── Build Gradio App ──────────────────────────────────────────────────────
def create_app():
    load_engine()

    with gr.Blocks(
        title="Multimodal Search Engine",
        theme=gr.themes.Soft(),
        css=".gr-button { font-weight: 500; }",
    ) as demo:

        gr.Markdown("# Multimodal Search Engine")
        gr.Markdown("**CLIP + FAISS** — Search images with text, find similar images, or classify with custom labels.")

        with gr.Row():
            with gr.Column(scale=3):
                pass
            with gr.Column(scale=1):
                index_info = gr.Textbox(
                    label="Index Info",
                    value=get_index_info,
                    interactive=False,
                    lines=5,
                )

        with gr.Tabs():

            # ── Tab 1: Text Search ──────────────────────────────────
            with gr.TabItem("Text → Images"):
                gr.Markdown("Describe what you're looking for in natural language.")
                with gr.Row():
                    with gr.Column(scale=4):
                        text_input = gr.Textbox(
                            label="Search query",
                            placeholder="a golden retriever playing in the park",
                            lines=2,
                        )
                    with gr.Column(scale=1):
                        top_k_text = gr.Slider(
                            minimum=1, maximum=20, value=6, step=1,
                            label="Number of results",
                        )

                search_btn1 = gr.Button("Search", variant="primary")
                info_text = gr.Textbox(label="Search info", interactive=False)
                gallery_text = gr.Gallery(
                    label="Results",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )

                gr.Examples(
                    examples=[
                        ["a dog playing in the snow", 6],
                        ["city skyline at night with colorful lights", 6],
                        ["person hiking in mountains with backpack", 6],
                        ["fresh fruits on a wooden table", 6],
                        ["vintage car on an empty road", 6],
                    ],
                    inputs=[text_input, top_k_text],
                )

                search_btn1.click(
                    text_search,
                    inputs=[text_input, top_k_text],
                    outputs=[gallery_text, info_text],
                )

            # ── Tab 2: Image Search ─────────────────────────────────
            with gr.TabItem("Image → Similar Images"):
                gr.Markdown("Upload an image to find visually similar ones in the index.")
                with gr.Row():
                    with gr.Column():
                        image_input = gr.Image(
                            label="Query image",
                            type="pil",
                            height=300,
                        )
                        top_k_img = gr.Slider(
                            minimum=1, maximum=20, value=6, step=1,
                            label="Number of results",
                        )
                        search_btn2 = gr.Button("Find Similar", variant="primary")

                info_img = gr.Textbox(label="Search info", interactive=False)
                gallery_img = gr.Gallery(
                    label="Similar images",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )

                search_btn2.click(
                    image_search,
                    inputs=[image_input, top_k_img],
                    outputs=[gallery_img, info_img],
                )

            # ── Tab 3: Hybrid Search ────────────────────────────────
            with gr.TabItem("Hybrid (Image + Text)"):
                gr.Markdown(
                    "Combine an image + text for more precise results. "
                    "Example: upload a shoe + type 'blue leather' to find blue leather versions."
                )
                with gr.Row():
                    with gr.Column():
                        hybrid_image = gr.Image(
                            label="Reference image",
                            type="pil",
                            height=250,
                        )
                    with gr.Column():
                        hybrid_text = gr.Textbox(
                            label="Refine with text",
                            placeholder="blue leather",
                            lines=2,
                        )
                        text_weight_slider = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.1,
                            label="Text weight (0 = pure image, 1 = pure text)",
                        )
                        top_k_hybrid = gr.Slider(
                            minimum=1, maximum=20, value=6, step=1,
                            label="Number of results",
                        )
                        search_btn3 = gr.Button("Search", variant="primary")

                info_hybrid = gr.Textbox(label="Search info", interactive=False)
                gallery_hybrid = gr.Gallery(
                    label="Results",
                    columns=3,
                    height=500,
                    object_fit="contain",
                )

                search_btn3.click(
                    hybrid_search_fn,
                    inputs=[hybrid_image, hybrid_text, top_k_hybrid, text_weight_slider],
                    outputs=[gallery_hybrid, info_hybrid],
                )

            # ── Tab 4: Zero-shot Classification ────────────────────
            with gr.TabItem("Zero-shot Classify"):
                gr.Markdown(
                    "Classify any image into custom categories — no training needed. "
                    "Works because CLIP's image and text encoders share the same embedding space."
                )
                with gr.Row():
                    with gr.Column():
                        classify_image_input = gr.Image(
                            label="Image to classify",
                            type="pil",
                            height=300,
                        )
                    with gr.Column():
                        classify_labels = gr.Textbox(
                            label="Labels (comma-separated)",
                            value="cat, dog, car, person, food, building, nature, animal",
                            lines=3,
                        )
                        classify_btn = gr.Button("Classify", variant="primary")
                        classify_output = gr.Textbox(
                            label="Classification results",
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
    app.launch(
        server_port=7860,
        share=True,
        show_error=True,
    )
