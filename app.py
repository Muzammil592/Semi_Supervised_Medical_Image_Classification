"""
Day 5 — Interactive Demo (Gradio)
=====================================
A small web app: upload a chest X-ray, pick which trained model to run
(baseline / pseudo-labeling / FixMatch / fully supervised), and see the
predicted class with a per-class confidence bar.

The model picker is the point of the demo, not just the prediction: it
lets a visitor directly compare how the SAME image is classified by a
model trained on 15% labeled data alone vs. one that also learned from
the unlabeled pool — which is the entire thesis of this project, made
tangible in one click.

Usage:
    python src/app.py --ckpt_dir outputs/checkpoints
Then open the printed local URL (add --share for a public Gradio link).
"""
import argparse
import functools
import os
import sys

import gradio as gr
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(__file__))
from dataset import get_transforms
from train_baseline import build_model

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

MODEL_CHOICES = {
    "Supervised baseline (15% labeled)": "baseline_best.pt",
    "Pseudo-labeling": "pseudolabel_best.pt",
    "FixMatch": "fixmatch_best.pt",
    "Fully supervised (100%, reference ceiling)": "full_supervised_best.pt",
}


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@functools.lru_cache(maxsize=None)
def load_model(ckpt_path: str, device_str: str):
    """Cached so switching back to a previously-used model in the UI is instant."""
    device = torch.device(device_str)
    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_predict_fn(ckpt_dir: str, img_size: int, device: torch.device):
    _, eval_tf = get_transforms(img_size)
    available = {
        label: os.path.join(ckpt_dir, fname)
        for label, fname in MODEL_CHOICES.items()
        if os.path.exists(os.path.join(ckpt_dir, fname))
    }
    if not available:
        raise SystemExit(
            f"No checkpoints found in {ckpt_dir}. Train at least one model "
            "(e.g. train_baseline.py) before launching the demo."
        )

    def predict(image, model_choice):
        if image is None:
            return None
        ckpt_path = available[model_choice]
        model = load_model(ckpt_path, str(device))

        img_t = eval_tf(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(img_t)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu()

        return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

    return predict, list(available.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="outputs/checkpoints")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")
    predict, available_models = build_predict_fn(args.ckpt_dir, args.img_size, device)
    print(f"Loaded models available in demo: {available_models}")

    with gr.Blocks(title="Semi-Supervised Chest X-Ray Classifier") as demo:
        gr.Markdown(
            "# Semi-Supervised Chest X-Ray Classifier\n"
            "Upload a chest X-ray and pick which trained model to run. "
            "Compare a model trained on only **15% labeled data** against "
            "models that also learned from the **70% unlabeled pool** via "
            "pseudo-labeling or FixMatch — and against a fully-supervised "
            "reference ceiling trained on 100% labeled data.\n\n"
            "*Research/portfolio demo only — not a diagnostic tool.*"
        )
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Chest X-Ray")
                model_dropdown = gr.Dropdown(
                    choices=available_models,
                    value=available_models[0],
                    label="Model",
                )
                predict_btn = gr.Button("Classify", variant="primary")
            with gr.Column():
                label_output = gr.Label(num_top_classes=2, label="Prediction Confidence")

        predict_btn.click(
            fn=predict, inputs=[image_input, model_dropdown], outputs=label_output
        )
        # Re-run automatically when switching models on an already-uploaded image
        model_dropdown.change(
            fn=predict, inputs=[image_input, model_dropdown], outputs=label_output
        )

    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
