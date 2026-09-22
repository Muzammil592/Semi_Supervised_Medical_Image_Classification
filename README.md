# Semi-Supervised Medical Image Classification via Pseudo-Labeling & FixMatch

Training a strong image classifier usually assumes you have thousands of
**labeled** examples. In real deployments especially medical imaging
that assumption breaks: labeling a chest X-ray requires a radiologist's
time, and that time is expensive and scarce. This project asks a more
realistic question:

> **How much of that labeling cost can we recover using techniques that
> exploit unlabeled data instead?**

It builds an end-to-end semi-supervised learning (SSL) pipeline that trains
on **15% labeled data + 70% unlabeled data**, and benchmarks two SSL
techniques pseudo-labeling and FixMatch against a supervised-only
baseline on the same small labeled slice.

---

## Status

| Stage | Description | Status |
|---|---|---|
| 1 | Data preparation & supervised baseline | ✅ Implemented |
| 2 | Pseudo-labeling self-training | ✅ Implemented |
| 3 | FixMatch (consistency regularization) | ✅ Implemented |
| 4 | Benchmarking, ablations, embedding visualizations | ✅ Implemented |
| 5 | Gradio demo app + final documentation | 🔜 In progress |

All code below has been written and syntax-verified. Reported metrics are
pending a full training run on GPU hardware (see [Reproducing Results](#reproducing-results)) — this README will be updated with real numbers as each stage completes.

---

## Problem Statement

Supervised deep learning is label-hungry. In domains like medical imaging,
every label costs an expert's time, which caps how much labeled data a
team can realistically gather. Meanwhile, *unlabeled* images (raw scans
sitting in a hospital's archive, for instance) are comparatively cheap to
collect.

Semi-supervised learning tries to close that gap: use a small labeled set
to bootstrap a model, then use that model combined with clever
regularization to extract additional training signal from a much larger
pool of unlabeled data.

This project implements and compares four points on that spectrum:

1. **Supervised baseline** — train only on the 15% labeled slice. This is
   the realistic floor for a team that can't afford more annotation.
2. **Pseudo-labeling (self-training)** — use the baseline as a teacher,
   label the unlabeled pool, keep only high-confidence predictions, retrain
   on labeled + pseudo-labeled data.
3. **FixMatch (consistency regularization)** — instead of committing to
   pseudo-labels once, regenerate them every training batch from a weakly
   augmented view, and require the model to reproduce the same prediction
   under a strongly augmented (heavily distorted) view of the same image.
4. **Fully supervised ceiling** — train on 100% of the data with real
   labels, as a reference point for how much of the gap SSL actually closes.

The question this project answers isn't just "does SSL help" — it's
**"which SSL technique holds up better when the labeling budget is small
and the teacher model itself is imperfect, and how close does it get to
what full annotation would have bought us,"** which is the realistic
constraint most teams actually face.

---

## Dataset

**Chest X-Ray Images (Pneumonia)** — [Kaggle, paultimothymooney](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- Binary classification: `NORMAL` vs. `PNEUMONIA`
- ~5,800 images total, natural class imbalance (~73% pneumonia)
- Chosen because it's a realistic, widely-recognized medical imaging
  benchmark, small enough to iterate on quickly, and large enough that a
  15% labeled slice is still a few hundred images — a meaningfully
  data-constrained regime rather than a toy split.

The dataset's default train/val/test folders are merged and re-split by
this project's own pipeline (see below) to control exactly how much data
is labeled vs. unlabeled vs. held out, with stratified sampling to
preserve class balance across all three splits.

---

## Methodology

### 1. Data split
A single stratified split produces:
- **15% labeled** — real labels, used for supervised training
- **70% unlabeled** — labels hidden from the model; a ground-truth copy is
  retained *only* for diagnostics (measuring pseudo-label accuracy), never
  fed into any loss function
- **15% test** — held out untouched until final evaluation, shared across
  all three setups for a fair comparison

### 2. Supervised baseline
ImageNet-pretrained **ResNet-18**, fine-tuned on the 15% labeled slice
only. This is the number every SSL technique needs to beat.

### 3. Pseudo-labeling (self-training)
The baseline model acts as a **teacher**: it scores every unlabeled image,
and predictions with confidence ≥ 0.95 are treated as labels. A fresh
**student** model is then trained on labeled + high-confidence
pseudo-labeled data. A threshold sweep (coverage vs. pseudo-label accuracy)
is used to justify the confidence cutoff rather than picking it arbitrarily.

### 4. FixMatch (consistency regularization)
Addresses pseudo-labeling's main weakness — a weak teacher's *confident
mistakes* get permanently baked into the training set. FixMatch instead:
- Generates a pseudo-label from a **weakly** augmented view (flip + small
  shift), every batch, from the current (constantly improving) model
- Only keeps it if confidence ≥ threshold
- Requires the model to predict that same label from a **strongly**
  augmented view (RandAugment + Cutout-style erasing) of the same image

```
loss = CE(labeled)  +  λ_u · mean( CE(strong_view, pseudo_label) · mask )
```

This couples "is the model confident?" with "is the model *consistent*
under distortion?" — a stricter, self-correcting bar than a static
confidence threshold alone.

### 5. Fully supervised reference ceiling
Trains on the entire pool — the original 15% labeled data plus the 70%
"unlabeled" set, using its hidden ground-truth labels as if they had been
annotated all along. This isn't a technique to deploy (the whole premise
of the project is that you don't have these labels); it's a benchmark that
answers "how much performance is actually on the table," so the SSL
results can be read as *percent of the labeling-cost gap closed*, not just
raw accuracy numbers in isolation.

### 6. Benchmarking & visualization
All four checkpoints are evaluated on the same held-out test set and
compared via:
- A **grouped bar chart** across accuracy/F1/precision/recall
- **Precision-recall curves** overlaid for all four setups (more
  informative than accuracy alone given the dataset's class imbalance)
- **Confusion matrices** side by side, to check which technique reduces
  costlier error types (e.g. pneumonia misclassified as normal)
- **t-SNE/UMAP projections** of the penultimate-layer features, comparing
  the baseline's learned representation against an SSL model's — the
  visual test of whether the unlabeled data actually improved the
  features, not just the final decision boundary


## Reproducing Results

Requires a GPU (Colab's free T4 tier is sufficient). See the per-stage
READMEs for full detail and troubleshooting; the short version:

```bash
pip install -r requirements.txt

# 1. Baseline
python src/prepare_data.py --data_root data/raw --out_dir outputs/manifests \
    --labeled_frac 0.15 --test_frac 0.15 --seed 42
python src/train_baseline.py --manifest_dir outputs/manifests --epochs 15

# 2. Pseudo-labeling
python src/generate_pseudo_labels.py --manifest_dir outputs/manifests \
    --ckpt_path outputs/checkpoints/baseline_best.pt --threshold 0.95
python src/train_pseudolabel.py --manifest_dir outputs/manifests --epochs 15

# 3. FixMatch
python src/train_fixmatch.py --manifest_dir outputs/manifests --epochs 15 \
    --batch_size_labeled 16 --mu 3 --threshold 0.95 --lambda_u 1.0

# 4. Fully supervised ceiling + benchmarking
python src/train_full_supervised.py --manifest_dir outputs/manifests --epochs 15
python src/benchmark.py --outputs_dir outputs
python src/plot_pr_confusion.py --manifest_dir outputs/manifests
python src/plot_embeddings.py --manifest_dir outputs/manifests \
    --model_a_ckpt outputs/checkpoints/baseline_best.pt --model_a_label "Baseline (15%)" \
    --model_b_ckpt outputs/checkpoints/fixmatch_best.pt --model_b_label "FixMatch" \
    --method tsne
```

Each stage writes its own `metrics_*.json` to `outputs/`; `benchmark.py`
aggregates all four into the comparison table below and prints a
ready-to-paste markdown version.

Additional diagnostics tracked per SSL stage:
- **Pseudo-label accuracy** at the chosen confidence threshold (Day 2 &
  Day 3), to quantify how much the confidence filter actually improves
  label quality over the teacher's raw accuracy
- **Coverage vs. accuracy tradeoff** across a threshold sweep (Day 2)
- **Mask rate** (fraction of unlabeled batch used) and pseudo-label
  accuracy trend across training epochs (Day 3)
- **Percent of the labeling-cost gap closed**: `(SSL score − baseline
  score) / (fully-supervised score − baseline score)`, computed once real
  numbers are in — this is the single most recruiter-legible statistic in
  the whole project, since it directly answers "was the SSL technique
  worth it."

---

## Tech Stack

- **Model**: ResNet-18 (ImageNet-pretrained backbone, fine-tuned)
- **Framework**: PyTorch + torchvision
- **Augmentation**: torchvision `RandAugment`, `RandomErasing`
- **Evaluation**: scikit-learn (accuracy, F1, precision, recall, confusion
  matrix, precision-recall curves), matplotlib for benchmarking plots
- **Feature visualization**: scikit-learn t-SNE / UMAP on penultimate-layer embeddings
- **Planned (Day 5)**: Gradio for the interactive demo app

## Limitations

- Results reported here reflect a single train/val/test split and a single
  random seed; a production-grade evaluation would average over multiple
  seeds and report variance.
- The confidence threshold (0.95) and FixMatch's `λ_u`/`μ` hyperparameters
  were chosen based on common defaults from the literature, not an
  exhaustive search. The threshold sweep (Day 2) and benchmarking scripts
  (Day 4) exist to interrogate these choices, but neither has been run
  across multiple hyperparameter settings yet — that would be a natural
  extension beyond the current 5-day scope.
- This is trained on a single public dataset with known class imbalance;
  conclusions about SSL's effectiveness here shouldn't be assumed to
  generalize to other medical imaging tasks without re-validation.
