# Day 1 — Setup & Data Preparation

**Goal today:** get a clean, reproducible 15/70/15 split and a supervised
baseline number you'll spend the rest of the week trying to beat.

## 0. Where to run this

You need a GPU for reasonable training speed. Easiest options:
- **Google Colab** (free T4 GPU) — upload this `ssl_project/` folder, or `git clone` it if you push to GitHub first.
- **Kaggle Notebooks** — nice because the dataset is already hosted there, no download needed.

## 1. Environment

```bash
pip install -r requirements.txt
```

## 2. Get the dataset

Chest X-Ray Pneumonia dataset (Kaggle: `paultimothymooney/chest-xray-pneumonia`).

```bash
# via Kaggle CLI (needs ~/.kaggle/kaggle.json API token)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/kaggle_raw
```

The Kaggle download comes pre-split into `train/`, `val/`, `test/` folders,
each containing `NORMAL/` and `PNEUMONIA/` subfolders. **Merge them into one
pool** — we want to control the split ourselves rather than inherit theirs
(their `val/` set is tiny, only 16 images):

```bash
mkdir -p data/raw/NORMAL data/raw/PNEUMONIA
find data/kaggle_raw -path "*/NORMAL/*" -name "*.jpeg" -exec cp {} data/raw/NORMAL/ \;
find data/kaggle_raw -path "*/PNEUMONIA/*" -name "*.jpeg" -exec cp {} data/raw/PNEUMONIA/ \;
```

You should end up with:
```
data/raw/
    NORMAL/       (~1,575 images)
    PNEUMONIA/    (~4,265 images)
```

Note the class imbalance (~73% PNEUMONIA) — mention this in your README later;
it's why we use F1, not just accuracy, and why stratified splitting matters.

## 3. Build the 15/70/15 split

```bash
python src/prepare_data.py \
    --data_root data/raw \
    --out_dir outputs/manifests \
    --labeled_frac 0.15 \
    --test_frac 0.15 \
    --seed 42
```

This writes:
- `outputs/manifests/labeled.csv` — 15% of data, with real labels
- `outputs/manifests/unlabeled.csv` — 70%, `label=-1` (true label kept as
  `true_label` for later diagnostics only — never train on it)
- `outputs/manifests/test.csv` — 15% held out, untouched until final eval
- `outputs/manifests/class_map.csv` — class name ↔ integer label mapping

Sanity-check the class balance held up across splits:
```bash
python -c "
import pandas as pd
for f in ['labeled','unlabeled','test']:
    df = pd.read_csv(f'outputs/manifests/{f}.csv')
    col = 'true_label' if f=='unlabeled' else 'label'
    print(f, df['class_name'].value_counts(normalize=True).round(3).to_dict())
"
```

## 4. Train the supervised baseline (labeled subset only)

```bash
python src/train_baseline.py \
    --manifest_dir outputs/manifests \
    --epochs 15 \
    --batch_size 32 \
    --lr 1e-4
```

This fine-tunes an ImageNet-pretrained ResNet-18 on just your 15% labeled
slice, early-stops on validation F1, and reports final test accuracy/F1/
precision/recall — saved to `outputs/metrics_baseline.json`.

## Day 1 checklist

- [ ] Dataset downloaded and merged into `data/raw/<class>/`
- [ ] `prepare_data.py` run, class balance verified across all 3 splits
- [ ] Baseline model trained, `outputs/metrics_baseline.json` populated
- [ ] Record these baseline numbers — you'll need them for Day 4's comparison table:

| Setup                          | Accuracy | F1 | Precision | Recall |
|---------------------------------|----------|----|-----------|--------|
| Supervised (15% labeled only)   |          |    |           |        |

## What's next (Day 2 preview)

Day 2 reuses `outputs/manifests/unlabeled.csv` and the baseline checkpoint
at `outputs/checkpoints/baseline_best.pt` as the "teacher" model — no need
to touch Day 1's code again, just build on top of it.

# Day 2 — Pseudo-Labeling Pipeline (Self-Training)

**Goal today:** use the Day 1 teacher model to label the unlabeled pool,
keep only confident predictions, retrain, and beat the baseline.

Run these on the same machine/notebook where you did Day 1 — they expect
`outputs/manifests/` and `outputs/checkpoints/baseline_best.pt` to already exist.

## 1. Score the unlabeled pool with the teacher model

```bash
python src/generate_pseudo_labels.py \
    --manifest_dir outputs/manifests \
    --ckpt_path outputs/checkpoints/baseline_best.pt \
    --threshold 0.95
```

This prints diagnostics like:
```
Teacher accuracy on ALL unlabeled images:  91.2%
Kept (>= threshold):                        612 / 980 (62.4%)
Teacher accuracy on KEPT (pseudo-label) set: 98.1%
```
That gap (91.2% → 98.1%) is the entire point of thresholding: you trade
coverage for cleaner labels. Note both numbers — they go straight into your
README's methodology section.

It writes:
- `outputs/manifests/pseudo_labels_all.csv` — every unlabeled image scored
- `outputs/manifests/pseudo_labels_kept.csv` — only the ones that passed threshold, ready to train on

## 2. (Optional but recommended) Sweep the threshold

```bash
python src/analyze_pseudo_labels.py --manifest_dir outputs/manifests
```

Produces `outputs/figures/threshold_sweep.png` (coverage vs. pseudo-label
accuracy at several thresholds) and `confidence_histogram.png` (correct vs.
incorrect predictions by confidence). These two plots are the kind of thing
that makes a portfolio project look rigorous instead of arbitrary — use them
to justify your choice of 0.95 in the README rather than just asserting it.

## 3. Retrain on labeled + pseudo-labeled data

```bash
python src/train_pseudolabel.py \
    --manifest_dir outputs/manifests \
    --epochs 15 --batch_size 32 --lr 1e-4
```

Trains a fresh ResNet-18 on (original labeled minus val split) + (kept
pseudo-labels), validates on real labels only, evaluates on the same
`test.csv` as Day 1. Saves `outputs/metrics_pseudolabel.json`.

## Update your comparison table

| Setup                                  | Accuracy | F1 | Precision | Recall |
|------------------------------------------|----------|----|-----------|--------|
| Supervised (15% labeled only) — Day 1     |          |    |           |        |
| Pseudo-labeling (15% + high-conf pseudo) — Day 2 |    |    |           |        |

If pseudo-labeling *doesn't* beat the baseline, that's not a failure —
it's a finding. Common causes worth investigating and writing up:
- **Confirmation bias**: the teacher is weak (only 15% data), so its
  confident mistakes get reinforced. Check the "pseudo-label accuracy on
  kept" number from step 1 — if it's not meaningfully higher than overall
  teacher accuracy, thresholding isn't filtering out much.
- **Class imbalance compounding**: if the teacher is biased toward
  PNEUMONIA (the majority class), pseudo-labels will skew the same way,
  making the student more imbalanced, not less. Check the
  `pred_class_name` value counts printed in step 1.
- **Threshold too low/high**: use the sweep plot to see if a different
  threshold changes the outcome.

This diagnosis is exactly what Day 3 (FixMatch) is designed to fix — weak/
strong augmentation consistency is more robust to a weak teacher than raw
confidence thresholding.
