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
