#!/usr/bin/env python3
"""
Thermal facial emotion classifier for the Mendeley "Comprehensive Facial Thermal Dataset"
(Abuhussein, Darwish & Hassanien, 2024 - DOI 10.17632/8885sc9p4z.1).

What this fixes compared with a typical first attempt:
  1. Pretrained CNN (ResNet18 / EfficientNet-B0 / ConvNeXt-Tiny) fine-tuned, instead of a CNN from scratch.
  2. Colour-palette control: the dataset mixes RAINBOW / IRONBOW / White Hot / Red Hot / ICEBLUE, so the same
     temperature has a different colour in each palette. You can train on all, on one, or per palette.
  3. Leak-free evaluation: near-duplicate frames (burst shots) are grouped so they never sit in both
     train and test. If you know which person is in each image, pass --subjects to get a person-independent split.
  4. Augmentation, label smoothing, cosine LR, best-epoch selection on an inner validation split
     (never on the test fold), flip test-time augmentation.
  5. Honest reporting: 5-fold mean +/- std accuracy and macro-F1, confusion matrix, per-palette accuracy.

Google Colab (Runtime -> Change runtime type -> T4 GPU):
  !pip install -q torch torchvision scikit-learn pandas matplotlib pillow
  !python thermal_emotion_train.py --data "/content/drive/MyDrive/Facial Emotion" --scan-only
  !python thermal_emotion_train.py --data "/content/drive/MyDrive/Facial Emotion" --palette-mode all
  !python thermal_emotion_train.py --data "/content/drive/MyDrive/Facial Emotion" --palette-mode per-palette

--data must point at the folder that contains the emotion folders (happy / sad / angry / normal / surprised).
Labels, palette, angle and zoom are read from the folder names, whatever their exact spelling.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------------------------
# 1. Reading the dataset folder tree
# ----------------------------------------------------------------------------------------------

IMG_EXT = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

EMOTIONS = {  # canonical name -> substrings that identify it in a folder name
    "happy": ["happ"],
    "sad": ["sad"],
    "angry": ["angr", "anger"],
    "neutral": ["neutral", "normal", "natural"],
    "surprise": ["surpris", "surpise", "suprise"],
}
PALETTES = {  # order matters: checked top to bottom
    "ironbow": ["ironbow", "iron", "irnbow"],
    "rainbow": ["rainbow"],
    "whitehot": ["whitehot"],
    "redhot": ["redhot"],
    "iceblue": ["iceblue", "ice"],
}
ANGLES = {"front": ["front"], "left": ["left"], "right": ["right"]}


def _norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _match(part, table):
    p = _norm(part)
    for name, keys in table.items():
        if any(k in p for k in keys):
            return name
    return None


def scan_dataset(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in IMG_EXT or not path.is_file():
            continue
        parts = path.relative_to(root).parts[:-1]
        emotion = palette = angle = None
        for part in parts:
            emotion = emotion or _match(part, EMOTIONS)
            palette = palette or _match(part, PALETTES)
            angle = angle or _match(part, ANGLES)
        if emotion is None:  # fall back to the file name
            emotion = _match(path.stem, EMOTIONS)
        if emotion is None:
            continue
        rows.append({
            "path": str(path),
            "rel": str(path.relative_to(root)),
            "emotion": emotion,
            "palette": palette or "unknown",
            "angle": angle or "unknown",
            "folder": str(Path(*parts)) if parts else ".",
        })
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit(f"No labelled images found under {root}. Point --data at the folder holding the emotion folders.")
    return df


# ----------------------------------------------------------------------------------------------
# 2. Image loading / palette handling
# ----------------------------------------------------------------------------------------------

def load_image(path, crop, palette_mode, size=256):
    img = Image.open(path).convert("RGB")
    if any(crop):
        w, h = img.size
        l, t, r, b = crop
        img = img.crop((int(l * w), int(t * h), int(w - r * w), int(h - b * h)))
    if palette_mode == "gray":
        img = img.convert("L").convert("RGB")
    return img.resize((size, size), Image.BILINEAR)


def save_sample_grid(df, out, crop, n_per=4):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pals = sorted(df.palette.unique())
    emos = sorted(df.emotion.unique())
    fig, axes = plt.subplots(len(pals), len(emos), figsize=(2.6 * len(emos), 2.2 * len(pals)), squeeze=False)
    for i, p in enumerate(pals):
        for j, e in enumerate(emos):
            ax = axes[i][j]
            ax.axis("off")
            sub = df[(df.palette == p) & (df.emotion == e)]
            if len(sub):
                ax.imshow(load_image(sub.iloc[len(sub) // 2].path, crop, "all"))
            if i == 0:
                ax.set_title(e, fontsize=10)
            if j == 0:
                ax.text(-0.05, 0.5, p, transform=ax.transAxes, ha="right", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close()


# ----------------------------------------------------------------------------------------------
# 3. Grouping so near-duplicate frames never leak between train and test
# ----------------------------------------------------------------------------------------------

def dhash_bits(path, crop, hash_size=16):
    img = load_image(path, crop, "gray", size=128).convert("L").resize((hash_size + 1, hash_size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()  # 256 bits


def near_duplicate_groups(df, crop, max_dist):
    """Burst frames of the same shot sit in the same leaf folder, so frames are only merged within a folder."""
    groups = np.empty(len(df), dtype=object)
    for folder, idx in df.groupby("folder").indices.items():
        bits = np.stack([dhash_bits(df.path.iat[i], crop) for i in idx]).astype(np.int32)
        dist = bits @ (1 - bits).T + (1 - bits) @ bits.T  # Hamming distance matrix
        parent = list(range(len(idx)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in zip(*np.where(np.triu(dist <= max_dist, k=1))):
            parent[find(i)] = find(j)
        for local, i in enumerate(idx):
            groups[i] = f"{folder}#{find(local)}"
    return groups


def make_groups(df, args):
    if args.subjects:
        m = pd.read_csv(args.subjects)  # columns: rel (path relative to --data) or filename, subject
        key = "rel" if "rel" in m.columns else "filename"
        if key == "filename":
            df["filename"] = df.rel.map(lambda r: Path(r).name)
        df = df.merge(m[[key, "subject"]], on=key, how="left")
        missing = df.subject.isna().sum()
        if missing:
            print(f"WARNING: {missing} images have no subject in {args.subjects}; they are dropped.")
            df = df.dropna(subset=["subject"]).reset_index(drop=True)
        df["group"] = df.subject.astype(str)
        print(f"Grouping by subject: {df.group.nunique()} people")
        return df
    if args.group_by == "time":
        # files are named imgYYYYMMDD_HHMMSS: frames of one emotion shot within --time-gap seconds of each other
        # (any palette) are one recording burst and must stay on the same side of the split
        ts = pd.to_datetime(df.rel.str.extract(r"(\d{8}_\d{6})")[0], format="%Y%m%d_%H%M%S", errors="coerce")
        if ts.isna().any():
            sys.exit(f"{ts.isna().sum()} file names have no imgYYYYMMDD_HHMMSS timestamp; use --group-by dup")
        df = df.assign(_ts=ts).sort_values(["emotion", "_ts"])
        new_burst = (df.emotion != df.emotion.shift()) | (df._ts.diff().dt.total_seconds() > args.time_gap)
        df["group"] = (df.emotion + "#" + new_burst.cumsum().astype(str))
        df = df.drop(columns="_ts").sort_index()
        sizes = Counter(df.group)
        print(f"  {len(df)} images -> {len(sizes)} recording bursts (largest {max(sizes.values())}, "
              f"median {int(np.median(list(sizes.values())))})")
    elif args.group_by == "folder":
        df["group"] = df.folder
    elif args.group_by == "none":
        df["group"] = np.arange(len(df)).astype(str)
    else:
        print("Hashing images to find near-duplicate frames ...")
        df["group"] = near_duplicate_groups(df, args.crop, args.dup_dist).astype(str)
        sizes = Counter(df.group)
        print(f"  {len(df)} images -> {len(sizes)} groups of near-identical frames "
              f"(largest {max(sizes.values())}, median {int(np.median(list(sizes.values())))})")
        if max(sizes.values()) > 0.05 * len(df):
            print("  WARNING: some groups are very large - lower --dup-dist if distinct shots are being merged.")
    return df


# ----------------------------------------------------------------------------------------------
# 4. Model + training
# ----------------------------------------------------------------------------------------------

ARCHS = ["efficientnet_b0", "efficientnet_b2", "resnet18", "resnet50", "convnext_tiny", "swin_t"]


def build_model(arch, n_classes, pretrained):
    """Returns (model, head_parameter_names)."""
    import torch.nn as nn
    from torchvision import models
    w = "DEFAULT" if pretrained else None
    try:
        if arch.startswith("resnet"):
            m = getattr(models, arch)(weights=w)
            m.fc = nn.Linear(m.fc.in_features, n_classes)
            head = "fc."
        elif arch.startswith("efficientnet"):
            m = getattr(models, arch)(weights=w)
            m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
            head = "classifier."
        elif arch.startswith("convnext"):
            m = getattr(models, arch)(weights=w)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_classes)
            head = "classifier."
        elif arch.startswith("swin"):
            m = getattr(models, arch)(weights=w)
            m.head = nn.Linear(m.head.in_features, n_classes)
            head = "head."
        else:
            raise ValueError(arch)
    except Exception as e:  # usually no internet for the weights download
        sys.exit(f"Could not build {arch} with pretrained weights ({e}). Check internet, or use --no-pretrained.")
    return m, head


class ImgDS:
    def __init__(self, images, labels, tf):
        self.images, self.labels, self.tf = images, labels, tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]


def transforms(img_size):
    from torchvision import transforms as T
    norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.RandomRotation(10),
        # brightness/contrast only: hue shifts would change what temperature a colour means
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(), norm,
        T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
    ])
    test = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    return train, test


def predict(model, loader, device, tta=True):
    import torch
    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            p = model(x).softmax(1)
            if tta:
                p = (p + model(torch.flip(x, dims=[3])).softmax(1)) / 2
            outs.append(p.cpu())
    return torch.cat(outs).numpy()


def train_one(train_imgs, train_y, val_imgs, val_y, n_classes, args, device, arch):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    tf_train, tf_test = transforms(args.img_size)
    dl_tr = DataLoader(ImgDS(train_imgs, train_y, tf_train), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, drop_last=len(train_imgs) > args.batch)
    dl_va = DataLoader(ImgDS(val_imgs, val_y, tf_test), batch_size=args.batch * 2, num_workers=args.workers)

    model, head = build_model(arch, n_classes, not args.no_pretrained)
    model = model.to(device)
    # pretrained backbone learns slowly, the new classifier head learns fast
    head_p = [p for n, p in model.named_parameters() if n.startswith(head)]
    body_p = [p for n, p in model.named_parameters() if not n.startswith(head)]
    opt = torch.optim.AdamW([{"params": body_p, "lr": args.lr},
                             {"params": head_p, "lr": args.lr * args.head_lr_mult}], weight_decay=args.wd)
    steps = args.epochs * max(1, len(dl_tr))
    warm = max(1, len(dl_tr))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def mixed_loss(x, y):
        if args.mixup <= 0 or random.random() > 0.5:
            return loss_fn(model(x), y)
        lam = float(np.random.beta(args.mixup, args.mixup))
        perm = torch.randperm(x.size(0), device=x.device)
        if random.random() < 0.5:  # mixup
            x = lam * x + (1 - lam) * x[perm]
        else:  # cutmix
            H, W = x.shape[2:]
            rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
            cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
            y0, y1, x0, x1 = max(cy - rh // 2, 0), min(cy + rh // 2, H), max(cx - rw // 2, 0), min(cx + rw // 2, W)
            x = x.clone()
            x[:, :, y0:y1, x0:x1] = x[perm][:, :, y0:y1, x0:x1]
            lam = 1 - (y1 - y0) * (x1 - x0) / (H * W)
        out = model(x)
        return lam * loss_fn(out, y) + (1 - lam) * loss_fn(out, y[perm])
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, best_state, patience = -1, None, 0
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), torch.as_tensor(y).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = mixed_loss(x, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
        pv = predict(model, dl_va, device, tta=False).argmax(1)
        f1 = f1_score(val_y, pv, average="macro")
        print(f"    epoch {ep + 1:2d}/{args.epochs}  train loss {tot / len(train_imgs):.3f}  "
              f"val acc {np.mean(pv == np.array(val_y)):.3f}  val macro-F1 {f1:.3f}")
        if f1 > best_f1:
            best_f1, patience = f1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                print("    early stop")
                break
    model.load_state_dict(best_state)
    return model


def run_cv(df, args, device, tag, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score

    classes = sorted(df.emotion.unique())
    y = df.emotion.map({c: i for i, c in enumerate(classes)}).values
    groups = df.group.values
    print(f"\n=== {tag}: {len(df)} images, classes {classes}, {len(set(groups))} groups ===")
    print("Loading images into memory ...")
    imgs = [load_image(p, args.crop, args.palette_mode) for p in df.path]

    if args.holdout_regex:
        te_mask = df.rel.str.contains(args.holdout_regex, regex=True).values
        folds = [(np.where(~te_mask)[0], np.where(te_mask)[0])]
        print(f"Hold-out test: {te_mask.sum()} images matching {args.holdout_regex!r}, training on {(~te_mask).sum()}")
    elif args.split != "random" and len(set(groups)) < args.folds:
        sys.exit(f"Only {len(set(groups))} groups - need at least {args.folds}. Lower --folds or --dup-dist.")
    if args.holdout_regex:
        pass
    elif args.split == "random":
        folds = list(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed).split(imgs, y))
    else:
        folds = list(StratifiedGroupKFold(args.folds, shuffle=True, random_state=args.seed).split(imgs, y, groups))
    if args.max_folds:
        folds = folds[:args.max_folds]

    _, tf_test = transforms(args.img_size)
    preds = np.zeros((len(df), len(classes)))
    arch_preds = {a: np.zeros((len(df), len(classes))) for a in args.archs}
    fold_id = np.full(len(df), -1)
    rows = []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        # inner validation split (by group) for picking the best epoch - the test fold is never looked at
        n_inner = max(2, round(1 / args.val_frac))
        if len(set(groups[tr])) >= n_inner:
            inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=args.seed + k)
            itr, iva = next(inner.split(tr, y[tr], groups[tr]))
        else:
            inner = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=args.seed + k)
            itr, iva = next(inner.split(tr, y[tr]))
        tr_i, va_i = tr[itr], tr[iva]
        print(f"\n  Fold {k + 1}/{len(folds)}: train {len(tr_i)}  val {len(va_i)}  test {len(te)}")
        dl_te = DataLoader(ImgDS([imgs[i] for i in te], y[te].tolist(), tf_test), batch_size=args.batch * 2,
                           num_workers=args.workers)
        fold_probs = []
        for arch in args.archs:
            print(f"   [{arch}]")
            model = train_one([imgs[i] for i in tr_i], y[tr_i].tolist(), [imgs[i] for i in va_i],
                              y[va_i].tolist(), len(classes), args, device, arch)
            p = predict(model, dl_te, device, tta=True)
            fold_probs.append(p)
            arch_preds[arch][te] = p
            a = accuracy_score(y[te], p.argmax(1))
            rows.append({"fold": k + 1, "model": arch, "accuracy": a,
                         "macro_f1": f1_score(y[te], p.argmax(1), average="macro"), "n_test": len(te)})
            print(f"   [{arch}] fold {k + 1} TEST accuracy {a:.3f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        p = np.mean(fold_probs, axis=0)
        preds[te], fold_id[te] = p, k
        acc = accuracy_score(y[te], p.argmax(1))
        f1 = f1_score(y[te], p.argmax(1), average="macro")
        rows.append({"fold": k + 1, "model": "ENSEMBLE" if len(args.archs) > 1 else args.archs[0],
                     "accuracy": acc, "macro_f1": f1, "n_test": len(te), "minutes": (time.time() - t0) / 60})
        print(f"  Fold {k + 1} TEST  accuracy {acc:.3f}  macro-F1 {f1:.3f}  ({(time.time() - t0) / 60:.1f} min)")

    done = fold_id >= 0
    all_rows = pd.DataFrame(rows)
    final_name = "ENSEMBLE" if len(args.archs) > 1 else args.archs[0]
    res = all_rows[all_rows.model == final_name].drop_duplicates("fold", keep="last")
    if len(args.archs) > 1:
        print("\n  Per-model accuracy (mean over folds):")
        print(all_rows.groupby("model").accuracy.agg(["mean", "std"]).round(3).to_string())
    pred_df = df[done].copy()
    pred_df["true"] = [classes[i] for i in y[done]]
    pred_df["pred"] = [classes[i] for i in preds[done].argmax(1)]
    pred_df["fold"] = fold_id[done] + 1
    for i, c in enumerate(classes):
        pred_df[f"p_{c}"] = preds[done][:, i]
    pred_df.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    all_rows.to_csv(out_dir / f"{tag}_folds.csv", index=False)
    save_confusion(pred_df, classes, out_dir / f"{tag}_confusion.png", tag)

    per_pal = pred_df.assign(ok=pred_df.true == pred_df.pred).groupby("palette").ok.agg(["mean", "count"])
    per_pal.columns = ["accuracy", "n"]
    summary = {
        "tag": tag, "images": int(done.sum()), "folds": len(rows), "split": args.split,
        "grouping": "subject" if args.subjects else args.group_by, "models": "+".join(args.archs),
        "palette_mode": args.palette_mode,
        "accuracy_mean": float(res.accuracy.mean()), "accuracy_std": float(res.accuracy.std(ddof=0)),
        "macro_f1_mean": float(res.macro_f1.mean()), "macro_f1_std": float(res.macro_f1.std(ddof=0)),
        "chance_level": 1 / len(classes),
    }
    print(f"\n  {tag}: accuracy {summary['accuracy_mean']:.3f} +/- {summary['accuracy_std']:.3f}   "
          f"macro-F1 {summary['macro_f1_mean']:.3f} +/- {summary['macro_f1_std']:.3f}   (chance {1 / len(classes):.2f})")
    print("  Accuracy by palette:\n" + per_pal.round(3).to_string())
    return summary, per_pal


def train_final(df, args, device, out_dir):
    """Train on ALL images (small group-held-out val for epoch picking) and save the deployable model."""
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    classes = sorted(df.emotion.unique())
    y = df.emotion.map({c: i for i, c in enumerate(classes)}).values
    imgs = [load_image(p, args.crop, args.palette_mode) for p in df.path]
    itr, iva = next(StratifiedGroupKFold(n_splits=max(2, round(1 / args.val_frac)), shuffle=True,
                                         random_state=args.seed).split(imgs, y, df.group.values))
    states = {}
    for arch in args.archs:
        print(f"\n=== Final model [{arch}] on {len(itr)} train / {len(iva)} val images ===")
        m = train_one([imgs[i] for i in itr], y[itr].tolist(), [imgs[i] for i in iva], y[iva].tolist(),
                      len(classes), args, device, arch)
        states[arch] = {k: v.cpu() for k, v in m.state_dict().items()}
    path = out_dir / "final_model.pt"
    torch.save({"classes": classes, "archs": args.archs, "states": states, "img_size": args.img_size,
                "crop": args.crop, "palette_mode": args.palette_mode}, path)
    print(f"Saved {path}  (load with load_final_model / predict_image in this script)")


def load_final_model(path, device="cpu"):
    import torch
    ck = torch.load(path, map_location=device)
    models_ = []
    for arch in ck["archs"]:
        m, _ = build_model(arch, len(ck["classes"]), pretrained=False)
        m.load_state_dict(ck["states"][arch])
        models_.append(m.to(device).eval())
    return ck, models_


def predict_image(path_or_ck, image_path, device="cpu"):
    """Example: ck, ms = load_final_model('results/final_model.pt'); predict_image((ck, ms), 'face.bmp')"""
    import torch
    ck, ms = load_final_model(path_or_ck, device) if isinstance(path_or_ck, (str, Path)) else path_or_ck
    _, tf = transforms(ck["img_size"])
    x = tf(load_image(image_path, ck["crop"], ck["palette_mode"])).unsqueeze(0).to(device)
    with torch.no_grad():
        p = sum((m(x).softmax(1) + m(torch.flip(x, dims=[3])).softmax(1)) / 2 for m in ms) / len(ms)
    p = p[0].cpu().numpy()
    return dict(sorted(((c, round(float(v), 3)) for c, v in zip(ck["classes"], p)), key=lambda t: -t[1]))


def save_confusion(pred_df, classes, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(pred_df.true, pred_df.pred, labels=classes, normalize="true")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ----------------------------------------------------------------------------------------------
# 5. Main
# ----------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder that contains the emotion folders")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scan-only", action="store_true", help="just count images and save a sample grid")
    ap.add_argument("--palette-mode", default="all",
                    help="all | gray | per-palette | one palette name (whitehot, ironbow, rainbow, redhot, iceblue)")
    ap.add_argument("--crop", type=float, nargs=4, default=[0, 0, 0, 0], metavar=("L", "T", "R", "B"),
                    help="fraction to cut from each side, e.g. 0 0 0.15 0 removes a scale bar on the right")
    ap.add_argument("--archs", nargs="+", default=["efficientnet_b0"], choices=ARCHS,
                    help="one model, or several to train and average (ensemble)")
    ap.add_argument("--head-lr-mult", type=float, default=10, help="classifier head lr = lr x this")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup/cutmix alpha, e.g. 0.4 (0 = off)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--split", default="grouped", choices=["grouped", "random"],
                    help="random = old-style split, only for showing how much leakage inflates accuracy")
    ap.add_argument("--group-by", default="dup", choices=["dup", "time", "folder", "none"],
                    help="dup = near-duplicate frames kept together (default); folder = whole leaf folders held out")
    ap.add_argument("--subjects", help="CSV with columns rel (or filename) and subject -> person-independent split")
    ap.add_argument("--time-gap", type=float, default=60, help="seconds between frames that still count as one burst")
    ap.add_argument("--include-regex", help="keep only images whose relative path matches this regex")
    ap.add_argument("--holdout-regex", help="instead of k-fold CV: test on images matching this regex, train on the rest")
    ap.add_argument("--dup-dist", type=int, default=6, help="max Hamming distance (of 64) to call two frames duplicates")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=0, help="run only the first N folds (quick test)")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone learning rate")
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only N random images (quick test)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save-final", action="store_true", help="after CV, train on all data and save final_model.pt")
    ap.add_argument("--skip-cv", action="store_true", help="only train and save the final model")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = scan_dataset(args.data)
    print(f"Found {len(df)} labelled images\n")
    print(pd.crosstab(df.emotion, df.palette, margins=True).to_string(), "\n")
    print(pd.crosstab(df.emotion, df.angle).to_string(), "\n")
    save_sample_grid(df, out_dir / "samples_per_palette.png", args.crop)
    print(f"Saved {out_dir / 'samples_per_palette.png'} - open it and check for on-screen overlays to --crop.")
    if args.scan_only:
        return

    import torch
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + ("" if device.type == "cuda" else "  (no GPU - this will be slow; use a Colab T4)"))

    if args.include_regex:
        df = df[df.rel.str.contains(args.include_regex, regex=True)].reset_index(drop=True)
        print(f"--include-regex kept {len(df)} images")
    if args.limit:
        df = df.sample(min(args.limit, len(df)), random_state=args.seed).reset_index(drop=True)
    df = make_groups(df.reset_index(drop=True), args)

    mode = args.palette_mode.lower()
    summaries, palette_tables = [], []
    if mode == "per-palette":
        args.palette_mode = "all"
        for pal in sorted(df.palette.unique()):
            sub = df[df.palette == pal].reset_index(drop=True)
            if sub.emotion.nunique() < 2 or len(sub) < 50:
                continue
            s, _ = run_cv(sub, args, device, f"palette_{pal}", out_dir)
            summaries.append(s)
    elif mode in PALETTES or mode == "unknown":
        sub = df[df.palette == mode].reset_index(drop=True)
        if sub.empty:
            sys.exit(f"No images with palette '{mode}'. Palettes found: {sorted(df.palette.unique())}")
        args.palette_mode = "all"
        s, _ = run_cv(sub, args, device, f"palette_{mode}", out_dir)
        summaries.append(s)
    elif mode in ("all", "gray"):
        args.palette_mode = mode
        if not args.skip_cv:
            s, t = run_cv(df, args, device, f"{mode}_palettes", out_dir)
            summaries.append(s)
            t.to_csv(out_dir / f"{mode}_palettes_by_palette.csv")
        if args.save_final or args.skip_cv:
            train_final(df, args, device, out_dir)
    else:
        sys.exit(f"Unknown --palette-mode {mode}")

    if not summaries:
        return
    pd.DataFrame(summaries).to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.json", "w") as f:
        json.dump({"args": vars(args), "results": summaries}, f, indent=2, default=str)
    print(f"\nAll results saved in {out_dir.resolve()}")
    print(pd.DataFrame(summaries)[["tag", "images", "accuracy_mean", "accuracy_std", "macro_f1_mean", "chance_level"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
