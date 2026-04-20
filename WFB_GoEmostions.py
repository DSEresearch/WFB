import os
import re
import json
import math
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


DEFAULT_LABEL_MAP = {
    "positive": [
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism",
        "pride", "relief"
    ],
    "negative": [
        "anger", "annoyance", "disappointment", "disapproval", "disgust",
        "embarrassment", "fear", "grief", "nervousness", "remorse",
        "sadness"
    ],
    "ambiguous": ["confusion", "curiosity", "realization", "surprise"],
    "neutral": ["neutral"],
}
COARSE_TO_ID = {"positive": 0, "negative": 1, "ambiguous": 2, "neutral": 3}
ID_TO_COARSE = {v: k for k, v in COARSE_TO_ID.items()}
TOKEN_RE = re.compile(r"[A-Za-z0-9']+|[^\w\s]")

# Category-conditioned nonlinear STA patterns.
# Important: this introduces label information into Delta t and is therefore
# not a fair reproduction of the paper's class-independent STA design.
CATEGORY_PATTERN_CONFIG = {
    "positive": {"beta": 5.0, "tau": 0.32, "lambda1": 0.10, "lambda2": 0.00, "power": 0.90},
    "negative": {"beta": 7.5, "tau": 0.68, "lambda1": 0.12, "lambda2": 0.00, "power": 1.20},
    "ambiguous": {"beta": 8.0, "tau": 0.50, "lambda1": 0.08, "lambda2": 0.10, "power": 1.00},
    "neutral": {"beta": 3.0, "tau": 0.50, "lambda1": 0.03, "lambda2": 0.00, "power": 1.05},
}


def simple_tokenize(text: str):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, vocab_size=20000, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(simple_tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for token, freq in counter.most_common():
        if len(vocab) >= vocab_size:
            break
        if freq >= min_freq:
            vocab[token] = len(vocab)
    return vocab


def normalize_time_curve(dt: torch.Tensor) -> torch.Tensor:
    dt = (dt - dt.min()) / (dt.max() - dt.min() + 1e-8)
    dt[0] = 0.0
    dt[-1] = 1.0
    return dt


def category_conditioned_sta_time(n_tokens: int, label_id: int):
    if n_tokens <= 1:
        return [0.0]
    category = ID_TO_COARSE[int(label_id)]
    cfg = CATEGORY_PATTERN_CONFIG[category]
    p = torch.linspace(0.0, 1.0, steps=n_tokens, dtype=torch.float32)

    base = torch.sigmoid(cfg["beta"] * (p - cfg["tau"]))
    base = torch.pow(base, cfg["power"])

    curve = base + cfg["lambda1"] * torch.sin(math.pi * p)

    if category == "ambiguous":
        # Non-monotone center-emphasis while keeping sentence endpoints anchored.
        curve = curve + cfg["lambda2"] * torch.sin(2.0 * math.pi * p)
    elif category == "neutral":
        # Very mild nonlinearity around a near-linear schedule.
        curve = 0.65 * p + 0.35 * curve

    curve = normalize_time_curve(curve)
    return curve.tolist()


def invert_label_map(label_map):
    out = {}
    for coarse, fine_list in label_map.items():
        for fine in fine_list:
            out[fine] = coarse
    return out


def resolve_goemotions_features(ds):
    return ds.features["labels"].feature.names


def map_example_to_coarse(example_labels, fine_names, fine_to_coarse):
    coarse_hits = []
    for label_id in example_labels:
        fine = fine_names[label_id]
        coarse = fine_to_coarse.get(fine)
        if coarse is not None:
            coarse_hits.append(coarse)
    if not coarse_hits:
        return None
    unique = set(coarse_hits)
    if len(unique) == 1:
        coarse = coarse_hits[0]
    else:
        for coarse in ["negative", "positive", "ambiguous", "neutral"]:
            if coarse in unique:
                break
    return COARSE_TO_ID[coarse]


def prepare_split(ds_split, fine_names, fine_to_coarse):
    rows = []
    for ex in ds_split:
        y = map_example_to_coarse(ex["labels"], fine_names, fine_to_coarse)
        if y is None:
            continue
        text = ex["text"].strip()
        if text:
            rows.append({"text": text, "label": y})
    return pd.DataFrame(rows)


class GoEmotionsCategoryPatternDataset(Dataset):
    def __init__(self, df, vocab, max_len=40):
        self.df = df.reset_index(drop=True)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = simple_tokenize(row["text"])[: self.max_len] or ["<unk>"]
        token_ids = [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]
        dt = category_conditioned_sta_time(len(token_ids), int(row["label"]))
        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "delta_t": torch.tensor(dt, dtype=torch.float32),
            "label": torch.tensor(int(row["label"]), dtype=torch.long),
        }


def collate_batch(batch):
    bs = len(batch)
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = torch.zeros(bs, max_len, dtype=torch.long)
    delta_t = torch.zeros(bs, max_len, dtype=torch.float32)
    mask = torch.zeros(bs, max_len, dtype=torch.float32)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    for i, item in enumerate(batch):
        n = len(item["input_ids"])
        input_ids[i, :n] = item["input_ids"]
        delta_t[i, :n] = item["delta_t"]
        mask[i, :n] = 1.0
    return {"input_ids": input_ids, "delta_t": delta_t, "mask": mask, "labels": labels}


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        with torch.no_grad():
            self.embedding.weight[pad_idx].fill_(0.0)

    def forward(self, input_ids):
        return self.embedding(input_ids)


def masked_mean(x, mask):
    masked = x * mask.unsqueeze(-1)
    denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def time_descriptors(delta_t, mask):
    lengths = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    dt_mean = (delta_t * mask).sum(dim=1, keepdim=True) / lengths
    dt_centered = (delta_t - dt_mean) * mask
    dt_std = torch.sqrt((dt_centered ** 2).sum(dim=1, keepdim=True) / lengths + 1e-8)
    dt_first = delta_t[:, :1]
    dt_last = torch.gather(delta_t, 1, (lengths.long() - 1).clamp_min(0))
    return torch.cat([dt_mean, dt_std, dt_first, dt_last], dim=1)


class ConcatFFNStrict(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=4):
        super().__init__()
        self.encoder = TokenEmbedding(vocab_size, embed_dim)
        self.out = nn.Linear(embed_dim + 4, num_classes)

    def forward(self, input_ids, delta_t, mask):
        token_emb = self.encoder(input_ids)
        x = masked_mean(token_emb, mask)
        t = time_descriptors(delta_t, mask)
        return self.out(torch.cat([x, t], dim=1)), {}


class WFBFFNStrictFixed(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=4):
        super().__init__()
        self.encoder = TokenEmbedding(vocab_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

        self.Ax_raw = nn.Parameter(torch.full((embed_dim,), -0.1))
        self.At_raw = nn.Parameter(torch.full((embed_dim,), -0.1))
        self.k_raw = nn.Parameter(torch.full((embed_dim,), -1.2))
        self.omega_raw = nn.Parameter(torch.full((embed_dim,), -1.2))
        self.theta_x = nn.Parameter(torch.zeros(embed_dim))
        self.theta_t = nn.Parameter(torch.zeros(embed_dim))
        self.out = nn.Linear(embed_dim, num_classes)

    def positive_params(self):
        Ax = F.softplus(self.Ax_raw) + 1e-4
        At = F.softplus(self.At_raw) + 1e-4
        k = F.softplus(self.k_raw) + 1e-4
        omega = F.softplus(self.omega_raw) + 1e-4
        return Ax, At, k, omega

    def wave_state(self, input_ids, delta_t, mask):
        x_raw = self.encoder(input_ids)
        x = torch.tanh(self.norm(x_raw))
        t = delta_t.unsqueeze(-1)
        Ax_vec, At_vec, k_vec, omega_vec = self.positive_params()
        Ax = Ax_vec.view(1, 1, -1)
        At = At_vec.view(1, 1, -1)
        k = k_vec.view(1, 1, -1)
        omega = omega_vec.view(1, 1, -1)
        theta_x = self.theta_x.view(1, 1, -1)
        theta_t = self.theta_t.view(1, 1, -1)
        theta = theta_x + theta_t

        phi = k * x - omega * t - theta
        f = Ax * At * torch.cos(phi)
        sentence_state = masked_mean(f, mask)
        aux = {
            "x": x,
            "t": t,
            "phi": phi,
            "f": f,
            "Ax": Ax,
            "At": At,
            "k": k,
            "omega": omega,
            "theta_x": theta_x,
            "theta_t": theta_t,
        }
        return sentence_state, aux

    def forward(self, input_ids, delta_t, mask):
        sent, aux = self.wave_state(input_ids, delta_t, mask)
        return self.out(sent), aux


def compute_class_weights(labels, num_classes=4):
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def laplacian_penalty_from_aux(aux, mask):
    valid = mask.unsqueeze(-1)
    penalty = 0.5 * (aux["k"] ** 4) * (aux["f"] ** 2)
    penalty = penalty * valid
    denom = valid.sum().clamp_min(1.0)
    return penalty.sum() / denom


def _reduce_grad(grad_tensor, valid):
    grad_tensor = grad_tensor * valid
    denom = valid.sum().clamp_min(1.0)
    return grad_tensor.sum(dim=(0, 1)) / denom


def override_temporal_grads(model, aux, mask, temporal_mode="laplacian", lambda_laplacian=1e-5):
    if aux["f"].grad is None:
        return

    delta = aux["f"].grad.detach()
    t = aux["t"].detach()
    phi = aux["phi"].detach()
    Ax = aux["Ax"].detach()
    At = aux["At"].detach()
    k = aux["k"].detach()
    valid = mask.unsqueeze(-1)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    grad_At_std = delta * Ax * cos_phi
    grad_omega_std = delta * t * Ax * At * sin_phi
    grad_theta_t_std = delta * Ax * At * sin_phi

    grad_At_lap = (k ** 4) * (Ax ** 2) * At * (cos_phi ** 2)
    grad_omega_lap = (k ** 4) * t * (Ax ** 2) * (At ** 2) * cos_phi * sin_phi
    grad_theta_t_lap = (k ** 4) * (Ax ** 2) * (At ** 2) * cos_phi * sin_phi

    if temporal_mode == "laplacian":
        grad_At = lambda_laplacian * grad_At_lap
        grad_omega = lambda_laplacian * grad_omega_lap
        grad_theta_t = lambda_laplacian * grad_theta_t_lap
    elif temporal_mode == "combined":
        grad_At = grad_At_std + lambda_laplacian * grad_At_lap
        grad_omega = grad_omega_std + lambda_laplacian * grad_omega_lap
        grad_theta_t = grad_theta_t_std + lambda_laplacian * grad_theta_t_lap
    else:
        return

    at_grad = _reduce_grad(grad_At, valid)
    omega_grad = _reduce_grad(grad_omega, valid)
    theta_t_grad = _reduce_grad(grad_theta_t, valid)

    model.At_raw.grad = at_grad * torch.sigmoid(model.At_raw.detach())
    model.omega_raw.grad = omega_grad * torch.sigmoid(model.omega_raw.detach())
    model.theta_t.grad = theta_t_grad


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    ys_true, ys_pred = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        delta_t = batch["delta_t"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        logits, _ = model(input_ids, delta_t, mask)
        preds = logits.argmax(dim=1)
        ys_true.extend(labels.cpu().numpy().tolist())
        ys_pred.extend(preds.cpu().numpy().tolist())

    acc = accuracy_score(ys_true, ys_pred)
    mse = mean_squared_error(ys_true, ys_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(ys_true, ys_pred, average="macro", zero_division=0)
    return {"Accuracy": acc, "MSE": mse, "Precision": prec, "Recall": rec, "F1": f1}


def train_one_epoch(model, loader, optimizer, criterion, device, variant="standard", lambda_laplacian=1e-5, grad_clip=1.0, warmup=False):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        delta_t = batch["delta_t"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits, aux = model(input_ids, delta_t, mask)
        ce_loss = criterion(logits, labels)

        if variant == "standard" or not isinstance(model, WFBFFNStrictFixed):
            loss = ce_loss
            loss.backward()
        else:
            aux["f"].retain_grad()
            if warmup:
                loss = ce_loss
                loss.backward()
            elif variant == "laplacian":
                ce_loss.backward(retain_graph=True)
                override_temporal_grads(model, aux, mask, temporal_mode="laplacian", lambda_laplacian=lambda_laplacian)
                loss = ce_loss
            elif variant == "combined":
                penalty = laplacian_penalty_from_aux(aux, mask)
                total = ce_loss + lambda_laplacian * penalty
                total.backward(retain_graph=True)
                override_temporal_grads(model, aux, mask, temporal_mode="combined", lambda_laplacian=lambda_laplacian)
                loss = total
            else:
                raise ValueError(f"Unknown variant: {variant}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.detach().item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_metric_charts(results_df, output_dir):
    metrics = ["Accuracy", "MSE", "Precision", "Recall", "F1"]
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.15
    for i, metric in enumerate(metrics):
        plt.bar(x + (i - 2) * width, results_df[metric].values, width=width, label=metric)
    plt.xticks(x, results_df["Model"].values, rotation=20, ha="right")
    plt.ylabel("Score")
    plt.title("Category-Conditioned STA Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "grouped_metrics.png"), dpi=200)
    plt.close()


def build_optimizer(model, base_lr, wave_lr, weight_decay):
    if not isinstance(model, WFBFFNStrictFixed):
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    wave_param_names = {"Ax_raw", "At_raw", "k_raw", "omega_raw", "theta_x", "theta_t"}
    wave_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in wave_param_names:
            wave_params.append(param)
        else:
            other_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
            {"params": wave_params, "lr": wave_lr, "weight_decay": weight_decay},
        ]
    )


def main():
    parser = argparse.ArgumentParser(description="Strict single-layer GoEmotions evaluation with category-conditioned nonlinear Delta t patterns.")
    parser.add_argument("--output_dir", type=str, default="outputs_category_patterns")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab_size", type=int, default=20000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wave_lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--lambda_laplacian", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = load_dataset("go_emotions")
    fine_names = resolve_goemotions_features(ds["train"])
    fine_to_coarse = invert_label_map(DEFAULT_LABEL_MAP)

    train_df = prepare_split(ds["train"], fine_names, fine_to_coarse)
    valid_df = prepare_split(ds["validation"], fine_names, fine_to_coarse)
    test_df = prepare_split(ds["test"], fine_names, fine_to_coarse)

    vocab = build_vocab(train_df["text"].tolist(), vocab_size=args.vocab_size, min_freq=args.min_freq)

    train_ds = GoEmotionsCategoryPatternDataset(train_df, vocab, max_len=args.max_len)
    valid_ds = GoEmotionsCategoryPatternDataset(valid_df, vocab, max_len=args.max_len)
    test_ds = GoEmotionsCategoryPatternDataset(test_df, vocab, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    class_weights = compute_class_weights(train_df["label"].values)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    model_specs = [
        ("Concat FFN", ConcatFFNStrict(len(vocab), embed_dim=args.embed_dim), "standard"),
        ("STD-WFB-FFN", WFBFFNStrictFixed(len(vocab), embed_dim=args.embed_dim), "standard"),
        ("Laplacian-WFB-FFN", WFBFFNStrictFixed(len(vocab), embed_dim=args.embed_dim), "laplacian"),
        ("STD-Laplacian-WFB-FFN", WFBFFNStrictFixed(len(vocab), embed_dim=args.embed_dim), "combined"),
    ]

    history_rows = []
    results = []

    for model_name, model, variant in model_specs:
        model = model.to(device)
        optimizer = build_optimizer(model, args.lr, args.wave_lr, args.weight_decay)

        best_state = None
        best_valid_f1 = -1.0

        for epoch in range(1, args.epochs + 1):
            warmup = epoch <= args.warmup_epochs
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                variant=variant,
                lambda_laplacian=args.lambda_laplacian,
                grad_clip=args.grad_clip,
                warmup=warmup,
            )
            valid_metrics = evaluate_model(model, valid_loader, device)
            history_rows.append({
                "Model": model_name,
                "Epoch": epoch,
                "TrainLoss": train_loss,
                **{f"Val_{k}": v for k, v in valid_metrics.items()},
            })

            if valid_metrics["F1"] > best_valid_f1:
                best_valid_f1 = valid_metrics["F1"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
            torch.save(best_state, os.path.join(args.output_dir, f"{model_name.replace(' ', '_')}.pt"))

        test_metrics = evaluate_model(model, test_loader, device)
        results.append({"Model": model_name, **test_metrics})

    results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False).reset_index(drop=True)
    history_df = pd.DataFrame(history_rows)

    results_csv = os.path.join(args.output_dir, "category_conditioned_results.csv")
    results_xlsx = os.path.join(args.output_dir, "category_conditioned_results.xlsx")
    history_csv = os.path.join(args.output_dir, "category_conditioned_history.csv")

    results_df.to_csv(results_csv, index=False)
    history_df.to_csv(history_csv, index=False)
    with pd.ExcelWriter(results_xlsx, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="Results", index=False)
        history_df.to_excel(writer, sheet_name="History", index=False)

    save_metric_charts(results_df, args.output_dir)

    with open(os.path.join(args.output_dir, "category_pattern_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "warning": "Category-conditioned Delta t introduces label leakage and is not a fair reproduction of class-independent STA.",
            "patterns": CATEGORY_PATTERN_CONFIG,
            "label_map": DEFAULT_LABEL_MAP,
            "args": vars(args),
        }, f, indent=2)

    print("\nResults")
    print(results_df.to_string(index=False))
    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
