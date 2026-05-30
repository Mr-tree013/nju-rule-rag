"""
Fine-tune BGE-Reranker-v2-m3 on NJU-domain relevance pairs.

Builds training data from eval gold-source annotations, fine-tunes the
cross-encoder for 2-3 epochs, evaluates, and saves the model.

Usage:
    python scripts/finetune_reranker.py           # train + eval
    python scripts/finetune_reranker.py --eval-only  # eval only
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "eval" / "reranker_train.jsonl"
TEST_FILE = ROOT / "data" / "eval" / "reranker_test.jsonl"
MODEL_NAME = "BAAI/bge-reranker-v2-m3"
OUTPUT_DIR = ROOT / "data" / "models" / "bge-reranker-nju"

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1


def load_pairs(path: Path) -> list[InputExample]:
    examples = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            examples.append(InputExample(
                texts=[d["query"], d["content"]],
                label=float(d["label"]),
            ))
    return examples


def evaluate(model: CrossEncoder, examples: list[InputExample]) -> dict:
    """Compute accuracy, precision, recall on test set."""
    correct = 0
    tp = fp = fn = tn = 0
    for ex in examples:
        score = model.predict([ex.texts])[0]
        pred = 1 if score > 0 else 0
        true = int(ex.label)
        if pred == true:
            correct += 1
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
        else:
            tn += 1

    n = len(examples)
    acc = correct / n if n else 0
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    # Also check score distribution
    scores = [model.predict([ex.texts])[0] for ex in examples[:30]]
    pos_scores = [model.predict([ex.texts])[0] for ex in examples if ex.label == 1][:20]
    neg_scores = [model.predict([ex.texts])[0] for ex in examples if ex.label == 0][:20]

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "n": n,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "pos_score_mean": sum(pos_scores) / len(pos_scores) if pos_scores else 0,
        "neg_score_mean": sum(neg_scores) / len(neg_scores) if neg_scores else 0,
        "score_separation": (
            (sum(pos_scores) / len(pos_scores) - sum(neg_scores) / len(neg_scores))
            if pos_scores and neg_scores else 0
        ),
    }


def main():
    # Force unbuffered output for WSL2
    sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
    eval_only = "--eval-only" in sys.argv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)
    print(f"Train file: {TRAIN_FILE} ({TRAIN_FILE.stat().st_size if TRAIN_FILE.exists() else 0} bytes)", flush=True)
    print(f"Test file:  {TEST_FILE} ({TEST_FILE.stat().st_size if TEST_FILE.exists() else 0} bytes)", flush=True)

    if not TRAIN_FILE.exists():
        print("ERROR: Training data not found. Run build step first.", file=sys.stderr)
        return 1

    if eval_only:
        print("\n=== Evaluation only ===")
        model = CrossEncoder(str(OUTPUT_DIR), device=device)
        test_examples = load_pairs(TEST_FILE)
        metrics = evaluate(model, test_examples)
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        return 0

    # ── Load data ──────────────────────────────────────────────────
    print(f"\nLoading training data...")
    train_examples = load_pairs(TRAIN_FILE)
    test_examples = load_pairs(TEST_FILE) if TEST_FILE.exists() else []
    print(f"  Train: {len(train_examples)}, Test: {len(test_examples)}")

    # ── Load model ─────────────────────────────────────────────────
    print(f"\nLoading base model: {MODEL_NAME}")
    model = CrossEncoder(
        MODEL_NAME,
        device=device,
        local_files_only=True,
        trust_remote_code=True,
    )

    # ── Baseline evaluation ────────────────────────────────────────
    print("\n=== Baseline (before fine-tuning) ===")
    base_metrics = evaluate(model, test_examples)
    for k, v in base_metrics.items():
        print(f"  {k}: {v}")

    # ── Train ──────────────────────────────────────────────────────
    train_loader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)

    steps_per_epoch = len(train_loader)
    warmup_steps = int(steps_per_epoch * EPOCHS * WARMUP_RATIO)
    total_steps = steps_per_epoch * EPOCHS

    print(f"\n=== Training ===")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: {LEARNING_RATE}")

    t0 = time.time()

    model.fit(
        train_dataloader=train_loader,
        epochs=EPOCHS,
        warmup_steps=warmup_steps,
        optimizer_params={"lr": LEARNING_RATE},
        output_path=str(OUTPUT_DIR),
        save_best_model=True,
        show_progress_bar=True,
    )

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.0f}s ({train_time/60:.1f} min)")

    # ── Final evaluation ───────────────────────────────────────────
    print("\n=== After fine-tuning ===")
    ft_metrics = evaluate(model, test_examples)
    for k, v in ft_metrics.items():
        print(f"  {k}: {v}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n=== Summary ===")
    print(f"  Base F1:     {base_metrics['f1']:.4f}")
    print(f"  Fine-tuned F1: {ft_metrics['f1']:.4f}")
    print(f"  Improvement: {ft_metrics['f1'] - base_metrics['f1']:+.4f}")
    print(f"  Score separation: {base_metrics['score_separation']:.4f} → {ft_metrics['score_separation']:.4f}")
    print(f"  Model saved to: {OUTPUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
