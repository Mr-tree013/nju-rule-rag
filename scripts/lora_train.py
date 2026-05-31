"""
QLoRA fine-tune Qwen3-8B for NJU Rule RAG (PR #2).

Follows ROADMAP_LORA hyperparameters:
  rank=16, lr=1e-4, QLoRA 4-bit NF4, target q_proj/k_proj/v_proj/o_proj
  batch=1, grad_accum=16, epochs=2, max_seq_length=2048

Usage:
    python scripts/lora_train.py --debug --max_samples=50 --epochs=1  # dry-run
    python scripts/lora_train.py                                      # full training
"""

import json, os, sys, time
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "training" / "lora_train.jsonl"
HOLDOUT_FILE = ROOT / "data" / "training" / "lora_holdout.jsonl"
OUTPUT_DIR = ROOT / "data" / "lora_adapters" / "nju-v1"

# Hyperparameters from ROADMAP_LORA §3.1
MODEL_ID = "Qwen/Qwen3-8B"
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 16
EPOCHS = 2
LR = 1e-4
WARMUP_RATIO = 0.05


def format_chat(example):
    """Format as Qwen3 chat template with label masking."""
    system = "你是南大学长，根据参考资料回答学弟学妹的校规问题。只用参考资料中的信息，不要编造。资料不足以回答时，诚实说'这个问题我看到的资料里没写'。"
    user = f"问题：{example['query']}"
    answer = example.get("answer", "")
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]
    return {"text": messages}


def load_data(path: Path, max_samples: int = 0):
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            if p.get("answer"):
                pairs.append(p)
    if max_samples and max_samples < len(pairs):
        pairs = pairs[:max_samples]
    return pairs


def main():
    debug = "--debug" in sys.argv
    max_samples = 0
    epochs = EPOCHS
    for arg in sys.argv[1:]:
        if arg.startswith("--max_samples="):
            max_samples = int(arg.split("=", 1)[1])
        elif arg.startswith("--epochs="):
            epochs = int(arg.split("=", 1)[1])

    if debug:
        max_samples = max_samples or 50
        epochs = 1
        print(f"*** DEBUG MODE: {max_samples} samples, {epochs} epoch ***")

    # Load data
    train_pairs = load_data(TRAIN_FILE, max_samples)
    holdout_pairs = load_data(HOLDOUT_FILE, max_samples // 6 if max_samples else 0)
    print(f"Train: {len(train_pairs)}, Holdout: {len(holdout_pairs)}")

    # Format as chat
    train_data = [format_chat(p) for p in train_pairs]
    eval_data = [format_chat(p) for p in holdout_pairs] if holdout_pairs else None

    # Load tokenizer
    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        messages = example["text"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        result = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)
        # Label masking: only compute loss on assistant tokens
        # Find where the assistant message starts
        assistant_start = text.rfind("<|im_start|>assistant")
        if assistant_start >= 0:
            input_ids = result["input_ids"]
            # Tokenize the prefix (everything before assistant response)
            prefix = text[:assistant_start + len("<|im_start|>assistant\n")]
            prefix_tokens = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            labels = [-100] * len(prefix_tokens) + input_ids[len(prefix_tokens):]
        else:
            labels = input_ids.copy()
        result["labels"] = labels
        return result

    ds_train = Dataset.from_list(train_data).map(tokenize, remove_columns=["text"])
    ds_eval = Dataset.from_list(eval_data).map(tokenize, remove_columns=["text"]) if eval_data else None

    # Load model with 4-bit quantization
    print(f"\nLoading model: {MODEL_ID} (4-bit QLoRA)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training args
    steps_per_epoch = len(ds_train) // (BATCH_SIZE * GRAD_ACCUM)
    output_dir = str(OUTPUT_DIR)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        eval_strategy="steps" if ds_eval else "no",
        eval_steps=100 if ds_eval else None,
        load_best_model_at_end=True if ds_eval else False,
        metric_for_best_model="eval_loss" if ds_eval else None,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        dataloader_pin_memory=False,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    callbacks = []
    if ds_eval:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    steps = steps_per_epoch * epochs
    print(f"\n=== Training ===")
    print(f"  Examples: {len(ds_train)} train + {len(ds_eval) if ds_eval else 0} eval")
    print(f"  Epochs: {epochs}")
    print(f"  Batch: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM} eff")
    print(f"  Steps: ~{steps} ({steps_per_epoch}/epoch)")
    print(f"  LR: {LR}")
    print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save adapter
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"Adapter saved to: {OUTPUT_DIR}")
    print(f"Size: {sum(f.stat().st_size for f in OUTPUT_DIR.rglob('*') if f.is_file()) / 1e6:.1f} MB")

    # Save training log
    log = {
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_examples": len(ds_train),
        "eval_examples": len(ds_eval) if ds_eval else 0,
        "epochs": epochs,
        "training_time_s": elapsed,
        "final_loss": trainer.state.log_history[-1].get("loss", 0) if trainer.state.log_history else 0,
    }
    with open(OUTPUT_DIR / "training_log.json", "w") as f:
        json.dump(log, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
