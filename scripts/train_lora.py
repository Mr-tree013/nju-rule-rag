"""
QLoRA fine-tune Qwen3-8B on NJU-domain QA pairs.

Uses 4-bit quantization to fit in 16GB VRAM.
LoRA rank=8, target q_proj+v_proj, lr=5e-5.

Usage:
    python scripts/train_lora.py [--test]  # --test for 1-epoch dry run
"""

import json, sys, time
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    BitsAndBytesConfig, DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

ROOT = Path(__file__).resolve().parent.parent
TRAIN_FILE = ROOT / "data" / "training" / "all_pairs_filtered.jsonl"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
OUT_DIR = ROOT / "data" / "models" / "qwen3-8b-nju-lora"

MODEL_ID = "Qwen/Qwen3-8B"
MAX_LENGTH = 768
BATCH_SIZE = 2
GRAD_ACCUM = 4
EPOCHS = 3
LR = 5e-5


def format_example(query: str, pos_content: str) -> str:
    """Format as instruction-tuning example."""
    return f"""<|im_start|>system
你是南大学长，根据参考资料回答学弟学妹的校规问题。只使用参考资料中的信息，资料中没有的信息不要编造。如果资料不足以回答，诚实说明。<|im_end|>
<|im_start|>user
参考资料：
{pos_content}

问题：{query}<|im_end|>
<|im_start|>assistant
"""


def load_data():
    """Load training pairs with chunk content."""
    # Load chunks
    chunks = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c["content"]

    # Load training pairs
    pairs = []
    with open(TRAIN_FILE, encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            pos_id = p["positive_chunk_id"]
            if pos_id in chunks:
                pairs.append((p["query"], chunks[pos_id]))

    print(f"Loaded {len(pairs)} training examples")
    return pairs


def main():
    test_mode = "--test" in sys.argv
    if test_mode:
        print("*** TEST MODE: 1 epoch, 100 examples ***")

    # Load data
    pairs = load_data()
    if test_mode:
        pairs = pairs[:100]

    # Format as texts
    texts = []
    for query, content in pairs:
        # Truncate content for token budget
        truncated = content[:1200] if len(content) > 1200 else content
        texts.append(format_example(query, truncated))

    dataset = Dataset.from_dict({"text": texts})

    print(f"\nLoading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_LENGTH, padding=False,
        )

    dataset = dataset.map(tokenize, remove_columns=["text"])

    print(f"\nLoading model: {MODEL_ID} (4-bit QLoRA)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    epochs = 1 if test_mode else EPOCHS
    steps = len(dataset) // BATCH_SIZE * epochs

    training_args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=epochs,
        learning_rate=LR,
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        report_to="none",
        dataloader_pin_memory=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    print(f"\n=== Training ===")
    print(f"  Examples: {len(dataset)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps: ~{steps}")
    print(f"  LR: {LR}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining completed in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save LoRA adapter
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(OUT_DIR))
    tokenizer.save_pretrained(str(OUT_DIR))
    print(f"LoRA adapter saved to: {OUT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
