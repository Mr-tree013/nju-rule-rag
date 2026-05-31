"""
Three-layer LoRA evaluation (ROADMAP_LORA §6).

Layer 1: A/B test 10 canary questions (base vs LoRA)
Layer 2: Holdout loss check
Layer 3: Full eval_generation.py comparison

Usage:
    python scripts/eval_lora.py --layer1  # A/B canary test
    python scripts/eval_lora.py --layer2  # holdout loss
    python scripts/eval_lora.py --layer3  # full eval + report
    python scripts/eval_lora.py --all     # run all layers
"""

import csv, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CANARY_FILE = ROOT / "data" / "eval" / "lora_canaries.csv"
LORA_BASE = "http://localhost:8001/v1"
OLLAMA_BASE = "http://localhost:11434/v1"

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def ask_model(base_url: str, model: str, question: str) -> dict:
    """Send question to LLM endpoint and return response."""
    import requests
    resp = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "temperature": 0.15,
            "max_tokens": 400,
        },
        timeout=60,
    )
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]
    return {"content": f"ERROR: {resp.status_code}"}


def layer1_ab_test():
    """Compare base vs LoRA on 10 canary questions."""
    canaries = []
    with open(CANARY_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            canaries.append(row)
    
    print("=" * 70)
    print("Layer 1: Canary A/B Test (10 questions)")
    print("=" * 70)
    
    improved = 0
    worse = 0
    same = 0
    
    for i, c in enumerate(canaries):
        qid = c["id"]
        question = c["question"]
        expected = c["expected_behavior"]
        
        # Get base answer
        base_resp = ask_model(OLLAMA_BASE, "qwen3:8b-nothink", question)
        base_answer = base_resp.get("content", "ERROR")
        
        # Get LoRA answer
        lora_resp = ask_model(LORA_BASE, "nju-lora", question)
        lora_answer = lora_resp.get("content", "ERROR")
        
        print(f"\n[{i+1}/10] {qid}: {question[:60]}")
        print(f"  Expected: {expected}")
        print(f"  Base:    {base_answer[:150]}")
        print(f"  LoRA:    {lora_answer[:150]}")
        
        # Manual judgment needed — just print for human review
        verdict = input(f"  Verdict (b=base wins, l=lora wins, s=same, q=quit): ").strip().lower()
        if verdict == "q":
            break
        elif verdict == "l":
            improved += 1
            print(f"  {GREEN}LORA BETTER{RESET}")
        elif verdict == "b":
            worse += 1
            print(f"  {RED}BASE BETTER{RESET}")
        else:
            same += 1
            print(f"  {YELLOW}SAME{RESET}")
    
    print(f"\n--- Layer 1 Result ---")
    print(f"  Improved: {improved}, Worse: {worse}, Same: {same}")
    print(f"  Decision: ", end="")
    if improved >= 3:
        print(f"{GREEN}PROCEED to Layer 2{RESET}")
    elif improved <= 1:
        print(f"{RED}STOP — check data quality{RESET}")
    else:
        print(f"{YELLOW}MARGINAL — proceed with caution{RESET}")
    
    return improved, worse, same


def layer2_holdout():
    """Check holdout loss from training log."""
    log_path = ROOT / "data" / "lora_adapters" / "nju-v1" / "training_log.json"
    if not log_path.exists():
        print(f"{RED}ERROR: {log_path} not found. Train first.{RESET}")
        return
    
    with open(log_path) as f:
        log = json.load(f)
    
    final_loss = log.get("final_loss", 999)
    print(f"\n--- Layer 2: Holdout Loss ---")
    print(f"  Final loss: {final_loss:.4f}")
    
    if final_loss < 0.6:
        print(f"  {RED}WARN: Loss < 0.6 — possible overfitting{RESET}")
    elif 0.6 <= final_loss <= 1.2:
        print(f"  {GREEN}OK: Healthy loss range{RESET}")
    else:
        print(f"  {RED}WARN: Loss > 1.2 — model may not have learned{RESET}")
    
    return final_loss


def layer3_full_eval():
    """Run full eval_generation.py with both models and compare."""
    print(f"\n--- Layer 3: Full Eval ---")
    
    # Run base eval
    print("\nRunning base eval (Ollama)...")
    subprocess.run([
        "python", str(ROOT / "scripts" / "eval_generation.py"),
        str(ROOT / "data" / "eval" / "results.csv"),
    ], cwd=str(ROOT), env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    
    base_summary = json.load(open(ROOT / "data" / "eval" / "gen_summary.json"))
    base_f = base_summary.get("avg_faithfulness", 0)
    
    print(f"\nBase F: {base_f:.2f}")
    print("Now run with LoRA model (set LLM_BASE_URL=http://localhost:8001/v1 LLM_MODEL=nju-lora)")
    print("Then compare results.")
    
    return base_f


def main():
    if "--all" in sys.argv or "--layer1" in sys.argv:
        layer1_ab_test()
    if "--all" in sys.argv or "--layer2" in sys.argv:
        layer2_holdout()
    if "--all" in sys.argv or "--layer3" in sys.argv:
        layer3_full_eval()
    if not any(f in sys.argv for f in ["--all", "--layer1", "--layer2", "--layer3"]):
        print("Usage: python scripts/eval_lora.py [--layer1|--layer2|--layer3|--all]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
