import requests, csv, time, json

with open("data/eval/questions.csv", encoding="utf-8-sig") as f:
    questions = list(csv.DictReader(f))

results = []
t_start = time.time()
for i, row in enumerate(questions):
    q = row["question"]
    t0 = time.time()
    try:
        resp = requests.post("http://localhost:8000/ask", json={"question": q}, timeout=30)
        lat = time.time() - t0
        d = resp.json()
        cached = d["debug"].get("cached", False)
        gen = d["debug"]["timing"]["generate_ms"]
        rr = d["debug"]["timing"]["rerank_ms"]
        ans_len = len(d.get("answer", ""))
        kw = row.get("expected_source_keyword", "")
        kw_hit = kw and kw in d.get("answer", "") if kw else False
        results.append({"lat": lat, "cached": cached, "gen": gen, "ans_len": ans_len})
    except Exception as e:
        lat = time.time() - t0
        results.append({"lat": lat, "cached": False, "gen": 0, "ans_len": 0, "error": str(e)})
    
    if (i+1) % 30 == 0:
        live_avg = sum(r["lat"] for r in results if not r.get("cached")) / max(1, len([r for r in results if not r.get("cached")]))
        print(f"[{i+1}/144] live avg: {live_avg:.1f}s")

live = [r for r in results if not r.get("cached")]
cached = [r for r in results if r.get("cached")]
errors = [r for r in results if "error" in r]
total = time.time() - t_start

print(f"\n=== Windows 144Q Eval ===")
print(f"Total questions: {len(results)}")
print(f"Live (uncached): {len(live)}")
print(f"Cached: {len(cached)}")
print(f"Errors: {len(errors)}")
if live:
    avg_lat = sum(r["lat"] for r in live) / len(live)
    avg_gen = sum(r["gen"] for r in live) / len(live)
    retries = len([r for r in live if r["gen"] > 6000])
    print(f"Avg latency: {avg_lat:.1f}s")
    print(f"Avg generate: {avg_gen:.0f}ms")
    print(f"Retries: {retries}/{len(live)} ({retries/len(live)*100:.0f}%)")
    print(f"Fastest: {min(r['lat'] for r in live):.1f}s")
    print(f"Slowest: {max(r['lat'] for r in live):.1f}s")
print(f"Total wall time: {total:.0f}s")

# Save
with open("data/eval/win_results.json", "w") as f:
    json.dump({"avg_latency": avg_lat if live else 0, "total": len(results), "live": len(live)}, f)
print("Saved to data/eval/win_results.json")
