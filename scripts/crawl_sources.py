"""
Crawl public web pages and PDFs listed in data/sources.csv.

Skips sources that require login or have no URL.
Saves raw files and per-source metadata to data/raw/.
"""

import csv
import hashlib
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = ROOT / "data" / "sources.csv"
RAW_DIR = ROOT / "data" / "raw"

USER_AGENT = (
    "NJU-Rule-RAG-Bot/0.1 "
    "(student project; contact: see repository README)"
)

REQUEST_TIMEOUT = 30  # seconds
DELAY = 1.0           # seconds between requests


def sha256_hex(file_path: Path) -> str:
    """Return SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def crawl() -> int:
    """Run crawl.  Returns 0 on success, 1 on fatal errors."""

    if not SOURCES_CSV.exists():
        print(f"Error: {SOURCES_CSV} not found.", file=sys.stderr)
        return 1

    with open(SOURCES_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # counters
    success = 0
    failed = 0
    skipped = 0

    for row in sources:
        source_id = row.get("source_id", "").strip()
        title = row.get("title", "").strip()
        url = row.get("url", "").strip()
        source_type = row.get("source_type", "").strip().lower()
        need_login = row.get("need_login", "").strip().lower()
        department = row.get("department", "").strip()
        scope = row.get("scope", "").strip()
        priority_str = row.get("priority", "").strip()

        try:
            priority = int(priority_str) if priority_str else 5
        except ValueError:
            priority = 5

        # ── skip logic ────────────────────────────────────────────

        if not source_id:
            print("[WARN] Row missing source_id, skipping.")
            skipped += 1
            continue

        if need_login == "yes":
            print(f"[SKIP] {source_id}: need_login=yes")
            skipped += 1
            continue

        if not url:
            print(f"[SKIP] {source_id}: no URL (local file, skip crawl)")
            skipped += 1
            continue

        # ── determine output path ──────────────────────────────────

        # Use source_type or guess from URL suffix.
        if source_type == "pdf" or url.lower().endswith(".pdf"):
            ext = ".pdf"
        else:
            ext = ".html"

        output_file = RAW_DIR / f"{source_id}{ext}"
        metadata_file = RAW_DIR / f"{source_id}.metadata.json"

        # ── fetch ──────────────────────────────────────────────────

        metadata = {
            "source_id": source_id,
            "title": title,
            "url": url,
            "department": department,
            "scope": scope,
            "priority": priority,
            "status_code": None,
            "content_type": None,
            "fetched_at": None,
            "sha256": None,
            "file_path": str(output_file),
            "error": None,
        }

        try:
            print(f"[FETCH] {source_id}: {url}")
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT},
                timeout=REQUEST_TIMEOUT,
                allow_redirects=True,
            )
            metadata["status_code"] = resp.status_code
            metadata["content_type"] = resp.headers.get("Content-Type", "")

            if resp.status_code != 200:
                metadata["error"] = f"HTTP {resp.status_code}"
                print(f"  [FAIL] HTTP {resp.status_code}")
                failed += 1
            else:
                output_file.write_bytes(resp.content)
                metadata["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                metadata["sha256"] = sha256_hex(output_file)
                print(f"  [OK] saved to {output_file} ({len(resp.content)} bytes)")
                success += 1

        except requests.Timeout:
            metadata["error"] = "timeout"
            print(f"  [FAIL] Request timed out after {REQUEST_TIMEOUT}s")
            failed += 1
        except requests.ConnectionError as exc:
            metadata["error"] = f"connection error: {exc}"
            print(f"  [FAIL] Connection error: {exc}")
            failed += 1
        except requests.RequestException as exc:
            metadata["error"] = f"request error: {exc}"
            print(f"  [FAIL] {exc}")
            failed += 1

        # ── save metadata (always, even on failure) ───────────────

        import json
        metadata_file.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # ── polite delay ──────────────────────────────────────────

        time.sleep(DELAY)

    # ── summary ──────────────────────────────────────────────────

    total = success + failed + skipped
    print()
    print("─" * 50)
    print(f"Crawl complete. Total sources: {total}")
    print(f"  Success: {success}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    print(f"  Output:  {RAW_DIR}/")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(crawl())
