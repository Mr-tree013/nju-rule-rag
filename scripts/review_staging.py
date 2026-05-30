"""
Review staged documents and accept/reject into corpus (F.3).

Usage:
    python scripts/review_staging.py          # interactive review
    python scripts/review_staging.py --list   # list staged documents
    python scripts/review_staging.py --accept STAGED_FILE  # auto-accept one file
"""

import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STAGING_DIR = Path(os.getenv("STAGING_DIR", ROOT / "data" / "staging"))
PROCESSED_DIR = ROOT / "data" / "processed"
ARCHIVE_DIR = Path(os.getenv("ARCHIVE_DIR", ROOT / "data" / "archive"))
SOURCES_CSV = ROOT / "data" / "sources.csv"
STAGING_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def list_staged():
    """Return list of (md_file, meta_file) tuples sorted by time."""
    items = []
    for f in sorted(STAGING_DIR.glob("*.md"), reverse=True):
        meta = Path(str(f).replace(".md", ".meta.json"))
        if not meta.exists():
            meta = None
        items.append((f, meta))
    return items


def show_document(md_file: Path, meta_file: Path | None):
    """Display document for review."""
    print(f"\n{'='*60}")
    print(f"File: {md_file.name}")
    if meta_file and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        print(f"URL: {meta.get('url', '?')}")
        print(f"Hash: {meta.get('content_hash', '?')[:16]}...")
        print(f"Fetched: {meta.get('fetched_at', '?')}")
    print(f"Size: {md_file.stat().st_size} bytes")
    print(f"{'='*60}")

    content = md_file.read_text(encoding="utf-8")
    # Show first 40 lines
    lines = content.split("\n")
    for line in lines[:40]:
        print(line)
    if len(lines) > 40:
        print(f"\n... ({len(lines) - 40} more lines)")
    print(f"{'='*60}")


def generate_source_id(md_file: Path, meta_file: Path | None) -> str:
    """Generate a source_id from the document."""
    # Try to extract from URL or content
    if meta_file and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        url = meta.get("url", "")
        if "nju.edu.cn" in url:
            # Extract path-based ID
            parts = url.replace("https://", "").replace("http://", "").split("/")
            if len(parts) > 2:
                return f"nju-web-{parts[1]}-{meta.get('content_hash','')[:6]}"
    # Fall back to hash-based
    import hashlib
    h = hashlib.sha256(str(md_file).encode()).hexdigest()[:8]
    return f"nju-web-{h}"


def register_source(source_id: str, title: str, filename: str, url: str = "",
                    department: str = "", scope: str = "本科生", priority: int = 3,
                    topics: str = "") -> None:
    """Add a row to sources.csv."""
    with open(SOURCES_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    new_row = {fn: "" for fn in fieldnames}
    new_row.update({
        "source_id": source_id,
        "title": title,
        "filename": filename,
        "url": url,
        "source_type": "markdown",
        "department": department,
        "scope": scope,
        "priority": str(priority),
        "need_login": "no",
        "update_frequency": "monthly",
        "note": f"auto-ingested via review_staging {datetime.now().strftime('%Y-%m-%d')}",
        "topics": topics or department,
        "crawl_url": url,
        "crawl_method": "static",
        "crawl_frequency": "monthly",
        "last_crawled_at": datetime.now().strftime("%Y-%m-%d"),
        "content_hash": "",
        "stale_after_days": "180",
        "auth_required": "false",
        "chunk_strategy": "heading",
    })
    rows.append(new_row)

    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sources.csv] Registered {source_id}")


def accept_document(md_file: Path, meta_file: Path | None, auto: bool = False):
    """Accept document: move to processed/, register in sources.csv."""
    url = ""
    title = ""
    department = ""
    if meta_file and meta_file.exists():
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        url = meta.get("url", "")
        title = meta.get("title", "")

    content = md_file.read_text(encoding="utf-8")
    if not title:
        # Extract title from first heading
        for line in content.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
    if not title:
        title = md_file.stem

    source_id = generate_source_id(md_file, meta_file)

    # Copy to processed/
    dest = PROCESSED_DIR / md_file.name
    shutil.copy2(str(md_file), str(dest))

    # Register
    register_source(source_id, title, md_file.name, url=url, department=department)

    # Remove from staging
    md_file.unlink(missing_ok=True)
    if meta_file:
        meta_file.unlink(missing_ok=True)

    print(f"[ACCEPT] {title}")
    print(f"  source_id: {source_id}")
    print(f"  file: {dest}")
    print(f"  Next: PYTHONPATH=. python scripts/build_chunks.py && python scripts/build_index.py")


def reject_document(md_file: Path, meta_file: Path | None):
    """Reject document: archive it."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"rejected_{ts}_{md_file.name}"
    shutil.move(str(md_file), str(ARCHIVE_DIR / archive_name))
    if meta_file and meta_file.exists():
        shutil.move(str(meta_file), str(ARCHIVE_DIR / archive_name.replace(".md", ".meta.json")))
    print(f"[REJECT] Archived to {ARCHIVE_DIR / archive_name}")


def interactive_review():
    """Interactive review loop."""
    items = list_staged()
    if not items:
        print("No staged documents. Submit URLs via POST /admin/ingest_url")
        return

    print(f"Found {len(items)} staged document(s).\n")
    for i, (md_file, meta_file) in enumerate(items):
        show_document(md_file, meta_file)
        while True:
            action = input("\n[a]ccept / [r]eject / [s]kip / [q]uit? ").lower().strip()
            if action in ("a", "accept"):
                accept_document(md_file, meta_file)
                break
            elif action in ("r", "reject"):
                reject_document(md_file, meta_file)
                break
            elif action in ("s", "skip"):
                print(f"[SKIP] {md_file.name}")
                break
            elif action in ("q", "quit"):
                print("Quit.")
                return
            else:
                print("Invalid choice. Enter a/r/s/q.")


def main():
    if "--list" in sys.argv:
        items = list_staged()
        print(f"Staged documents: {len(items)}")
        for md_file, _ in items:
            print(f"  {md_file.name} ({md_file.stat().st_size} bytes)")
        return 0

    if "--accept" in sys.argv:
        idx = sys.argv.index("--accept")
        if idx + 1 < len(sys.argv):
            fname = sys.argv[idx + 1]
            md_file = STAGING_DIR / fname
            if not md_file.exists():
                print(f"Error: {md_file} not found", file=sys.stderr)
                return 1
            meta_file = Path(str(md_file).replace(".md", ".meta.json"))
            if not meta_file.exists():
                meta_file = None
            accept_document(md_file, meta_file)
            return 0

    # Interactive mode
    interactive_review()
    return 0


if __name__ == "__main__":
    sys.exit(main())
