"""
Build chunks from processed markdown files.

Reads data/processed/*.md, splits by article headings, cleans artifacts,
and writes data/chunks/chunks.jsonl with metadata from data/sources.csv.
"""

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCES_CSV = ROOT / "data" / "sources.csv"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"


def load_sources():
    """Return dict mapping filename → source metadata."""
    by_filename = {}
    with open(SOURCES_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("filename", "").strip()
            if fn:
                by_filename[fn] = row
    return by_filename


def clean_text(text):
    """Remove page numbers, form feeds, and normalize whitespace."""
    text = re.sub(r"\s*—+\s*\d+\s*—+\s*", "\n", text)
    text = text.replace("\f", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    return text.strip()


def split_by_article(content):
    """
    Split content by Chinese article headings.
    Returns list of (heading, body) tuples.
    """
    heading_pattern = re.compile(
        r"((?:^|\n)\s*(?:第[一二三四五六七八九十百\d]+条|"
        r"[一二三四五六七八九十]+、|"
        r"（[一二三四五六七八九十]+）|"
        r"##\s+.+?$|"
        r"###\s+.+?$))",
        re.MULTILINE,
    )

    parts = heading_pattern.split(content)
    chunks = []

    if not parts:
        return chunks

    preamble = parts[0].strip()
    if preamble:
        chunks.append(("前言", preamble))

    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lstrip("#").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            chunks.append((heading, body))

    return chunks


def split_by_markdown_sections(content):
    """Split by ## level sections for non-regulatory docs like 办事流程."""
    sections = re.split(r"\n(?=## \d+\.)", content)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n", 1)
        heading = lines[0].lstrip("#").strip() if lines else ""
        body = lines[1].strip() if len(lines) > 1 else ""
        if body:
            chunks.append((heading, body))
    return chunks


def should_use_article_split(content):
    """Check if the document uses article-style headings (第X条)."""
    return bool(re.search(r"第[一二三四五六七八九十百\d]+条", content))


def main():
    if not PROCESSED_DIR.exists():
        print(f"Error: {PROCESSED_DIR} not found", file=sys.stderr)
        sys.exit(1)

    sources_by_filename = load_sources()
    if not sources_by_filename:
        print("Error: no sources with filename in sources.csv", file=sys.stderr)
        sys.exit(1)

    processed_files = list(PROCESSED_DIR.glob("*.md"))
    if not processed_files:
        print(f"No .md files found in {PROCESSED_DIR}")
        sys.exit(1)

    all_chunks = []
    matched = 0
    unmatched = []

    for md_path in sorted(processed_files):
        filename = md_path.name
        source_row = sources_by_filename.get(filename)

        if source_row is None:
            unmatched.append(filename)
            continue

        matched += 1
        source_id = source_row["source_id"]
        content = md_path.read_text(encoding="utf-8")

        content = clean_text(content)

        if should_use_article_split(content):
            chunk_parts = split_by_article(content)
        else:
            chunk_parts = split_by_markdown_sections(content)

        for i, (heading, body) in enumerate(chunk_parts):
            body = clean_text(body)
            if len(body) < 20:
                continue

            chunk = {
                "chunk_id": f"{source_id}-{i:04d}",
                "source_id": source_id,
                "title": source_row["title"],
                "url": source_row.get("url", ""),
                "department": source_row.get("department", ""),
                "scope": source_row.get("scope", ""),
                "priority": int(source_row.get("priority", 5)),
                "section": heading,
                "content": body,
            }
            all_chunks.append(chunk)

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Matched: {matched} files")
    print(f"Unmatched: {unmatched if unmatched else 'none'}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Written to: {CHUNKS_FILE}")


if __name__ == "__main__":
    main()
