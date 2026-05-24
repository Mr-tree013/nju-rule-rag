"""
Build chunks from processed markdown files.

Reads data/processed/*.md, splits by article headings, cleans artifacts,
outputs data/chunks/chunks.jsonl and data/chunks/chunk_stats.json.
"""

import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
SOURCES_CSV = ROOT / "data" / "sources.csv"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
STATS_FILE = ROOT / "data" / "chunks" / "chunk_stats.json"

# ── chunk size thresholds ────────────────────────────────────────

MAX_CN_CHARS = 1000   # split body if Chinese chars exceed this
MIN_CN_CHARS = 20     # skip body if Chinese chars below this

# ── noise ─────────────────────────────────────────────────────────

NOISE_LINES = {
    "首页", "下一页", "上一页", "返回目录", "返回首页",
    "导航", "页脚", "版权声明",
}


# ── helpers ──────────────────────────────────────────────────────

def load_sources():
    """Return dict mapping filename → source metadata."""
    by_filename = {}
    with open(SOURCES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if fn:
                by_filename[fn] = row
    return by_filename


def _count_cn(text: str) -> int:
    return sum(1 for ch in text if "一" <= ch <= "鿿")


def clean_text(text: str) -> str:
    """Remove page numbers, noise lines, and normalize whitespace."""
    # page numbers like "— 12 —"
    text = re.sub(r"\s*—+\s*\d+\s*—+\s*", "\n", text)
    text = text.replace("\f", "\n")

    # drop lines that are pure noise
    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {x.lower() for x in NOISE_LINES}:
            continue
        kept.append(stripped)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── splitting ────────────────────────────────────────────────────

def split_by_article(content: str) -> list[tuple[str, str]]:
    """Split content by Chinese article headings (第X条, 一、, （一）)."""
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
            chunks.extend(_maybe_split_long(heading, body))

    return chunks


def split_by_markdown_sections(content: str) -> list[tuple[str, str]]:
    """Split by ## N. sections for non-regulatory docs."""
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
            chunks.extend(_maybe_split_long(heading, body))
    return chunks


def split_handbook(content: str) -> list[tuple[str, str]]:
    """Split student handbook: ### sub-documents → articles within each."""
    sub_docs = re.split(r"\n(?=### \d+\.)", content)
    all_sections = []

    for sub_doc in sub_docs:
        sub_doc = sub_doc.strip()
        if not sub_doc:
            continue

        lines = sub_doc.split("\n", 1)
        doc_title = lines[0].lstrip("#").strip() if lines else ""
        doc_body = lines[1] if len(lines) > 1 else ""

        if not doc_body.strip():
            continue

        doc_body = clean_text(doc_body)

        if should_use_article_split(doc_body):
            sub_chunks = split_by_article(doc_body)
            for heading, body in sub_chunks:
                prefix = f"{doc_title} — {heading}" if heading != "前言" else f"{doc_title} — 前言"
                all_sections.append((prefix, body))
        else:
            all_sections.extend(_maybe_split_long(doc_title, doc_body))

    return all_sections


def _maybe_split_long(heading: str, body: str) -> list[tuple[str, str]]:
    """If *body* exceeds MAX_CN_CHARS, split by paragraph boundaries."""
    if _count_cn(body) <= MAX_CN_CHARS:
        return [(heading, body)]

    paragraphs = re.split(r"\n\n+", body)
    result = []
    buf = ""
    part = 1

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        trial = buf + ("\n\n" if buf else "") + para
        if _count_cn(trial) > MAX_CN_CHARS and buf:
            label = f"{heading}（{part}）" if part > 1 else heading
            result.append((label, buf))
            buf = para
            part += 1
        else:
            buf = trial

    if buf:
        label = f"{heading}（{part}）" if part > 1 else heading
        result.append((label, buf))

    return result


def is_handbook(content: str) -> bool:
    return len(re.findall(r"\n### \d+\.", content)) >= 5


def should_use_article_split(content: str) -> bool:
    return bool(re.search(r"第[一二三四五六七八九十百\d]+条", content))


# ── main ─────────────────────────────────────────────────────────

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

        if is_handbook(content):
            chunk_parts = split_handbook(content)
        elif should_use_article_split(content):
            chunk_parts = split_by_article(content)
        else:
            chunk_parts = split_by_markdown_sections(content)

        # Build chunks, skipping those with too-short bodies.
        source_chunks = []
        for i, (heading, body) in enumerate(chunk_parts):
            body = clean_text(body)
            if _count_cn(body) < MIN_CN_CHARS:
                continue  # too short, skip

            source_chunks.append({
                "chunk_id": f"{source_id}-{i:04d}",
                "source_id": source_id,
                "title": source_row["title"],
                "url": source_row.get("url", ""),
                "department": source_row.get("department", ""),
                "scope": source_row.get("scope", ""),
                "priority": int(source_row.get("priority", 5)),
                "section": heading,
                "content": body,
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

        all_chunks.extend(source_chunks)

    # ── write chunks.jsonl ───────────────────────────────────────

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── stats ────────────────────────────────────────────────────

    lengths = [len(c["content"]) for c in all_chunks]
    cn_lengths = [_count_cn(c["content"]) for c in all_chunks]
    per_source: dict[str, int] = {}
    per_priority: dict[str, int] = {}
    for c in all_chunks:
        sid = c["source_id"]
        per_source[sid] = per_source.get(sid, 0) + 1
        p = str(c["priority"])
        per_priority[p] = per_priority.get(p, 0) + 1

    stats = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(all_chunks),
        "avg_chars": round(sum(lengths) / len(all_chunks)) if all_chunks else 0,
        "avg_cn_chars": round(sum(cn_lengths) / len(all_chunks)) if all_chunks else 0,
        "min_cn_chars": min(cn_lengths) if cn_lengths else 0,
        "max_cn_chars": max(cn_lengths) if cn_lengths else 0,
        "too_short_cn": sum(1 for n in cn_lengths if n < MIN_CN_CHARS),
        "too_long_cn": sum(1 for n in cn_lengths if n > MAX_CN_CHARS),
        "per_source": per_source,
        "per_priority": per_priority,
        "sources_matched": matched,
        "sources_unmatched": unmatched,
    }

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Matched: {matched} files")
    print(f"Unmatched: {unmatched if unmatched else 'none'}")
    print(f"Total chunks: {len(all_chunks)}")
    print(f"Written: {CHUNKS_FILE}")
    print(f"Stats:   {STATS_FILE}")


if __name__ == "__main__":
    main()
