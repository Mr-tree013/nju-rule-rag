"""
Build chunks from processed markdown files.

Reads data/processed/*.md, data/sources.csv, and optionally
data/raw/*.metadata.json.  Splits by article headings, handles
long/short chunks, and writes data/chunks/chunks.jsonl + chunk_stats.json.
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
RAW_DIR = ROOT / "data" / "raw"
CHUNKS_FILE = ROOT / "data" / "chunks" / "chunks.jsonl"
STATS_FILE = ROOT / "data" / "chunks" / "chunk_stats.json"

# ── thresholds ─────────────────────────────────────────────────────

MAX_CN = 800    # split when Chinese chars exceed this
MIN_CN = 30     # merge when Chinese chars below this
TARGET_CN = 650 # target upper bound when splitting long paragraphs

# ── noise patterns ─────────────────────────────────────────────────

NOISE_LINES = {
    "首页", "下一页", "上一页", "返回目录", "返回首页",
    "导航", "页脚", "版权声明", "版权所有",
}

NOISE_CONTENT = [
    "导航栏", "版权归", "All Rights Reserved",
]


# ── helpers ────────────────────────────────────────────────────────

def _count_cn(text: str) -> int:
    return sum(1 for ch in text if "一" <= ch <= "鿿")


def is_noise(text: str) -> bool:
    """Return True if *text* looks like nav, copyright, or blank noise."""
    stripped = text.strip()
    if not stripped:
        return True
    if len(stripped) < 10:
        return True
    for pattern in NOISE_CONTENT:
        if pattern in stripped:
            return True
    return False


def clean_text(text: str) -> str:
    """Remove page numbers, noise lines, and normalize whitespace."""
    text = re.sub(r"\s*—+\s*\d+\s*—+\s*", "\n", text)
    text = text.replace("\f", "\n")

    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in {x.lower() for x in NOISE_LINES}:
            continue
        if len(stripped) < 3 and not stripped:
            continue
        kept.append(stripped)

    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── data loading ───────────────────────────────────────────────────

def load_sources():
    """Return dict mapping filename → source metadata row."""
    by_filename = {}
    with open(SOURCES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fn = row.get("filename", "").strip()
            if fn:
                by_filename[fn] = row
    return by_filename


def load_raw_metadata(source_id: str) -> dict | None:
    """Load raw metadata.json for a source, or None if not found."""
    meta_path = RAW_DIR / f"{source_id}.metadata.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_fetched_at(source_row: dict, source_id: str) -> str:
    """
    Return fetched_at for a source.

    Priority:
      1. data/raw/{source_id}.metadata.json → fetched_at
      2. sources.csv → last_checked column (if present)
      3. current time
    """
    meta = load_raw_metadata(source_id)
    if meta and meta.get("fetched_at"):
        return str(meta["fetched_at"])

    # Some sources.csv rows may have a last_checked field
    if source_row.get("last_checked", "").strip():
        return source_row["last_checked"].strip()

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ── splitting ──────────────────────────────────────────────────────

# Matches: 第X条, 一、二、, （一）（二）, ## headings, ### headings, 1. 2. 3.
HEADING_RE = re.compile(
    r"((?:^|\n)\s*(?:"
    r"第[一二三四五六七八九十百\d]+条|"
    r"[一二三四五六七八九十]+、|"
    r"（[一二三四五六七八九十]+）|"
    r"\d{1,2}\.\s|"
    r"##\s+.+?$|"
    r"###\s+.+?$"
    r"))",
    re.MULTILINE,
)


def split_by_article(content: str) -> list[tuple[str, str]]:
    """Split content by article/heading patterns. Returns [(heading, body), ...]."""
    parts = HEADING_RE.split(content)
    chunks = []

    if not parts:
        return chunks

    preamble = parts[0].strip()
    if not is_noise(preamble):
        chunks.append(("前言", preamble))

    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lstrip("# ").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body and not is_noise(body):
            chunks.extend(_split_long(heading, body))

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
        heading = lines[0].lstrip("# ").strip() if lines else ""
        body = lines[1].strip() if len(lines) > 1 else ""
        if body and not is_noise(body):
            chunks.extend(_split_long(heading, body))
    return chunks


def split_handbook(content: str) -> list[tuple[str, str]]:
    """Split student handbook: first by ### sub-documents, then by articles."""
    sub_docs = re.split(r"\n(?=### \d+\.)", content)
    all_sections = []

    for sub_doc in sub_docs:
        sub_doc = sub_doc.strip()
        if not sub_doc:
            continue

        lines = sub_doc.split("\n", 1)
        doc_title = lines[0].lstrip("# ").strip() if lines else ""
        doc_body = lines[1] if len(lines) > 1 else ""

        if not doc_body.strip():
            continue

        doc_body = clean_text(doc_body)

        if should_use_article_split(doc_body):
            for heading, body in split_by_article(doc_body):
                label = f"{doc_title} — {heading}" if heading != "前言" else f"{doc_title} — 前言"
                all_sections.append((label, body))
        else:
            all_sections.extend(_split_long(doc_title, doc_body))

    return all_sections


def _split_long(heading: str, body: str) -> list[tuple[str, str]]:
    """Split *body* by paragraph boundaries if it exceeds MAX_CN."""
    if _count_cn(body) <= MAX_CN:
        return [(heading, body)]

    # Try paragraph-level splitting first
    paragraphs = re.split(r"\n\n+", body)
    result = []
    buf = ""
    part = 1

    for para in paragraphs:
        para = para.strip()
        if is_noise(para):
            continue

        trial = buf + ("\n\n" if buf else "") + para
        if _count_cn(trial) > TARGET_CN and buf:
            label = heading if part == 1 else f"{heading}（{part}）"
            result.extend(_force_split(label, buf))
            buf = para
            part += 1
        else:
            buf = trial

    if buf:
        label = heading if part == 1 else f"{heading}（{part}）"
        result.extend(_force_split(label, buf))

    return result


def _force_split(heading: str, text: str) -> list[tuple[str, str]]:
    """If *text* still exceeds MAX_CN, split by sentence boundaries."""
    if _count_cn(text) <= MAX_CN:
        return [(heading, text)]

    # Split on Chinese punctuation boundaries: 。；！
    sentences = re.split(r"(?<=[。；！])", text)
    chunks = []
    buf = ""
    part = 1

    for sent in sentences:
        sent = sent.strip()
        if is_noise(sent):
            continue

        trial = buf + sent
        if _count_cn(trial) > TARGET_CN and buf:
            label = heading if part == 1 else f"{heading}（{part}）"
            chunks.append((label, buf))
            buf = sent
            part += 1
        else:
            buf = trial

    if buf:
        label = heading if part == 1 else f"{heading}（{part}）"
        chunks.append((label, buf))

    return chunks if chunks else [(heading, text)]


def _merge_short(chunks: list[dict]) -> list[dict]:
    """
    Merge adjacent short chunks from the same source.

    First pass (backward): merge short chunk into previous.
    Second pass (forward): if the first chunk is still short, merge next into it.
    """
    if not chunks:
        return chunks

    # Backward pass
    merged = []
    for c in chunks:
        if (
            merged
            and _count_cn(c["content"]) < MIN_CN
            and c["source_id"] == merged[-1]["source_id"]
        ):
            prev = merged[-1]
            prev["content"] = prev["content"] + "\n\n" + c["content"]
        else:
            merged.append(c)

    # Forward pass: if first chunk(s) are still short, merge with next
    # Work from the end so indices stay valid
    i = len(merged) - 2
    while i >= 0:
        curr = merged[i]
        nxt = merged[i + 1]
        if (
            _count_cn(curr["content"]) < MIN_CN
            and curr["source_id"] == nxt["source_id"]
        ):
            nxt["content"] = curr["content"] + "\n\n" + nxt["content"]
            nxt["article"] = curr["article"]  # keep the first chunk's article tag
            merged.pop(i)
        i -= 1

    return merged


def is_handbook(content: str) -> bool:
    return len(re.findall(r"\n### \d+\.", content)) >= 5


def should_use_article_split(content: str) -> bool:
    return bool(re.search(r"第[一二三四五六七八九十百\d]+条", content))


# ── main ───────────────────────────────────────────────────────────

def main():
    if not PROCESSED_DIR.exists():
        print(f"Error: {PROCESSED_DIR} not found", file=sys.stderr)
        sys.exit(1)

    sources_by_filename = load_sources()
    if not sources_by_filename:
        print("Error: no sources with filename in sources.csv", file=sys.stderr)
        sys.exit(1)

    processed_files = sorted(PROCESSED_DIR.glob("*.md"))
    if not processed_files:
        print(f"No .md files found in {PROCESSED_DIR}")
        sys.exit(1)

    all_chunks = []
    matched = 0
    unmatched = []

    for md_path in processed_files:
        filename = md_path.name
        source_row = sources_by_filename.get(filename)
        if source_row is None:
            unmatched.append(filename)
            continue

        matched += 1
        source_id = source_row["source_id"]
        fetched_at = get_fetched_at(source_row, source_id)
        content = md_path.read_text(encoding="utf-8")
        content = clean_text(content)

        if is_handbook(content):
            chunk_parts = split_handbook(content)
        elif should_use_article_split(content):
            chunk_parts = split_by_article(content)
        else:
            chunk_parts = split_by_markdown_sections(content)

        source_chunks = []
        for i, (heading, body) in enumerate(chunk_parts):
            body = clean_text(body)
            if is_noise(body):
                continue

            source_chunks.append({
                "chunk_id": "",   # assigned after merge
                "source_id": source_id,
                "title": source_row["title"],
                "url": source_row.get("url", ""),
                "department": source_row.get("department", ""),
                "scope": source_row.get("scope", ""),
                "priority": int(source_row.get("priority", 5)),
                "article": heading,
                "content": body,
                "fetched_at": fetched_at,
            })

        # Merge and assign stable IDs
        merged = _merge_short(source_chunks)
        for idx, chunk in enumerate(merged):
            chunk["chunk_id"] = f"{source_id}-{idx:04d}"

        all_chunks.extend(merged)

    # ── final noise check across all chunks ────────────────────────

    all_chunks = [c for c in all_chunks if not is_noise(c["content"])]

    # ── write chunks.jsonl ─────────────────────────────────────────

    CHUNKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # ── stats ──────────────────────────────────────────────────────

    cn_lengths = [_count_cn(c["content"]) for c in all_chunks]
    per_source: dict[str, int] = {}
    per_priority: dict[str, int] = {}
    for c in all_chunks:
        sid = c["source_id"]
        per_source[sid] = per_source.get(sid, 0) + 1
        p = str(c["priority"])
        per_priority[p] = per_priority.get(p, 0) + 1

    too_short = sum(1 for n in cn_lengths if n < MIN_CN)
    too_long = sum(1 for n in cn_lengths if n > MAX_CN)

    stats = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_chunks": len(all_chunks),
        "avg_chars": round(sum(len(c["content"]) for c in all_chunks) / len(all_chunks)) if all_chunks else 0,
        "avg_cn_chars": round(sum(cn_lengths) / len(all_chunks)) if all_chunks else 0,
        "min_cn_chars": min(cn_lengths) if cn_lengths else 0,
        "max_cn_chars": max(cn_lengths) if cn_lengths else 0,
        "too_short_cn": too_short,
        "too_long_cn": too_long,
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
    print(f"  too short (<{MIN_CN}cn): {too_short}")
    print(f"  too long  (>{MAX_CN}cn): {too_long}")
    print(f"Written: {CHUNKS_FILE}")
    print(f"Stats:   {STATS_FILE}")


if __name__ == "__main__":
    main()
