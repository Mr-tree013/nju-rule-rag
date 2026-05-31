"""
URL fetcher with change detection (F.1 + F.2).

Downloads URLs, converts HTML to markdown, computes content hash,
compares with sources.csv to detect changes. Only saves to staging
if content has actually changed. Supports ETag/If-Modified-Since
for zero-cost change detection on supported servers.
"""

import csv
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent.parent

USER_AGENT = (
    "NJU-Rule-RAG-Bot/0.6 "
    "(student project; contact: see repository README)"
)
REQUEST_TIMEOUT = 30
DELAY = 1.5  # polite delay between requests


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_source_hashes() -> dict[str, str]:
    """Load content_hash from sources.csv. Returns {source_id: hash}."""
    sources_path = ROOT / "data" / "sources.csv"
    if not sources_path.exists():
        return {}
    hashes = {}
    with open(sources_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            sid = row.get("source_id", "").strip()
            ch = row.get("content_hash", "").strip()
            if sid and ch:
                hashes[sid] = ch
    return hashes


def fetch_url(url: str, etag: str = "", last_modified: str = "") -> dict[str, Any]:
    """Fetch a URL with optional conditional headers.

    Returns dict with keys: status_code, text, content_type, etag, last_modified, not_modified.
    """
    headers = {"User-Agent": USER_AGENT}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified

    resp = requests.get(
        url,
        headers=headers,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )

    result = {
        "status_code": resp.status_code,
        "text": "",
        "content_type": resp.headers.get("Content-Type", ""),
        "etag": resp.headers.get("ETag", ""),
        "last_modified": resp.headers.get("Last-Modified", ""),
        "not_modified": resp.status_code == 304,
    }

    if resp.status_code == 304:
        return result

    resp.raise_for_status()

    # Decode text
    try:
        result["text"] = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            result["text"] = resp.content.decode("gbk", errors="replace")
        except Exception:
            result["text"] = resp.content.decode("utf-8", errors="replace")

    return result


def html_to_markdown(html: str, url: str = "") -> str:
    """Convert HTML to plain markdown. Strips nav/footer/header, preserves structure."""
    # Remove non-content blocks
    for tag in ["script", "style", "nav", "footer", "header"]:
        html = re.sub(
            rf"<{tag}[^>]*>.*?</{tag}>", "", html,
            flags=re.DOTALL | re.IGNORECASE,
        )

    # Convert headings
    for level in range(6, 0, -1):
        html = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            rf"\n\n{'#' * level} \1\n\n",
            html, flags=re.DOTALL | re.IGNORECASE,
        )

    # Convert paragraphs and line breaks
    html = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", html, flags=re.DOTALL | re.IGNORECASE)

    # Remove remaining tags
    html = re.sub(r"<[^>]+>", "", html)

    # Decode common entities
    for entity, char in [("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">"),
                          ("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'")]:
        html = html.replace(entity, char)

    # Clean whitespace
    html = re.sub(r"\n{3,}", "\n\n", html)
    lines = [line.strip() for line in html.split("\n") if line.strip()]
    body = "\n\n".join(lines)

    # Extract title
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    return f"# {title}\n\n> 来源：{url}\n> 抓取时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{body}"


def fetch_and_stage(
    url: str,
    source_id: str = "",
    staging_dir: Path | None = None,
    known_hash: str = "",
    etag: str = "",
    last_modified: str = "",
) -> dict[str, Any]:
    """Fetch URL, detect changes, save to staging if content changed.

    Returns metadata dict with status: 'unchanged', 'staged', 'new', 'error'.
    """
    if staging_dir is None:
        staging_dir = ROOT / "data" / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    sid = source_id or f"url_{sha256_text(url)[:8]}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    result: dict[str, Any] = {
        "source_id": sid,
        "url": url,
        "status": "error",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "content_hash": "",
        "old_hash": known_hash,
        "file_path": "",
        "change_detected": False,
        "error": "",
    }

    try:
        raw = fetch_url(url, etag=etag, last_modified=last_modified)

        if raw["not_modified"]:
            result["status"] = "unchanged"
            result["note"] = "Server returned 304 Not Modified"
            return result

        text = raw["text"]
        result["status_code"] = raw["status_code"]
        result["content_type"] = raw["content_type"]
        new_hash = sha256_text(text)
        result["content_hash"] = new_hash

        # Check if content actually changed
        if known_hash and new_hash == known_hash:
            result["status"] = "unchanged"
            result["note"] = "Content hash identical to stored hash"
            return result

        # Convert to markdown
        if "text/html" in raw["content_type"] or raw["content_type"] == "":
            md = html_to_markdown(text, url)
        else:
            md = text

        # Save to staging
        md_file = staging_dir / f"{sid}_{ts}.md"
        md_file.write_text(md, encoding="utf-8")
        result["file_path"] = str(md_file)
        result["content_length"] = len(md)
        result["change_detected"] = True
        result["etag"] = raw["etag"]
        result["last_modified"] = raw["last_modified"]

        if not known_hash:
            result["status"] = "new"
            result["note"] = "New source, no previous hash"
        else:
            result["status"] = "staged"
            result["note"] = f"Content changed (hash: {known_hash[:12]}... → {new_hash[:12]}...)"

        # Save metadata alongside
        meta_file = staging_dir / f"{sid}_{ts}.meta.json"
        meta_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    except requests.HTTPError as e:
        code = e.response.status_code if e.response else "?"
        result["error"] = f"HTTP {code}"
    except requests.Timeout:
        result["error"] = f"timeout after {REQUEST_TIMEOUT}s"
    except requests.ConnectionError:
        result["error"] = "connection failed"
    except Exception as e:
        result["error"] = str(e)[:200]

    time.sleep(DELAY)
    return result


def crawl_sources(
    sources_csv: Path | None = None,
    staging_dir: Path | None = None,
    limit: int = 0,
) -> dict[str, int]:
    """Crawl all sources in sources.csv that have URLs.

    Returns stats dict with counts for: total, unchanged, staged, new, error, skipped.
    """
    if sources_csv is None:
        sources_csv = ROOT / "data" / "sources.csv"
    if staging_dir is None:
        staging_dir = ROOT / "data" / "staging"

    if not sources_csv.exists():
        return {"error": -1, "msg": f"{sources_csv} not found"}

    with open(sources_csv, encoding="utf-8-sig") as f:
        sources = list(csv.DictReader(f))

    known_hashes = _load_source_hashes()
    stats = {"total": 0, "unchanged": 0, "staged": 0, "new": 0, "error": 0, "skipped": 0}

    for row in sources:
        if limit and stats["total"] >= limit:
            break

        sid = row.get("source_id", "").strip()
        url = row.get("url", "").strip()
        need_login = row.get("need_login", "").strip().lower()

        if not sid or not url or need_login == "yes":
            stats["skipped"] += 1
            continue

        stats["total"] += 1
        print(f"[{stats['total']}] {sid}: {url[:80]}...")

        result = fetch_and_stage(
            url=url,
            source_id=sid,
            staging_dir=staging_dir,
            known_hash=known_hashes.get(sid, ""),
        )

        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        print(f"  → {status}: {result.get('note', result.get('error', ''))}")

    return stats
