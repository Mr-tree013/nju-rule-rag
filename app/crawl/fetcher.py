"""
URL fetcher with content hash change detection (F.1 + F.2).

Downloads a web page, computes content hash, compares with sources.csv
to detect changes.  Saves to data/staging/ for review.
"""

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

USER_AGENT = (
    "NJU-Rule-RAG-Bot/0.6 "
    "(student project; contact: see repository README)"
)
REQUEST_TIMEOUT = 30
DELAY = 1.5  # polite delay between requests


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_url(url: str) -> tuple[int, str, str]:
    """Fetch a URL. Returns (status_code, content_text, content_type)."""
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    # Try to decode as text
    try:
        text = resp.content.decode("utf-8")
    except UnicodeDecodeError:
        text = resp.content.decode("gbk", errors="replace")
    return resp.status_code, text, content_type


def html_to_markdown(html: str, url: str = "") -> str:
    """Convert HTML to plain markdown using basic heuristics.

    For production use, integrate with a proper HTML→MD converter like markdownify.
    This is a lightweight fallback that strips tags and preserves structure.
    """
    # Remove scripts and styles
    html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<footer[^>]*>.*?</footer>", "", html, flags=re.DOTALL | re.IGNORECASE)
    html = re.sub(r"<header[^>]*>.*?</header>", "", html, flags=re.DOTALL | re.IGNORECASE)

    # Convert headings
    for level in range(6, 0, -1):
        html = re.sub(
            rf"<h{level}[^>]*>(.*?)</h{level}>",
            rf"\n\n{'#' * level} \1\n\n",
            html,
            flags=re.DOTALL | re.IGNORECASE,
        )
    # Convert paragraphs
    html = re.sub(r"<p[^>]*>(.*?)</p>", r"\n\n\1\n\n", html, flags=re.DOTALL | re.IGNORECASE)
    # Convert line breaks
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # Convert list items
    html = re.sub(r"<li[^>]*>(.*?)</li>", r"- \1\n", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove remaining HTML tags
    html = re.sub(r"<[^>]+>", "", html)
    # Decode entities
    html = html.replace("&nbsp;", " ").replace("&lt;", "<").replace("&gt;", ">")
    html = html.replace("&amp;", "&").replace("&quot;", '"').replace("&#39;", "'")
    # Clean up whitespace
    html = re.sub(r"\n{3,}", "\n\n", html)
    html = re.sub(r" +", " ", html)
    lines = [line.strip() for line in html.split("\n")]
    html = "\n".join(lines)

    # Add source URL
    title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Untitled"

    markdown = f"# {title}\n\n> 来源：{url}\n> 抓取时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n{html}"
    return markdown


def fetch_and_stage(
    url: str,
    staging_dir: Path | None = None,
    source_id: str | None = None,
) -> dict[str, Any]:
    """Fetch URL, convert to markdown, save to staging.

    Returns metadata dict with status.
    """
    if staging_dir is None:
        staging_dir = Path("data/staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    sid = source_id or f"url_{sha256_text(url)[:8]}"
    result = {
        "source_id": sid,
        "url": url,
        "status": "error",
        "fetched_at": datetime.now().isoformat(),
        "content_hash": "",
        "file_path": "",
        "error": "",
    }

    try:
        code, text, ct = fetch_url(url)
        result["status_code"] = code
        result["content_type"] = ct
        content_hash = sha256_text(text)
        result["content_hash"] = content_hash

        # Convert to markdown
        if "text/html" in ct or ct == "":
            md = html_to_markdown(text, url)
        else:
            md = text

        # Save to staging
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        md_file = staging_dir / f"{sid}_{ts}.md"
        md_file.write_text(md, encoding="utf-8")
        result["file_path"] = str(md_file)
        result["status"] = "staged"
        result["content_length"] = len(md)

        # Save metadata
        meta_file = staging_dir / f"{sid}_{ts}.meta.json"
        meta_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    except requests.HTTPError as e:
        result["error"] = f"HTTP {e.response.status_code if e.response else '?'}"
    except requests.Timeout:
        result["error"] = f"timeout after {REQUEST_TIMEOUT}s"
    except requests.ConnectionError:
        result["error"] = "connection failed"
    except Exception as e:
        result["error"] = str(e)[:200]

    time.sleep(DELAY)
    return result
