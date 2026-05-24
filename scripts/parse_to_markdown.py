"""
Convert raw documents (HTML / PDF / DOC / DOCX / TXT) to clean Markdown.

Designed as a reusable module: import and call convert_file(), or run
directly as a CLI script.

Usage as library:
    from scripts.parse_to_markdown import convert_file
    md_text = convert_file(Path("some_file.pdf"))
    md_text = convert_file(Path("page.html"), url="https://jw.nju.edu.cn/...")

Usage as CLI:
    python scripts/parse_to_markdown.py page.html --url https://jw.nju.edu.cn/...
    python scripts/parse_to_markdown.py file.pdf --title "通知标题"
"""

import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ── format detection ──────────────────────────────────────────────

HTML_EXTENSIONS = {".html", ".htm"}
PDF_EXTENSIONS = {".pdf"}
DOC_EXTENSIONS = {".doc"}
DOCX_EXTENSIONS = {".docx"}
TEXT_EXTENSIONS = {".txt", ".md", ".markdown"}


def detect_format(file_path: Path) -> str:
    """Return format label: html / pdf / doc / docx / txt / unknown."""
    suffix = file_path.suffix.lower()
    if suffix in HTML_EXTENSIONS:
        return "html"
    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in DOC_EXTENSIONS:
        return "doc"
    if suffix in DOCX_EXTENSIONS:
        return "docx"
    if suffix in TEXT_EXTENSIONS:
        return "txt"
    # Fallback: try MIME sniffing
    try:
        with open(file_path, "rb") as f:
            head = f.read(4)
        if head.startswith(b"%PDF"):
            return "pdf"
        if head.startswith(b"<!DO") or head.startswith(b"<htm") or head.startswith(b"<HTM"):
            return "html"
        if head[:2] == b"PK":
            return "docx"
    except OSError:
        pass
    return "unknown"


# ── HTML ──────────────────────────────────────────────────────────

def _parse_html(file_path: Path) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise RuntimeError("beautifulsoup4 is required for HTML parsing")

    with open(file_path, encoding="utf-8", errors="replace") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "noscript", "iframe", "link", "meta"]):
        tag.decompose()

    # Try to find the main content area
    main = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", class_=re.compile(r"content|article|main|post", re.I))
        or soup.find("div", id=re.compile(r"content|article|main|post", re.I))
    )
    target = main if main else soup

    text = target.get_text()

    # Clean up whitespace
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)

    return _clean_whitespace("\n\n".join(lines))


# ── PDF ───────────────────────────────────────────────────────────

def _parse_pdf(file_path: Path) -> str:
    try:
        import fitz
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required for PDF parsing")

    doc = fitz.open(str(file_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return _clean_whitespace("\n\n".join(pages))


# ── DOC / DOCX ────────────────────────────────────────────────────

def _parse_doc(file_path: Path) -> str:
    """Convert .doc/.docx to text via LibreOffice headless."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        result = subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "txt:Text",
                "--outdir", str(tmp),
                str(file_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        txt_files = list(tmp.glob("*.txt"))
        if not txt_files:
            raise RuntimeError(
                f"LibreOffice produced no .txt file. stderr: {result.stderr[:200]}"
            )
        text = txt_files[0].read_text(encoding="utf-8", errors="replace")
        return _clean_whitespace(text)


def _parse_docx(file_path: Path) -> str:
    """Extract text from .docx using python-docx."""
    try:
        from docx import Document
    except ImportError:
        raise RuntimeError("python-docx is required for .docx parsing")

    doc = Document(str(file_path))
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


# ── TXT ───────────────────────────────────────────────────────────

def _parse_txt(file_path: Path) -> str:
    text = file_path.read_text(encoding="utf-8", errors="replace")
    return _clean_whitespace(text)


# ── cleaning ──────────────────────────────────────────────────────

NOISE_LINES = {
    "首页", "下一页", "上一页", "返回", "返回目录", "返回首页",
    "导航", "页脚", "版权声明",
}


def _clean_whitespace(text: str) -> str:
    """Normalize whitespace and drop obvious noise lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    lines = text.split("\n")
    kept = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in NOISE_LINES:
            continue
        if re.match(r"^\d{1,4}$", stripped):  # lone page number
            continue
        kept.append(stripped)

    return "\n\n".join(kept)


# ── main API ──────────────────────────────────────────────────────

def convert_file(
    file_path: Path,
    *,
    title: str = "",
    url: str = "",
    source_id: str = "",
) -> str:
    """
    Convert any supported file to clean Markdown text.

    Returns the markdown body.  If *title* is provided a heading block
    is prepended.

    Raises RuntimeError on unsupported format or parse failure.
    """
    fmt = detect_format(file_path)

    parsers = {
        "html": _parse_html,
        "pdf": _parse_pdf,
        "doc": _parse_doc,
        "docx": _parse_docx,
        "txt": _parse_txt,
    }

    parser = parsers.get(fmt)
    if parser is None:
        raise RuntimeError(f"Unsupported format: {fmt} ({file_path.suffix})")

    try:
        body = parser(file_path)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"LibreOffice conversion timed out: {file_path}")
    except Exception as exc:
        raise RuntimeError(f"Parse failed ({fmt}): {exc}") from exc

    if not body.strip():
        raise RuntimeError(f"Parse produced empty output: {file_path}")

    # Build header block
    if title or source_id or url:
        header_lines = []
        if title:
            header_lines.append(f"# {title}\n")
        meta_parts = []
        if source_id:
            meta_parts.append(f"source_id: {source_id}")
        if url:
            meta_parts.append(f"url: {url}")
        if meta_parts:
            header_lines.append("> " + "  |  ".join(meta_parts) + "\n")
        header = "\n".join(header_lines)
        return header + "\n" + body

    return body


# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert a document to Markdown"
    )
    parser.add_argument("file", type=Path, help="Path to input file")
    parser.add_argument("--title", default="", help="Document title")
    parser.add_argument("--url", default="", help="Source URL")
    parser.add_argument("--source-id", default="", help="Source ID")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output file (default: stdout)")
    args = parser.parse_args()

    if not args.file.exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    try:
        md = convert_file(
            args.file,
            title=args.title,
            url=args.url,
            source_id=args.source_id,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.write_text(md, encoding="utf-8")
        print(f"Written: {args.output} ({len(md)} chars)")
    else:
        print(md)
