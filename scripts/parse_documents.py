"""
Parse raw HTML / PDF / DOC files into clean Markdown.

Reads data/sources.csv for source metadata, finds source files in a
configurable input directory, extracts text, and writes:
  data/processed/{output_filename}.md
  data/processed/{source_id}.parse_metadata.json
"""

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
SOURCES_CSV = ROOT / "data" / "sources.csv"
PROCESSED_DIR = ROOT / "data" / "processed"

# Where to find raw files — override via env or edit here.
RAW_INPUT_DIR = Path(
    os.getenv("RAW_INPUT_DIR", "/mnt/c/Users/Mr.tree/Desktop/下载")
)

# Mapping source_id → input filename in RAW_INPUT_DIR.
# Only sources listed here will be parsed.  Add more as needed.
SOURCE_FILE_MAP = {
    "nju-jw-027": "南京大学本科毕业论文（设计）撰写规范、存档实施细则（试行）.doc",
    "nju-jw-028": "南京大学本科教务系统缓考申请学生使用手册.pdf",
    "nju-jw-029": "南京大学本科教务系统补考办理学生使用手册.pdf",
    "nju-jw-030": "南京大学本科教务系统学分认定学生使用手册（新）.pdf",
    "nju-jw-031": "学生注册与选课指南（中国大学MOOC平台SPOC专用）.pdf",
}

# Output filenames for data/processed/
OUTPUT_NAMES = {
    "nju-jw-027": "南京大学-本科毕业论文撰写规范与存档实施细则.md",
    "nju-jw-028": "南京大学-教务系统缓考申请学生使用手册.md",
    "nju-jw-029": "南京大学-教务系统补考办理学生使用手册.md",
    "nju-jw-030": "南京大学-教务系统学分认定学生使用手册.md",
    "nju-jw-031": "南京大学-MOOC平台SPOC选课指南.md",
}


# ── helpers ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and remove obvious noise."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove lines that are just page numbers
    text = re.sub(r"\n\s*\d{1,4}\s*\n", "\n", text)
    return text.strip()


def parse_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text.strip())
    doc.close()
    return clean_text("\n\n".join(pages))


def parse_doc(doc_path: Path) -> str:
    """Convert .doc to text via LibreOffice CLI, then clean."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        # LibreOffice headless convert → text
        subprocess.run(
            [
                "libreoffice",
                "--headless",
                "--convert-to", "txt:Text",
                "--outdir", str(tmp),
                str(doc_path),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        # LibreOffice outputs {stem}.txt
        txt_files = list(tmp.glob("*.txt"))
        if not txt_files:
            return ""
        text = txt_files[0].read_text(encoding="utf-8", errors="replace")
        return clean_text(text)


def build_header(source_row: dict) -> str:
    """Build a YAML-style header block for the markdown file."""
    return (
        f"# {source_row['title']}\n\n"
        f"> source_id: {source_row['source_id']}\n"
        f"> department: {source_row['department']}\n"
        f"> scope: {source_row['scope']}\n"
        f"> priority: {source_row['priority']}\n\n"
    )


# ── main ─────────────────────────────────────────────────────────

def main():
    if not RAW_INPUT_DIR.exists():
        print(f"Error: RAW_INPUT_DIR not found: {RAW_INPUT_DIR}", file=sys.stderr)
        sys.exit(1)

    with open(SOURCES_CSV, encoding="utf-8") as f:
        sources = {r["source_id"]: r for r in csv.DictReader(f)}

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    skipped = 0
    failed = 0

    for source_id, input_name in SOURCE_FILE_MAP.items():
        source_row = sources.get(source_id)
        if source_row is None:
            print(f"[SKIP] {source_id}: not in sources.csv")
            skipped += 1
            continue

        input_path = RAW_INPUT_DIR / input_name
        if not input_path.exists():
            print(f"[SKIP] {source_id}: file not found: {input_path}")
            skipped += 1
            continue

        output_name = OUTPUT_NAMES.get(source_id, f"{source_id}.md")
        output_path = PROCESSED_DIR / output_name
        meta_path = PROCESSED_DIR / f"{source_id}.parse_metadata.json"

        print(f"[PARSE] {source_id}: {input_name}")

        try:
            # Detect format and parse
            suffix = input_path.suffix.lower()
            if suffix == ".pdf":
                body = parse_pdf(input_path)
                parser = "pymupdf"
            elif suffix in (".doc", ".docx"):
                body = parse_doc(input_path)
                parser = "libreoffice"
            else:
                print(f"  [SKIP] unknown format: {suffix}")
                skipped += 1
                continue

            if not body or len(body.strip()) < 50:
                print(f"  [WARN] extracted text too short ({len(body)} chars), "
                      "marking need_manual_check=true")
                warning = "extracted_text_too_short"

            # Build full markdown
            header = build_header(source_row)
            full_md = header + body

            output_path.write_text(full_md, encoding="utf-8")
            print(f"  [OK] → {output_path} ({len(body)} chars)")

            # Write parse metadata
            metadata = {
                "source_id": source_id,
                "input_file": str(input_path),
                "output_file": str(output_path),
                "parser": parser,
                "text_length": len(body),
                "parsed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "warning": warning if body and len(body.strip()) < 50 else None,
            }
            meta_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Update sources.csv with the new filename
            source_row["filename"] = output_name

            success += 1

        except subprocess.TimeoutExpired:
            print(f"  [FAIL] LibreOffice timed out")
            failed += 1
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failed += 1

    # Write updated sources.csv
    with open(SOURCES_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(sources[next(iter(sources))].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sources.values())

    print()
    print(f"Parse complete.  Success: {success}  Failed: {failed}  Skipped: {skipped}")
    print(f"Sources CSV updated: {SOURCES_CSV}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
