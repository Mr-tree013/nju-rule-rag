"""
Annotate questions.csv with gold_source_ids based on topic and expected keywords.

Maps each question to the source documents that should contain the answer,
using a curated topic-to-source mapping derived from sources.csv metadata.
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_CSV = ROOT / "data" / "eval" / "questions.csv"

# Topic → expected source_id list (curated from sources.csv metadata)
# Sources are chosen by matching document title/topic to question topic.
GOLD_MAP: dict[str, list[str]] = {
    "选课": ["nju-jw-006"],
    "缓考": ["nju-jw-004", "nju-jw-028", "nju-jw-045", "nju-jw-033"],
    "补考": ["nju-jw-005", "nju-jw-029", "nju-jw-032", "nju-jw-033"],
    "重修": ["nju-jw-005"],
    "成绩": ["nju-jw-003", "nju-jw-043"],
    "转专业": ["nju-jw-011"],
    "学籍": ["nju-jw-001"],
    "休学": ["nju-jw-001", "nju-jw-012"],
    "退学": ["nju-jw-001"],
    "学籍异动": ["nju-jw-001"],
    "辅修": ["nju-jw-015", "nju-jw-046"],
    "交换": ["nju-jw-014", "nju-jw-041", "nju-jw-042"],
    "课程认定": ["nju-jw-014", "nju-jw-041", "nju-jw-042"],
    "全球科考": ["nju-jw-018", "nju-jw-019", "nju-jw-034"],
    "学业预警": ["nju-jw-008"],
    "作弊": ["nju-jw-002"],
    "处分": ["nju-jw-002", "nju-jw-001"],
    "毕业": ["nju-jw-007", "nju-jw-010", "nju-jw-039"],
    "学位": ["nju-jw-007"],
    "请假": ["nju-jw-001"],
    "宿舍": ["nju-life-001", "nju-life-002"],
    "校园网": ["nju-life-004"],
    "学生证": ["nju-life-008"],
    "校园卡": ["nju-life-012", "nju-life-014"],
    "军训": ["nju-life-015"],
    "校医院": ["nju-life-003"],
    "医保": ["nju-life-003"],
    "交通": ["nju-life-009", "nju-life-010"],
    "邮箱": ["nju-life-011"],
    "诈骗": ["nju-life-005"],
    "贷款": ["nju-life-013"],
    "报到": ["nju-life-007"],
}


def get_gold_ids(topic: str, expected_keyword: str) -> str:
    """Return semicolon-separated gold source_ids or 'unknown'.

    Matches each /-separated topic component AND the expected keyword
    to build a union of all relevant gold sources.
    """
    ids_set: set[str] = set()

    # Match each component of compound topics like "成绩/绩点" or "处分/退学/学位"
    for part in topic.replace("；", ";").replace("，", ",").split("/"):
        part = part.strip()
        if part in GOLD_MAP:
            ids_set.update(GOLD_MAP[part])

    # Also match keyword (don't skip just because topic matched)
    if expected_keyword:
        kw = expected_keyword.strip()
        if kw in GOLD_MAP:
            ids_set.update(GOLD_MAP[kw])
        else:
            for part in kw.replace("；", ";").replace("，", ",").split(";"):
                part = part.strip()
                if part in GOLD_MAP:
                    ids_set.update(GOLD_MAP[part])

    if ids_set:
        return ";".join(sorted(ids_set))
    return "unknown"


def main() -> int:
    if not QUESTIONS_CSV.exists():
        print(f"Error: {QUESTIONS_CSV} not found.", file=sys.stderr)
        return 1

    with open(QUESTIONS_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if "gold_source_ids" in fieldnames:
        print("gold_source_ids column already exists, updating values...")
    else:
        fieldnames = list(fieldnames) + ["gold_source_ids"]

    updated = 0
    unknown = 0
    for row in rows:
        topic = row.get("topic", "").strip()
        keyword = row.get("expected_source_keyword", "").strip()
        gold = get_gold_ids(topic, keyword)
        row["gold_source_ids"] = gold
        if gold == "unknown":
            unknown += 1
        else:
            updated += 1

    with open(QUESTIONS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Updated {QUESTIONS_CSV}")
    print(f"  {updated} questions annotated with gold_source_ids")
    print(f"  {unknown} questions marked 'unknown'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
