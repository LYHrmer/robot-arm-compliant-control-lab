"""Reject incomplete or skipped runs of the required CPU learning test modules."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def check_report(path: Path) -> int:
    required = {
        line.removesuffix(".py").replace("/", ".")
        for line in (ROOT / "environment/learning-tests.txt").read_text().splitlines()
        if line.strip()
    }
    report = ET.parse(path).getroot()
    cases = list(report.iter("testcase"))
    bad = [case.attrib.get("name", "<collection>") for case in cases
           if any(case.find(tag) is not None for tag in ("skipped", "failure", "error"))]
    missing = required - {case.attrib.get("classname", "") for case in cases}
    if not cases or bad or missing:
        raise ValueError(f"incomplete learning tests: missing={sorted(missing)}, unsuccessful={bad}")
    return len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    count = check_report(args.report)
    print(f"CPU learning gate: {count} tests, every required module present, zero skips/failures")


if __name__ == "__main__":
    main()
