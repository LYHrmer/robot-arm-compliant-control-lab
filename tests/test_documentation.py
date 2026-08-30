from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

REPOSITORY_ROOT = Path(__file__).parents[1]
MARKDOWN_PATHS = (
    REPOSITORY_ROOT / "README.md",
    *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
)
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK = re.compile(r"^\[[^\]]+\]:\s*(\S+)", re.MULTILINE)
CJK_BEFORE_INLINE_MATH = re.compile(r"[\u3000-\u9fff]\$(?!\$)")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        return target[1 : target.index(">")]
    return target.split(maxsplit=1)[0]


def test_markdown_has_no_c0_control_characters() -> None:
    violations: list[str] = []
    for path in MARKDOWN_PATHS:
        text = path.read_text(encoding="utf-8")
        for offset, character in enumerate(text):
            if (ord(character) < 32 and character not in "\t\n\r") or ord(character) == 127:
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, offset)} "
                    f"contains U+{ord(character):04X}"
                )

    assert not violations, "\n".join(violations)


def test_markdown_uses_github_math_delimiters() -> None:
    violations: list[str] = []
    for path in MARKDOWN_PATHS:
        text = path.read_text(encoding="utf-8")
        for delimiter in (r"\(", r"\)", r"\[", r"\]"):
            for match in re.finditer(re.escape(delimiter), text):
                violations.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, match.start())} "
                    f"uses {delimiter}"
                )

    assert not violations, "\n".join(violations)


def test_inline_math_is_separated_from_chinese_text() -> None:
    violations: list[str] = []
    for path in MARKDOWN_PATHS:
        text = path.read_text(encoding="utf-8")
        for match in CJK_BEFORE_INLINE_MATH.finditer(text):
            violations.append(
                f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, match.start())} "
                "needs a space before inline math"
            )

    assert not violations, "\n".join(violations)


def test_relative_markdown_links_exist() -> None:
    missing: list[str] = []
    for path in MARKDOWN_PATHS:
        text = path.read_text(encoding="utf-8")
        matches = (*INLINE_LINK.finditer(text), *REFERENCE_LINK.finditer(text))
        for match in matches:
            target = _link_target(match.group(1))
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            relative_target = unquote(target.split("#", 1)[0].split("?", 1)[0])
            if relative_target and not (path.parent / relative_target).exists():
                missing.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, match.start())} "
                    f"links to missing {relative_target}"
                )

    assert not missing, "\n".join(missing)
