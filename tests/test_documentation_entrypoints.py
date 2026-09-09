"""Safety and local-anchor checks for the three introductory documentation pages."""

from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENTRY_PAGES = (
    REPOSITORY_ROOT / "README.md",
    REPOSITORY_ROOT / "docs/recruiter_walkthrough.md",
    REPOSITORY_ROOT / "docs/tutorial/README.md",
)
MARKDOWN_PATHS = (
    REPOSITORY_ROOT / "README.md",
    *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
)

INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
REFERENCE_LINK = re.compile(
    r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)", re.MULTILINE
)
ATX_HEADING = re.compile(r"^ {0,3}#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE = re.compile(
    r"^ {0,3}(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)"
    r"^ {0,3}(?P=fence)[ \t]*$",
    re.MULTILINE | re.DOTALL,
)
GENERATOR_COMMAND = re.compile(
    r"^\s*(?:[A-Z_][A-Z0-9_]*=\S+\s+)*"
    r"(?:franka-control-lab|compliant-control-lab)\b"
    r"(?P<arguments>(?:[^\n\\]|\\\n)*)",
    re.MULTILINE,
)
OUTPUT_ARGUMENT = re.compile(
    r"(?<!\S)--output(?:=|\s+)"
    r"(?:\"(?P<double>[^\"]+)\"|'(?P<single>[^']+)'|(?P<plain>[^\s\\]+))"
)


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _raw_link_targets(text: str):
    for pattern in (INLINE_LINK, REFERENCE_LINK):
        for match in pattern.finditer(text):
            yield match.start(), match.group(1)


def _link_destination(source: Path, raw_target: str) -> tuple[Path, str] | None:
    """Resolve a Markdown target locally, decoding its path and fragment."""
    target = raw_target.strip()
    if target.startswith("<") and ">" in target:
        target = target[1 : target.index(">")]
    else:
        target = target.split(maxsplit=1)[0]

    parsed = urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return None

    decoded_path = unquote(parsed.path)
    destination = source if not decoded_path else (source.parent / decoded_path).resolve()
    if destination.is_dir():
        destination /= "README.md"
    return destination, unquote(parsed.fragment)


def _heading_text(markdown: str):
    """Yield ATX heading text while ignoring examples inside fenced code blocks."""
    in_fence: str | None = None
    for line in markdown.splitlines():
        fence_match = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
        if fence_match:
            marker = fence_match.group(1)
            if in_fence is None:
                in_fence = marker[0]
            elif marker[0] == in_fence:
                in_fence = None
            continue
        if in_fence is not None:
            continue
        match = ATX_HEADING.match(line)
        if match:
            yield match.group(1)


def _base_github_slug(heading: str) -> str:
    visible = html.unescape(heading)
    visible = re.sub(r"<[^>]*>", "", visible)
    visible = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", visible)
    visible = re.sub(r"\[([^\]]*)\]\[[^\]]*\]", r"\1", visible)
    visible = visible.strip().lower()
    visible = "".join(
        character
        for character in visible
        if character in "-_ " or not unicodedata.category(character).startswith(("P", "S"))
    )
    return re.sub(r"\s", "-", visible)


def _github_heading_anchors(markdown: str) -> set[str]:
    """Return GitHub-style slugs, including collision-safe duplicate suffixes."""
    anchors: set[str] = set()
    next_suffix: dict[str, int] = {}
    for heading in _heading_text(markdown):
        base = _base_github_slug(heading)
        candidate = base
        suffix = next_suffix.get(base, 1)
        while candidate in anchors:
            candidate = f"{base}-{suffix}"
            suffix += 1
        next_suffix[base] = suffix
        anchors.add(candidate)
    return anchors


def _is_scratch_output(raw_path: str) -> bool:
    path = PurePosixPath(raw_path)
    return (
        path.is_absolute()
        and len(path.parts) >= 3
        and path.parts[:2] == ("/", "tmp")
        and any("compliant-control" in part for part in path.parts[2:])
        and ".." not in path.parts
    )


def test_github_heading_anchors_handle_punctuation_and_duplicates() -> None:
    markdown = """# Setup!
## Setup!
## Setup-1
## Setup!
## 中文：示例
```
# Not a heading
```
"""

    assert _github_heading_anchors(markdown) == {
        "setup",
        "setup-1",
        "setup-1-1",
        "setup-2",
        "中文示例",
    }


def test_link_destination_decodes_local_targets_and_ignores_external(tmp_path: Path) -> None:
    source = tmp_path / "index.md"

    assert _link_destination(source, "guide%20name.md#caf%C3%A9") == (
        tmp_path / "guide name.md",
        "café",
    )
    assert _link_destination(source, "#same%20page") == (source, "same page")
    assert _link_destination(source, "https://example.com/guide.md#section") is None
    assert _link_destination(source, "mailto:maintainer@example.com") is None


def test_entry_page_local_heading_anchors_resolve() -> None:
    entry_pages = {path.resolve() for path in ENTRY_PAGES}
    anchor_cache: dict[Path, set[str]] = {}
    broken: list[str] = []

    for source in MARKDOWN_PATHS:
        text = source.read_text(encoding="utf-8")
        for offset, raw_target in _raw_link_targets(text):
            resolved = _link_destination(source, raw_target)
            if resolved is None:
                continue
            destination, fragment = resolved
            if not fragment or (source.resolve() not in entry_pages and destination not in entry_pages):
                continue
            if not destination.is_file() or destination.suffix.lower() != ".md":
                continue  # Path existence and repository containment are checked separately.
            anchors = anchor_cache.setdefault(
                destination,
                _github_heading_anchors(destination.read_text(encoding="utf-8")),
            )
            if fragment not in anchors:
                broken.append(
                    f"{source.relative_to(REPOSITORY_ROOT)}:{_line_number(text, offset)} "
                    f"-> {raw_target} (missing heading anchor #{fragment})"
                )

    assert not broken, "broken entry-page anchor(s):\n" + "\n".join(broken)


def test_entry_page_generators_write_only_to_scratch_paths() -> None:
    violations: list[str] = []
    for source in ENTRY_PAGES:
        text = source.read_text(encoding="utf-8")
        for block in FENCE.finditer(text):
            body = block.group("body")
            for output in OUTPUT_ARGUMENT.finditer(body):
                output_path = next(value for value in output.groups() if value is not None)
                if not _is_scratch_output(output_path):
                    line = _line_number(text, block.start("body") + output.start())
                    violations.append(
                        f"{source.relative_to(REPOSITORY_ROOT)}:{line} "
                        f"uses non-scratch --output {output_path}"
                    )
            for command in GENERATOR_COMMAND.finditer(body):
                output = OUTPUT_ARGUMENT.search(command.group("arguments"))
                line = _line_number(text, block.start("body") + command.start())
                if output is None:
                    violations.append(f"{source.relative_to(REPOSITORY_ROOT)}:{line} lacks --output")

    assert not violations, "unsafe introductory generator command(s):\n" + "\n".join(violations)
