"""Offline check for local Markdown links under README.md and docs/.

This module never opens a socket. External links (http/https/mailto) and
pure same-page anchors (`#section`) are recognized by scheme/prefix and
skipped without being dereferenced; every other link target is treated as a
local path and must resolve to a file or directory that exists on disk,
inside the repository.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_PATHS = (
    REPOSITORY_ROOT / "README.md",
    *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
)

# Inline links/images: [text](target) or ![alt](target).
INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
# Reference-style definitions: [label]: target
REFERENCE_LINK = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(\S+)", re.MULTILINE)

EXTERNAL_SCHEMES = ("http://", "https://", "mailto:")


def _iter_markdown_files():
    assert MARKDOWN_PATHS, "expected README.md and docs/ to be discoverable"
    for path in MARKDOWN_PATHS:
        assert path.is_file(), f"expected {path} to exist"
        yield path


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _raw_targets(text: str):
    for match in INLINE_LINK.finditer(text):
        yield match.start(), match.group(1)
    for match in REFERENCE_LINK.finditer(text):
        yield match.start(), match.group(1)


def _local_target(raw_target: str) -> str | None:
    """Return the local path portion of a link target, or None if external."""
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()
    else:
        # Inline links may carry a trailing "title" after a space.
        target = target.split(maxsplit=1)[0]
    if not target or target.startswith(EXTERNAL_SCHEMES):
        return None
    if target.startswith("#"):
        return None
    without_fragment = target.split("#", 1)[0]
    without_query = without_fragment.split("?", 1)[0]
    return unquote(without_query)


def test_documentation_tree_is_present() -> None:
    files = list(_iter_markdown_files())
    assert files, "expected at least README.md to be checked"


def test_local_markdown_links_resolve_to_existing_paths() -> None:
    missing: list[str] = []
    for path in _iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for offset, raw_target in _raw_targets(text):
            local_target = _local_target(raw_target)
            if not local_target:
                continue
            resolved = (path.parent / local_target).resolve()
            if not resolved.exists():
                missing.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, offset)} "
                    f"-> {local_target}"
                )

    assert not missing, "broken local link(s):\n" + "\n".join(missing)


def test_local_links_do_not_escape_the_repository() -> None:
    escaping: list[str] = []
    for path in _iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for offset, raw_target in _raw_targets(text):
            local_target = _local_target(raw_target)
            if not local_target:
                continue
            resolved = (path.parent / local_target).resolve()
            if REPOSITORY_ROOT not in (resolved, *resolved.parents):
                escaping.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, offset)} "
                    f"-> {local_target} resolves outside the repository"
                )

    assert not escaping, "link(s) escaping the repository:\n" + "\n".join(escaping)


def test_external_links_use_a_recognized_scheme() -> None:
    """Guard against a typo'd scheme being silently treated as a local path.

    Anything that looks like a URI scheme (``word:`` prefix) but is not one
    of the recognized external schemes would otherwise fall through to the
    local-path check below and either be mistakenly validated as a file or
    silently ignored. This test never opens a connection; it only inspects
    the literal text of the link.
    """
    uri_like = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")
    unrecognized: list[str] = []
    for path in _iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        for offset, raw_target in _raw_targets(text):
            target = raw_target.strip()
            if target.startswith("<") and target.endswith(">"):
                target = target[1:-1].strip()
            else:
                target = target.split(maxsplit=1)[0]
            if uri_like.match(target) and not target.startswith(EXTERNAL_SCHEMES):
                unrecognized.append(
                    f"{path.relative_to(REPOSITORY_ROOT)}:{_line_number(text, offset)} -> {target}"
                )

    assert not unrecognized, "unrecognized URI scheme(s):\n" + "\n".join(unrecognized)
