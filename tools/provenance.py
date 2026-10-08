"""Replay-relevant source identity, computed from the real import graph.

The historical rule hashes every file in the package (see
``compliant_control_lab.surface_experiment._source_hashes``), so editing any
module invalidates every archive even when that module cannot affect the replay.
This module walks the static import graph from a study entry point instead and
reports only the package modules that entry point actually reaches.

Nothing here imports or executes the analysed modules: the graph comes from AST
parsing, so the result does not depend on what the interpreter happened to load
first. That matters under pytest, where unrelated test modules have already
populated ``sys.modules``.

The velocity-cost, onset-observer and velocity-time studies use this rule.
Other studies retain their existing identity contracts; see the design document.
"""

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "compliant_control_lab"
PACKAGE_DIR = ROOT / "src" / PACKAGE

# Import roots inside the repository: a dotted prefix and the directory holding it.
# "tools" is walked so that a study reaches package modules through its helpers,
# but tools files never enter the reported identity; they stay explicitly listed.
SEARCH_ROOTS = ((PACKAGE, ROOT / "src"), ("tools", ROOT))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_file(name: str) -> Path | None:
    """Resolve a dotted module name to a repository file, or None if external."""
    for prefix, base in SEARCH_ROOTS:
        if name != prefix and not name.startswith(f"{prefix}."):
            continue
        relative = Path(*name.split("."))
        for candidate in (base / relative.with_suffix(".py"), base / relative / "__init__.py"):
            if candidate.is_file():
                return candidate.resolve()
    return None


def _referenced_modules(path: Path) -> set[str]:
    """Dotted names that `path` imports from inside the repository.

    ``from a.b import c`` is ambiguous: ``c`` may be a submodule or a symbol, so
    both ``a.b`` and ``a.b.c`` are offered and `_module_file` discards whichever
    does not exist.

    Relative imports raise instead of being skipped: the repository has none today,
    and silently ignoring one would drop a real dependency from the closure, which
    is exactly the false pass this module exists to avoid.
    """
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError(
                    f"{path}:{node.lineno} uses a relative import, which the closure "
                    "resolver does not follow; make it absolute"
                )
            if node.module:
                names.add(node.module)
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def closure_files(entry: Path | str) -> set[Path]:
    """Repository files reachable from `entry` through the static import graph."""
    start = Path(entry).resolve()
    if not start.is_file():
        raise FileNotFoundError(f"entry point is not a file: {entry}")
    # Importing any package module first executes the package __init__, which the
    # entry file need not mention; seed it so the closure matches real execution.
    seen, queue = {start}, [start]
    if (init := PACKAGE_DIR / "__init__.py").is_file():
        seen.add(init.resolve())
        queue.append(init.resolve())
    while queue:
        for name in _referenced_modules(queue.pop()):
            # Python executes each ancestor package's __init__, including tools
            # subpackages. They may import dependencies absent from the leaf.
            parts = name.split(".")
            for length in range(1, len(parts) + 1):
                resolved = _module_file(".".join(parts[:length]))
                if resolved is not None and resolved not in seen:
                    seen.add(resolved)
                    queue.append(resolved)
    return seen


def closure_identity(entry: Path | str) -> dict[str, str]:
    """Hash the package modules `entry` reaches, keyed by repository-relative path.

    Assets stay out: all 82 of them live under one Franka subdirectory that every
    Franka experiment needs in full, so narrowing them would require parsing the
    MuJoCo include and mesh graph for no practical gain.
    """
    reached = sorted(
        path for path in closure_files(entry)
        if path.is_relative_to(PACKAGE_DIR) and path.suffix == ".py"
    )
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in reached}


def _is_source_record(name: object, digest: object) -> bool:
    """Whether an entry looks like a path-to-SHA256 record."""
    return (
        isinstance(name, str) and isinstance(digest, str)
        and len(digest) == 64 and set(digest) <= set("0123456789abcdef")
    )


def verify_source_identity(
    saved: dict[str, str], live: dict[str, str], *,
    baseline: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Check a recorded identity against the current one without demanding equality.

    `live` is a study's full identity: the closure of package modules plus whatever
    it pins explicitly (assets, tools files, lock files). Every entry in `live` must
    appear in `saved` with the same digest. `saved` may additionally carry Python
    files under the package directory -- an archive written under the whole-package
    rule passes unchanged, with no archived bytes rewritten -- but any extra entry
    outside it is rejected as injected.

    `baseline` carries the one-time migration entries from
    ``tools.provenance_migration``: explicit (archived, migrated) digest pairs.
    Both ends must match; a later source edit invalidates the exception. Numerical
    reconstruction is a separate check and cannot establish source equivalence.
    """
    if not isinstance(saved, dict):
        raise ValueError("archived source identity must be an object")  # noqa: TRY004 - audit error
    if malformed := sorted(repr(name) for name, digest in saved.items()
                           if not _is_source_record(name, digest)):
        raise ValueError(f"archived source identity has malformed records: {malformed}")
    # The whole-package rule only ever pinned files under the package directory, so
    # that is the only kind of extra entry a sealed archive may carry. Anything else
    # is an injected record, which an ignored-extras rule would wave through.
    if unexpected := sorted(
        name for name in set(saved) - set(live)
        if not (name.startswith(f"src/{PACKAGE}/") and name.endswith(".py")
                and PurePosixPath(name).as_posix() == name
                and ".." not in PurePosixPath(name).parts)
    ):
        raise ValueError(f"archived source identity has unexpected records: {unexpected}")
    if missing := sorted(name for name in live if name not in saved):
        raise ValueError(f"archive does not record replay-relevant source: {missing}")
    allowed = baseline or {}
    mismatched = [name for name in live if saved[name] != live[name]]
    migrated = sorted(name for name in mismatched
                      if (saved[name], live[name]) == allowed.get(name))
    if differing := sorted(name for name in mismatched if name not in set(migrated)):
        raise ValueError(f"replay-relevant source differs: {differing}")
    if migrated:
        return f"migrated: {', '.join(migrated)}"
    return "exact" if saved == live else "closure-subset"


def verify_frozen_source(
    archive: Path | str, live: dict[str, str], *,
    baseline: dict[str, tuple[str, str]] | None = None, name: str = "source_hashes.json",
) -> str:
    """Read and check an archive's frozen source identity.

    Keeps the studies' established `frozen input differs` wording while adding the
    specific reason, so an audit failure still reads the same but says more.
    """
    saved = json.loads((Path(archive) / name).read_text())
    try:
        return verify_source_identity(saved, live, baseline=baseline)
    except ValueError as error:
        raise ValueError(f"frozen input differs: {name} ({error})") from error


def verify_inherited_identity(
    parent: dict[str, str], live: dict[str, str], *,
    baseline: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Check the files a parent archive pinned that this study still depends on.

    A child study reuses a parent archive's traces, so the parent's record of the
    shared sources must still hold. Only the overlap is meaningful: the parent
    pinned the whole package, including modules this study never imports, and it
    predates this study's own files so it cannot have recorded them.

    `baseline` crosses the migration the same way as in `verify_source_identity`.
    """
    shared = {name: digest for name, digest in live.items() if name in parent}
    if not shared:
        raise ValueError("parent archive and current identity share no source files")
    return verify_source_identity(parent, shared, baseline=baseline)

