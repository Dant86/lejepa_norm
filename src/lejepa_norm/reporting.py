"""Markdown table writer for experiment notes.

Generated content lives between marker comments so that re-running the analysis
replaces it in place and leaves hand-written text untouched:

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

New sections are inserted at the end of the note's `## Results` section (before
the next `## ` heading), or appended if the note has no such heading.
"""

import collections.abc
import os
import pathlib
import re

Cell = str | int | float | None

RESULTS_HEADING = "## Results"

NOTE_TEMPLATE = """# {title}
- **Branch / version:** <branch> @ <commit hash>
- **Run datetime (UTC):** <YYYY-MM-DDTHH:MM:SSZ>
- **Command(s):**
  `<command>`
- **Datasets:** <datasets with canonical and HF links>
- **Backbone:** <backbone with HF link>

## Question

## Setup (deviations from the base config)

## Results

## Interpretation

## Next steps
"""


def _begin(name: str) -> str:
    return f"<!-- BEGIN GENERATED: {name} -->"


def _end(name: str) -> str:
    return f"<!-- END GENERATED: {name} -->"


def format_cell(value: Cell, float_format: str = "{:.4g}") -> str:
    """Formats one table cell; pipes and newlines are escaped."""
    if value is None:
        text = ""
    elif isinstance(value, float):
        text = float_format.format(value)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def markdown_table(
    rows: collections.abc.Sequence[collections.abc.Mapping[str, Cell]],
    columns: collections.abc.Sequence[str] | None = None,
    float_format: str = "{:.4g}",
) -> str:
    """Renders rows as a GitHub-flavored markdown table.

    Args:
        rows: One mapping per row; missing keys render as empty cells.
        columns: Column order; defaults to the keys in first-seen order.
        float_format: Format applied to float cells.

    Returns:
        The table (no trailing newline), or `_No rows._` if `rows` is empty.
    """
    if not rows:
        return "_No rows._"
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    lines = [
        "| " + " | ".join(format_cell(c) for c in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(format_cell(row.get(c), float_format) for c in columns) + " |"
        for row in rows
    )
    return "\n".join(lines)


def image_link(
    image: str | os.PathLike[str], note: str | os.PathLike[str], alt: str
) -> str:
    """Markdown image embed of `image` with a path relative to `note`'s folder."""
    rel = os.path.relpath(pathlib.Path(image), pathlib.Path(note).parent)
    return f"![{alt}]({pathlib.PurePath(rel).as_posix()})"


def update_section(note: str | os.PathLike[str], name: str, content: str) -> None:
    """Inserts or replaces the generated section `name` in a markdown note.

    Idempotent: calling twice with the same content leaves the file unchanged.
    The note is created from `NOTE_TEMPLATE` if it does not exist.

    Args:
        note: Path of the experiment note.
        name: Section identifier (letters, digits, `_`, `-`, `.`).
        content: Markdown placed between the markers.

    Raises:
        ValueError: `name` has invalid characters, or the markers for `name`
            are malformed in the note.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.\-]+", name):
        raise ValueError(f"invalid section name {name!r}")
    path = pathlib.Path(note)
    text = path.read_text() if path.exists() else NOTE_TEMPLATE.format(title=path.stem)
    block = f"{_begin(name)}\n{content.strip()}\n{_end(name)}"
    begin, end = _begin(name), _end(name)
    if (text.count(begin), text.count(end)) == (1, 1):
        start, stop = text.index(begin), text.index(end) + len(end)
        if stop < start:
            raise ValueError(f"markers for {name!r} are out of order")
        text = text[:start] + block + text[stop:]
    elif begin in text or end in text:
        raise ValueError(f"malformed markers for section {name!r} in {path}")
    else:
        text = _insert_in_results(text, block)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def read_section(note: str | os.PathLike[str], name: str) -> str:
    """Returns the content of the generated section `name` of a note.

    Raises:
        ValueError: The section is not in the note exactly once.
    """
    text = pathlib.Path(note).read_text()
    begin, end = _begin(name), _end(name)
    if (text.count(begin), text.count(end)) != (1, 1):
        raise ValueError(f"section {name!r} not found exactly once in {note}")
    return text[text.index(begin) + len(begin) : text.index(end)].strip()


def parse_table(markdown: str) -> list[dict[str, str]]:
    """Parses the first markdown table in `markdown` (inverse of `markdown_table`).

    Returns:
        One mapping of column title to cell text per body row.
    """
    lines = [ln for ln in markdown.splitlines() if ln.startswith("|")]
    if len(lines) < 2:
        return []

    def cells(line: str) -> list[str]:
        parts = re.split(r"(?<!\\)\|", line.strip()[1:-1])
        return [c.strip().replace("\\|", "|") for c in parts]

    header = cells(lines[0])
    return [dict(zip(header, cells(ln), strict=True)) for ln in lines[2:]]


def _insert_in_results(text: str, block: str) -> str:
    lines = text.splitlines()
    try:
        heading = next(i for i, ln in enumerate(lines) if ln.strip() == RESULTS_HEADING)
    except StopIteration:
        return text.rstrip("\n") + "\n\n" + block + "\n"
    stop = next(
        (i for i in range(heading + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    # Insert after the last non-blank line of the Results section.
    insert_at = stop
    while insert_at > heading + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    new = lines[:insert_at] + ["", *block.splitlines(), ""] + lines[stop:]
    return "\n".join(new).rstrip("\n") + "\n"
