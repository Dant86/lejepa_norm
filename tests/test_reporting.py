"""Generated note sections are inserted under Results and updated idempotently."""

import pathlib

import pytest

from lejepa_norm import reporting

NOTE = """# Exp
## Question
Why?

## Results
Hand-written intro.

## Interpretation
Mine.
"""


def test_sections_insert_under_results_and_update_in_place(
    tmp_path: pathlib.Path,
) -> None:
    note = tmp_path / "idea" / "exp.md"
    note.parent.mkdir()
    note.write_text(NOTE)
    table = reporting.markdown_table([{"a": 1, "b": 0.123456, "c": "x|y"}])
    assert table.splitlines()[0] == "| a | b | c |"
    assert "| 1 | 0.1235 | x\\|y |" in table

    reporting.update_section(note, "t1", table)
    reporting.update_section(note, "t2", "second")
    once = note.read_text()
    reporting.update_section(note, "t1", table)
    reporting.update_section(note, "t2", "second")
    assert note.read_text() == once  # idempotent

    text = once
    assert text.index("Hand-written intro.") < text.index("t1 -->")
    assert text.index("t1 -->") < text.index("t2 -->") < text.index("## Interp")
    assert "Mine." in text

    reporting.update_section(note, "t1", "replaced")
    text = note.read_text()
    assert "replaced" in text and "0.1235" not in text
    assert text.count("BEGIN GENERATED: t1") == 1


def test_missing_note_is_created_from_template_and_links_are_relative(
    tmp_path: pathlib.Path,
) -> None:
    note = tmp_path / "idea" / "new.md"
    reporting.update_section(note, "x", "content")
    text = note.read_text()
    assert text.startswith("# new")
    assert text.index("## Results") < text.index("content") < text.index("## Interp")
    link = reporting.image_link(tmp_path / "idea" / "new_fig.png", note, "fig")
    assert link == "![fig](new_fig.png)"
    with pytest.raises(ValueError):
        reporting.update_section(note, "bad name", "x")
