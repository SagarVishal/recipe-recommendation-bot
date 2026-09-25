"""Phase 1 exit criterion.

These tests assert almost nothing about behaviour, which is the point.
They prove the package imports cleanly and the paths line up, so every
later phase has somewhere to put a real test and a green baseline to
break deliberately.
"""
from src import paths


def test_package_imports():
    """Every module imports without side effects at import time."""
    from src import build_corpus, paths  # noqa: F401


def test_committed_directories_exist():
    """Only the directories a fresh clone actually has.

    data/raw and data/processed are gitignored, so asserting they exist
    passes locally and fails in CI - which is exactly what happened.
    """
    assert paths.ROOT.is_dir()
    assert paths.CONFIG_DIR.is_dir()
    assert (paths.ROOT / "notebooks").is_dir()


def test_ensure_dirs_creates_the_data_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(paths, "PROCESSED_DIR", tmp_path / "processed")
    paths.ensure_dirs()
    assert (tmp_path / "raw").is_dir()
    assert (tmp_path / "processed").is_dir()


def test_paths_sit_inside_the_project():
    """Guards against an absolute path leaking in from someone's machine."""
    for path in (paths.RAW_CSV, paths.RECIPES_PARQUET, paths.VECTORS_NPY):
        assert paths.ROOT in path.parents


def test_notebook_is_valid_and_has_all_eight_sections():
    """The notebook is the deliverable, so CI checks it parses and is complete."""
    import ast
    import json

    notebook = json.loads((paths.ROOT / "notebooks" / "rag_walkthrough.ipynb").read_text())
    markdown = "\n".join(
        "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown"
    )
    for heading in (
        "1. Introduction to AI Frameworks",
        "2. Configure the LLM",
        "3. LLM Parameters",
        "4. Build a Basic Chatbot",
        "5. Conversation History & Memory",
        "6. Introduce RAG",
        "7. Build a Small RAG System",
        "8. Final Mini AI System",
    ):
        assert heading in markdown, f"missing section: {heading}"

    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if not source.strip().startswith("%pip"):
                ast.parse(source)  # raises SyntaxError, naming the cell


def test_no_api_key_was_committed():
    """A key in the history is a key that has to be revoked.

    Matches the real shape of a Google key - AIzaSy plus 33 more characters -
    rather than the prefix alone, so documentation placeholders like
    "AIza...your-key..." don't trip it.
    """
    import re

    real_key = re.compile(r"AIzaSy[A-Za-z0-9_\-]{33}")
    for path in paths.ROOT.rglob("*"):
        if ".git" in path.parts or "data/raw" in str(path):
            continue
        if path.is_file() and path.suffix in {".py", ".ipynb", ".md", ".yaml", ".csv", ".txt"}:
            assert not real_key.search(path.read_text(errors="ignore")), f"API key in {path}"


def test_no_undefined_names_anywhere():
    """A NameError shipped to the browser once: `rag.py` returned a variable
    that had been renamed. ast.parse catches syntax, not undefined names -
    pyflakes catches both, so it runs in CI."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "pyflakes",
         str(paths.ROOT / "src"), str(paths.ROOT / "app.py")],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"pyflakes findings:\n{result.stdout}"
