"""Phase 1 exit criterion.

These tests assert almost nothing about behaviour, which is the point.
They prove the package imports cleanly and the paths line up, so every
later phase has somewhere to put a real test and a green baseline to
break deliberately.
"""
from src import paths


def test_package_imports():
    """Every module imports without side effects at import time."""
    from src import cuisine, embed, ingest, normalise, query, respond, retrieve  # noqa: F401


def test_project_directories_exist():
    assert paths.ROOT.is_dir()
    assert paths.CONFIG_DIR.is_dir()
    assert paths.RAW_DIR.is_dir()
    assert paths.PROCESSED_DIR.is_dir()


def test_paths_sit_inside_the_project():
    """Guards against an absolute path leaking in from someone's machine."""
    for path in (paths.RAW_CSV, paths.RECIPES_PARQUET, paths.VECTORS_NPY):
        assert paths.ROOT in path.parents


def test_scoring_weights_are_sane():
    """The four ranking weights should sum to 1.0 and coverage should dominate."""
    from src import retrieve

    total = (
        retrieve.W_COVERAGE
        + retrieve.W_SIMILARITY
        + retrieve.W_MISSING
        + retrieve.W_REGION
    )
    assert abs(total - 1.0) < 1e-9
    assert retrieve.W_COVERAGE > retrieve.W_SIMILARITY
    assert retrieve.W_COVERAGE > retrieve.W_REGION
