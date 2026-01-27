from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-mypy",
        action="store_true",
        default=False,
        help="Update tests/mypy/outputs/* golden files from current mypy output.",
    )
