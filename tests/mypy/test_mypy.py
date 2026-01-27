from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

try:
    from mypy import api as mypy_api
except ImportError:  # pragma: no cover
    mypy_api = None


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = REPO_ROOT / "tests/mypy/outputs"

# Ensure mypy can find the test files regardless of invocation cwd.
os.chdir(REPO_ROOT)


cases: list[tuple[str, str]] = [
    ("mypy-plugin.ini", "constructor.py"),
    ("mypy-plugin-strict.ini", "constructor.py"),
]


def get_expected_return_code(source_code: str) -> int:
    """Return 1 if at least one `# MYPY:` comment was found, else 0."""
    return 1 if re.findall(r"^\s*# MYPY:", source_code, flags=re.MULTILINE) else 0


@pytest.mark.parametrize(["config_filename", "python_filename"], cases)
def test_mypy_results(
    config_filename: str, python_filename: str, request: pytest.FixtureRequest
) -> None:
    if mypy_api is None:  # pragma: no cover
        pytest.skip("mypy is not installed")

    input_path = REPO_ROOT / "tests/mypy/modules" / python_filename
    config_path = REPO_ROOT / "tests/mypy/configs" / config_filename
    output_path = OUTPUTS_DIR / config_path.name.replace(".", "_") / input_path.name

    cache_dir = f".mypy_cache/test-{os.path.splitext(config_filename)[0]}"
    command = [
        str(input_path),
        "--config-file",
        str(config_path),
        "--cache-dir",
        cache_dir,
        "--show-error-codes",
        "--show-traceback",
    ]
    mypy_out, mypy_err, mypy_returncode = mypy_api.run(command)

    assert mypy_err == ""

    # Normalize output (strip file paths; avoid OS differences).
    mypy_out = "\n".join(
        [".py:".join(line.split(".py:")[1:]) for line in mypy_out.split("\n") if line]
    ).strip()
    mypy_out = re.sub(r"\n\s*\n", r"\n", mypy_out)

    input_code = input_path.read_text()
    existing_output_code: str | None = output_path.read_text() if output_path.is_file() else None

    merged_output = merge_python_and_mypy_output(input_code, mypy_out)

    if merged_output == (existing_output_code or input_code):
        return

    if request.config.getoption("update_mypy"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(merged_output)
        return

    assert existing_output_code is not None, (
        f"No output file found for {python_filename} / {config_filename}. "
        "Run: uv run pytest --update-mypy"
    )
    assert merged_output == existing_output_code
    assert mypy_returncode == get_expected_return_code(existing_output_code)


def merge_python_and_mypy_output(source_code: str, mypy_output: str) -> str:
    merged_lines = [(line, False) for line in source_code.splitlines()]

    for line in mypy_output.splitlines()[::-1]:
        if not line:
            continue
        try:
            line_number, message = re.split(r":(?:\d+:)?", line, maxsplit=1)
            merged_lines.insert(int(line_number), (f"# MYPY: {message.strip()}", True))
        except ValueError:
            merged_lines.insert(0, (f"# MYPY: {line.strip()}", True))

    merged_lines = [
        line for line, is_mypy in merged_lines if is_mypy or not line.strip().startswith("# MYPY: ")
    ]
    return "\n".join(merged_lines) + "\n"
