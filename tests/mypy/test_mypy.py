from __future__ import annotations

import os
import re
from collections import defaultdict
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
    ("pyproject-plugin.toml", "constructor.py"),
    ("mypy-plugin.ini", "nullable_defaults.py"),
    ("mypy-plugin-strict.ini", "nullable_defaults.py"),
    ("mypy-plugin.ini", "untyped_fields.py"),
    ("mypy-plugin-no-untyped.ini", "untyped_fields.py"),
    ("mypy-plugin.ini", "inheritance.py"),
    ("mypy-plugin.ini", "relationships.py"),
    ("mypy-plugin-strict.ini", "relationships.py"),
    ("mypy-plugin-pydantic-first.ini", "relationships.py"),
    ("mypy-plugin-pydantic-last.ini", "relationships.py"),
    ("mypy-plugin.ini", "many_to_many_relationships.py"),
    ("mypy-plugin-strict.ini", "many_to_many_relationships.py"),
    ("mypy-plugin.ini", "joins.py"),
    ("mypy-plugin-strict.ini", "joins.py"),
    ("mypy-plugin.ini", "select_varargs.py"),
    ("mypy-plugin-strict.ini", "select_varargs.py"),
    ("mypy-plugin.ini", "session_exec.py"),
    ("mypy-plugin-strict.ini", "session_exec.py"),
    ("mypy-plugin.ini", "session_get.py"),
    ("mypy-plugin-strict.ini", "session_get.py"),
    ("mypy-plugin.ini", "model_construct.py"),
    ("mypy-plugin-strict.ini", "model_construct.py"),
    ("mypy-plugin-strict.ini", "generics.py"),
    ("mypy-plugin.ini", "expressions.py"),
    ("mypy-plugin-strict-no-explicit-any.ini", "expression_typing_strict.py"),
    ("mypy-plugin.ini", "label.py"),
    ("mypy-plugin-strict.ini", "label.py"),
    ("mypy-plugin.ini", "relationship_expressions.py"),
    ("mypy-plugin.ini", "relationship_comparators.py"),
    ("mypy-plugin.ini", "selectinload.py"),
    ("mypy-plugin-strict.ini", "selectinload.py"),
    ("mypy-plugin.ini", "getattr_column_property.py"),
    ("mypy-plugin-strict.ini", "getattr_column_property.py"),
    ("mypy-plugin.ini", "execution_options.py"),
    ("mypy-plugin-strict.ini", "execution_options.py"),
    ("mypy-plugin.ini", "aliases.py"),
    ("mypy-plugin-strict.ini", "aliases.py"),
    ("mypy-plugin.ini", "table_model_awareness.py"),
    ("mypy-plugin-strict.ini", "table_model_awareness.py"),
    ("mypy-plugin.ini", "table_dunders.py"),
    ("mypy-plugin-strict.ini", "table_dunders.py"),
    ("mypy-plugin-strict.ini", "mapped_fields.py"),
    ("mypy-plugin.ini", "field_sa_type.py"),
    ("mypy-plugin-strict.ini", "field_sa_type.py"),
    ("mypy-plugin-strict.ini", "model_config.py"),
    ("mypy-plugin-strict.ini", "relationship_comparators.py"),
    ("mypy-plugin-strict.ini", "docs_parity_read_connected_data.py"),
    ("mypy-plugin-strict.ini", "docs_parity_relationship_attributes.py"),
    ("mypy-plugin-strict.ini", "docs_parity_many_to_many_create_data.py"),
    ("mypy-plugin-strict.ini", "docs_parity_update_patterns.py"),
    ("mypy-plugin-strict.ini", "persisted_helpers.py"),
    ("mypy-plugin-strict.ini", "annotated_field_metadata.py"),
]


def get_expected_return_code(source_code: str) -> int:
    """Return 1 if at least one `# MYPY:` comment was found, else 0."""
    return 1 if re.findall(r"^\s*# MYPY:", source_code, flags=re.MULTILINE) else 0


def group_cases_by_config(cases: list[tuple[str, str]]) -> dict[str, list[str]]:
    """Convert (config, module) cases to config -> [modules] mapping."""
    by_config: dict[str, list[str]] = defaultdict(list)
    for config_filename, python_filename in cases:
        by_config[config_filename].append(python_filename)

    # Preserve order, avoid duplicates.
    out: dict[str, list[str]] = {}
    for config_filename, python_filenames in by_config.items():
        seen: set[str] = set()
        deduped: list[str] = []
        for python_filename in python_filenames:
            if python_filename in seen:
                continue
            seen.add(python_filename)
            deduped.append(python_filename)
        out[config_filename] = deduped
    return out


CASES_BY_CONFIG = group_cases_by_config(cases)


def normalize_mypy_output_by_file(mypy_out: str) -> dict[Path, str]:
    """Return mapping of input file path -> normalized mypy stdout for that file.

    Normalized output matches the old behavior: strip the leading file path up to
    (and including) `.py:`, leaving `line(:col)?: message` lines only.
    """
    lines_by_path: dict[Path, list[str]] = defaultdict(list)

    for line in mypy_out.splitlines():
        if not line:
            continue

        # Example: /abs/path/to/file.py:24:5: error: ...  [code]
        match = re.match(r"^(?P<path>.+?\.py):(?P<rest>.*)$", line)
        if match is None:
            continue

        path_text = match.group("path")
        rest = match.group("rest")

        p = Path(path_text)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        else:
            p = p.resolve()

        lines_by_path[p].append(rest)

    return {path: "\n".join(lines).strip() for path, lines in lines_by_path.items()}


@pytest.mark.parametrize("config_filename", list(CASES_BY_CONFIG.keys()))
def test_mypy_results(config_filename: str, request: pytest.FixtureRequest) -> None:
    if mypy_api is None:  # pragma: no cover
        pytest.skip("mypy is not installed")

    config_path = REPO_ROOT / "tests/mypy/configs" / config_filename
    output_dir = OUTPUTS_DIR / config_path.name.replace(".", "_")

    # Include pid to avoid cross-run incremental cache reuse (keeps coverage stable).
    cache_dir = f".mypy_cache/test-{os.path.splitext(config_filename)[0]}-{os.getpid()}"
    python_filenames = CASES_BY_CONFIG[config_filename]
    input_paths = [(REPO_ROOT / "tests/mypy/modules" / fn).resolve() for fn in python_filenames]

    command = [*(str(p) for p in input_paths)] + [
        "--config-file",
        str(config_path),
        "--cache-dir",
        cache_dir,
        "--show-error-codes",
        "--show-traceback",
    ]
    mypy_out, mypy_err, mypy_returncode = mypy_api.run(command)

    assert mypy_err == ""

    mypy_out_by_file = normalize_mypy_output_by_file(mypy_out)
    update_mypy = request.config.getoption("update_mypy")

    expected_run_return_code = 0
    for input_path in input_paths:
        # Normalize per-file output (avoid OS differences).
        file_mypy_out = mypy_out_by_file.get(input_path, "")
        file_mypy_out = re.sub(r"\n\s*\n", r"\n", file_mypy_out).strip()

        input_code = input_path.read_text()
        output_path = output_dir / input_path.name
        existing_output_code: str | None = (
            output_path.read_text() if output_path.is_file() else None
        )
        expected_output = existing_output_code or input_code

        merged_output = merge_python_and_mypy_output(input_code, file_mypy_out)

        if merged_output != expected_output:
            if update_mypy:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(merged_output)
            else:
                assert existing_output_code is not None, (
                    f"No output file found for {input_path.name} / {config_filename}. "
                    "Run: uv run pytest --update-mypy"
                )
                assert merged_output == existing_output_code

        actual_has_errors = 1 if file_mypy_out else 0
        expected_has_errors = get_expected_return_code(
            merged_output if update_mypy else expected_output
        )
        assert actual_has_errors == expected_has_errors, (
            f"Mismatch between golden expectations and mypy output for "
            f"{input_path.name} / {config_filename}."
        )
        expected_run_return_code = max(expected_run_return_code, expected_has_errors)

    assert mypy_returncode == expected_run_return_code


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
