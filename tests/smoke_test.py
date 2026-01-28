from __future__ import annotations

from importlib import metadata
from importlib.resources import files


def main() -> None:
    # Import package and key modules.
    import sqlmodel_mypy  # noqa: F401
    from sqlmodel_mypy import plugin as plugin_mod

    # Ensure wheel metadata is present.
    _ = metadata.version("sqlmodel-mypy-plugin")

    # Sanity check plugin entrypoint.
    assert callable(plugin_mod.plugin)

    # Ensure PEP 561 marker is included in the distribution.
    assert files("sqlmodel_mypy").joinpath("py.typed").is_file()


if __name__ == "__main__":
    main()
