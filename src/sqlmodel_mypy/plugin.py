"""Mypy plugin entry point for SQLModel."""

from __future__ import annotations

import sys
from collections.abc import Callable
from configparser import ConfigParser
from typing import Any

from mypy.nodes import TypeInfo
from mypy.options import Options
from mypy.plugin import ClassDefContext, Plugin, ReportConfigContext

from .transform import SQLModelTransformer

CONFIGFILE_KEY = "sqlmodel-mypy"
SQLMODEL_BASEMODEL_FULLNAME = "sqlmodel.main.SQLModel"
SQLMODEL_METACLASS_FULLNAME = "sqlmodel.main.SQLModelMetaclass"

# Increment when plugin changes should invalidate mypy cache.
__version__ = 2


def plugin(version: str) -> type[Plugin]:
    # `version` is the mypy version string; keep the signature required by mypy.
    return SQLModelMypyPlugin


class SQLModelMypyPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        self.plugin_config = SQLModelPluginConfig(options)
        self._plugin_data = {"__version__": __version__, **self.plugin_config.to_data()}
        super().__init__(options)

    def get_base_class_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        sym = self.lookup_fully_qualified(fullname)
        if sym and isinstance(sym.node, TypeInfo):
            if sym.node.fullname == SQLMODEL_BASEMODEL_FULLNAME or sym.node.has_base(
                SQLMODEL_BASEMODEL_FULLNAME
            ):
                return self._sqlmodel_model_class_callback
        return None

    def get_metaclass_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        if fullname == SQLMODEL_METACLASS_FULLNAME:
            return self._sqlmodel_metaclass_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> dict[str, Any]:
        # Used by mypy to decide whether to invalidate incremental caches.
        return self._plugin_data

    def _sqlmodel_model_class_callback(self, ctx: ClassDefContext) -> None:
        transformer = SQLModelTransformer(ctx.cls, ctx.reason, ctx.api, self.plugin_config)
        transformer.transform()

    def _sqlmodel_metaclass_callback(self, ctx: ClassDefContext) -> None:
        """Disable dataclass-transform handling for SQLModel metaclass.

        SQLModel decorates `SQLModelMetaclass` with `__dataclass_transform__`, but we want this plugin
        (not mypy's generic dataclass-transform logic) to generate the model signatures.
        """
        if self.plugin_config.debug_dataclass_transform:
            return
        info_metaclass = ctx.cls.info.declared_metaclass
        if info_metaclass is None:
            return
        if getattr(info_metaclass.type, "dataclass_transform_spec", None) is not None:
            info_metaclass.type.dataclass_transform_spec = None


class SQLModelPluginConfig:
    __slots__ = (
        "init_typed",
        "init_forbid_extra",
        "warn_untyped_fields",
        "debug_dataclass_transform",
    )

    init_typed: bool
    init_forbid_extra: bool
    warn_untyped_fields: bool
    debug_dataclass_transform: bool  # undocumented, for testing

    def __init__(self, options: Options) -> None:
        # Defaults
        self.init_typed = False
        self.init_forbid_extra = False
        self.warn_untyped_fields = True
        self.debug_dataclass_transform = False

        if options.config_file is None:
            return

        toml_config = parse_toml(options.config_file)
        if toml_config is not None:
            config = toml_config.get("tool", {}).get(CONFIGFILE_KEY, {})
            for key in self.__slots__:
                value = config.get(key, getattr(self, key))
                if not isinstance(value, bool):
                    raise ValueError(f"Configuration value must be a boolean for key: {key}")
                setattr(self, key, value)
            return

        parser = ConfigParser()
        parser.read(options.config_file)
        for key in self.__slots__:
            setattr(self, key, parser.getboolean(CONFIGFILE_KEY, key, fallback=getattr(self, key)))

    def to_data(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__slots__}


def parse_toml(config_file: str) -> dict[str, Any] | None:
    """Parse `pyproject.toml` and return it as a dict (or None if not TOML)."""
    if not config_file.endswith(".toml"):
        return None

    if sys.version_info >= (3, 11):
        import tomllib as toml_
    else:
        try:
            import tomli as toml_
        except ImportError:  # pragma: no cover
            return None

    with open(config_file, "rb") as f:
        return toml_.load(f)
