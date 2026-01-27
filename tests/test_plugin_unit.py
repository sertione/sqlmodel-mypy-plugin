from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mypy.nodes import Block, ClassDef, SymbolTable, TypeInfo
from mypy.options import Options

import sqlmodel_mypy.plugin as plugin_mod


def test_plugin_config_defaults_when_no_config_file() -> None:
    options = Options()
    options.config_file = None
    cfg = plugin_mod.SQLModelPluginConfig(options)
    assert cfg.init_typed is False
    assert cfg.init_forbid_extra is False
    assert cfg.warn_untyped_fields is True
    assert cfg.debug_dataclass_transform is False


def test_plugin_config_reads_toml(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.sqlmodel-mypy]
init_typed = true
init_forbid_extra = true
warn_untyped_fields = false
debug_dataclass_transform = true
""".lstrip()
    )

    options = Options()
    options.config_file = str(path)
    cfg = plugin_mod.SQLModelPluginConfig(options)
    assert cfg.init_typed is True
    assert cfg.init_forbid_extra is True
    assert cfg.warn_untyped_fields is False
    assert cfg.debug_dataclass_transform is True


def test_plugin_config_rejects_non_bool_values_in_toml(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.sqlmodel-mypy]
init_typed = 1
""".lstrip()
    )
    options = Options()
    options.config_file = str(path)

    with pytest.raises(ValueError):
        plugin_mod.SQLModelPluginConfig(options)


def test_plugin_config_reads_ini(tmp_path: Path) -> None:
    path = tmp_path / "mypy.ini"
    path.write_text(
        """
[sqlmodel-mypy]
init_typed = true
init_forbid_extra = false
warn_untyped_fields = false
debug_dataclass_transform = true
""".lstrip()
    )
    options = Options()
    options.config_file = str(path)

    cfg = plugin_mod.SQLModelPluginConfig(options)
    assert cfg.init_typed is True
    assert cfg.init_forbid_extra is False
    assert cfg.warn_untyped_fields is False
    assert cfg.debug_dataclass_transform is True


def test_parse_toml_returns_none_for_non_toml() -> None:
    assert plugin_mod.parse_toml("mypy.ini") is None


def test_hooks_selection_smoke() -> None:
    options = Options()
    options.config_file = None
    p = plugin_mod.SQLModelMypyPlugin(options)

    # Metaclass hook: purely string-based.
    assert (
        p.get_metaclass_hook(plugin_mod.SQLMODEL_METACLASS_FULLNAME)
        == p._sqlmodel_metaclass_callback
    )
    assert p.get_metaclass_hook("some.other.Meta") is None

    # Base-class hook depends on lookup_fully_qualified; monkeypatch it.
    cls = ClassDef("SQLModel", Block([]))
    info = TypeInfo(SymbolTable(), cls, "sqlmodel.main")
    cls.info = info
    info._fullname = plugin_mod.SQLMODEL_BASEMODEL_FULLNAME
    p.lookup_fully_qualified = lambda _fullname: SimpleNamespace(node=info)  # type: ignore[method-assign]
    assert p.get_base_class_hook("whatever") == p._sqlmodel_model_class_callback


def test_metaclass_callback_clears_dataclass_transform_spec() -> None:
    options = Options()
    options.config_file = None
    p = plugin_mod.SQLModelMypyPlugin(options)

    meta_type = SimpleNamespace(dataclass_transform_spec=object())
    declared_meta = SimpleNamespace(type=meta_type)
    ctx = SimpleNamespace(
        cls=SimpleNamespace(info=SimpleNamespace(declared_metaclass=declared_meta))
    )
    p._sqlmodel_metaclass_callback(ctx)  # type: ignore[arg-type]
    assert meta_type.dataclass_transform_spec is None


def test_metaclass_callback_respects_debug_flag() -> None:
    options = Options()
    options.config_file = None
    p = plugin_mod.SQLModelMypyPlugin(options)
    p.plugin_config.debug_dataclass_transform = True

    meta_type = SimpleNamespace(dataclass_transform_spec=object())
    declared_meta = SimpleNamespace(type=meta_type)
    ctx = SimpleNamespace(
        cls=SimpleNamespace(info=SimpleNamespace(declared_metaclass=declared_meta))
    )
    p._sqlmodel_metaclass_callback(ctx)  # type: ignore[arg-type]
    assert meta_type.dataclass_transform_spec is not None


def test_model_class_callback_invokes_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    options = Options()
    options.config_file = None
    p = plugin_mod.SQLModelMypyPlugin(options)

    called: dict[str, object] = {}

    class DummyTransformer:
        def __init__(self, cls: object, reason: object, api: object, plugin_config: object) -> None:
            called["init"] = (cls, reason, api, plugin_config)

        def transform(self) -> None:
            called["transform"] = True

    monkeypatch.setattr(plugin_mod, "SQLModelTransformer", DummyTransformer)

    ctx = SimpleNamespace(cls=object(), reason=object(), api=object())
    p._sqlmodel_model_class_callback(ctx)  # type: ignore[arg-type]
    assert called["transform"] is True
