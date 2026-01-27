from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mypy.nodes import (
    ARG_NAMED,
    ARG_OPT,
    ARG_STAR2,
    Block,
    CallExpr,
    ClassDef,
    FuncDef,
    IfStmt,
    NameExpr,
    SymbolTable,
    SymbolTableNode,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import AttributeContext, FunctionContext, FunctionSigContext, MethodSigContext
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    Type,
    TypeOfAny,
    TypeType,
)

import sqlmodel_mypy.plugin as plugin_mod


class DummyCheckerAPI:
    """Minimal CheckerPluginInterface stub for plugin hook unit tests."""

    def __init__(self) -> None:
        self.options = SimpleNamespace(strict_optional=True)
        self.failed: list[tuple[str, object | None]] = []
        self._infos: dict[str, TypeInfo] = {}

    def fail(self, msg: str, ctx: object, /, *, code: object | None = None) -> object | None:
        self.failed.append((msg, code))
        return None

    def named_generic_type(self, name: str, args: list[Type]) -> Instance:
        info = self._infos.get(name)
        if info is None:
            cls = ClassDef(name.split(".")[-1], Block([]))
            info = TypeInfo(SymbolTable(), cls, name.rsplit(".", 1)[0])
            cls.info = info
            info._fullname = name
            self._infos[name] = info
        return Instance(info, args)


def make_typeinfo(fullname: str) -> TypeInfo:
    module, name = fullname.rsplit(".", 1)
    cls = ClassDef(name, Block([]))
    info = TypeInfo(SymbolTable(), cls, module)
    cls.info = info
    info._fullname = fullname
    return info


def make_call(fullname: str) -> CallExpr:
    callee = NameExpr("callable")
    callee.fullname = fullname
    return CallExpr(callee, [], [], [])


def make_sqlmodel_class(fullname: str) -> TypeInfo:
    sqlmodel_info = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    info = make_typeinfo(fullname)
    info.mro = [info, sqlmodel_info, obj_info]
    return info


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


def test_iter_assignment_statements_includes_if_and_else() -> None:
    body1 = Block([plugin_mod.AssignmentStmt([NameExpr("a")], NameExpr("x"))])
    body1.body[0].new_syntax = True  # type: ignore[attr-defined]
    body2 = Block([plugin_mod.AssignmentStmt([NameExpr("b")], NameExpr("y"))])
    body2.body[0].new_syntax = True  # type: ignore[attr-defined]
    if_stmt = IfStmt([NameExpr("True")], [body1], body2)
    block = Block([if_stmt])

    names = [
        stmt.lvalues[0].name for stmt in plugin_mod._iter_assignment_statements_from_block(block)
    ]
    assert names == ["a", "b"]


def test_constructor_signature_hook_collects_fields_and_skips_relationships() -> None:
    # Base model with field `x`, derived model overrides `x` with a relationship.
    sqlmodel_base = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_base, obj_info]

    model_info = make_typeinfo("m.User")
    model_info.mro = [model_info, base_info, sqlmodel_base, obj_info]

    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])

    # Base: x: int = Field()
    base_var = Var("x", int_t)
    base_info.names["x"] = SymbolTableNode(0, base_var)
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    base_info.defn.defs.body.append(stmt)

    # Derived: x: int = Relationship(...)  (overrides base field, should be excluded from constructor)
    stmt2 = plugin_mod.AssignmentStmt(
        [NameExpr("x")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt2.new_syntax = True
    model_info.defn.defs.body.append(stmt2)

    # Derived: y: int = Field()
    model_var = Var("y", int_t)
    model_info.names["y"] = SymbolTableNode(0, model_var)
    stmt3 = plugin_mod.AssignmentStmt(
        [NameExpr("y")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt3.new_syntax = True
    model_info.defn.defs.body.append(stmt3)

    # Plugin lookup: resolve the SQLModel subclass and SQLModel base.
    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.lookup_fully_qualified = lambda full: SimpleNamespace(  # type: ignore[method-assign]
        node={"m.User": model_info}.get(full)
    )

    api = DummyCheckerAPI()
    default_sig = CallableType(
        [], [], [], Instance(model_info, []), Instance(make_typeinfo("builtins.function"), [])
    )
    hook = p.get_function_signature_hook("m.User")
    assert hook is not None
    sig = hook(
        FunctionSigContext(args=[], default_signature=default_sig, context=NameExpr("x"), api=api)
    )
    assert isinstance(sig, CallableType)
    assert sig.arg_names == ["y", "kwargs"]
    assert sig.arg_kinds == [ARG_NAMED, ARG_STAR2]


def test_constructor_signature_unwraps_mapped_when_typed() -> None:
    model_info = make_sqlmodel_class("m.User")

    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])

    mapped_info = make_typeinfo("sqlalchemy.orm.base.Mapped")
    mapped_t = Instance(mapped_info, [int_t])

    var = Var("x", mapped_t)
    model_info.names["x"] = SymbolTableNode(0, var)
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    model_info.defn.defs.body.append(stmt)

    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.plugin_config.init_typed = True
    p.lookup_fully_qualified = lambda full: SimpleNamespace(  # type: ignore[method-assign]
        node={"m.User": model_info}.get(full)
    )

    api = DummyCheckerAPI()
    default_sig = CallableType(
        [], [], [], Instance(model_info, []), Instance(make_typeinfo("builtins.function"), [])
    )
    hook = p.get_function_signature_hook("m.User")
    assert hook is not None
    sig = hook(
        FunctionSigContext(args=[], default_signature=default_sig, context=NameExpr("x"), api=api)
    )
    assert isinstance(sig, CallableType)
    assert sig.arg_names[0] == "x"
    assert sig.arg_kinds[0] == ARG_NAMED
    assert isinstance(sig.arg_types[0], Instance)
    assert sig.arg_types[0].type.fullname == "builtins.int"


def test_model_construct_signature_hook() -> None:
    model_info = make_sqlmodel_class("m.User")

    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])
    var = Var("x", int_t)
    model_info.names["x"] = SymbolTableNode(0, var)
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    model_info.defn.defs.body.append(stmt)

    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    default_sig = CallableType(
        [], [], [], Instance(model_info, []), Instance(make_typeinfo("builtins.function"), [])
    )
    ctx = MethodSigContext(
        type=TypeType(Instance(model_info, [])),
        args=[],
        default_signature=default_sig,
        context=NameExpr("x"),
        api=api,
    )
    sig = p._sqlmodel_model_construct_signature_callback(ctx)  # type: ignore[arg-type]
    assert isinstance(sig, CallableType)
    assert sig.arg_names[:2] == ["_fields_set", "x"]
    assert sig.arg_kinds[:2] == [ARG_OPT, ARG_NAMED]


def test_class_attribute_hook_wraps_to_instrumented_attribute() -> None:
    model_info = make_sqlmodel_class("m.User")

    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])
    var = Var("x", int_t)
    model_info.names["x"] = SymbolTableNode(0, var)
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    model_info.defn.defs.body.append(stmt)

    inst_attr_info = make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute")

    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.lookup_fully_qualified = lambda full: SimpleNamespace(  # type: ignore[method-assign]
        node={
            "m.User": model_info,
            "sqlalchemy.orm.attributes.InstrumentedAttribute": inst_attr_info,
        }.get(full)
    )

    hook = p.get_class_attribute_hook("m.User.x")
    assert hook is not None
    ctx = AttributeContext(
        type=TypeType(Instance(model_info, [])),
        default_attr_type=int_t,
        is_lvalue=False,
        context=NameExpr("x"),
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.attributes.InstrumentedAttribute"


def test_col_function_hook_returns_mapped_value_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    mapped_info = make_typeinfo("sqlalchemy.orm.base.Mapped")
    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])

    inst_attr_info = make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute")
    inst_attr_t = Instance(inst_attr_info, [int_t])

    p.lookup_fully_qualified = lambda full: SimpleNamespace(  # type: ignore[method-assign]
        node={
            "sqlalchemy.orm.base.Mapped": mapped_info,
            "sqlalchemy.orm.attributes.InstrumentedAttribute": inst_attr_info,
        }.get(full)
    )

    hook = p.get_function_hook(plugin_mod.SQLMODEL_COL_FULLNAME)
    assert hook is not None

    ctx = FunctionContext(
        arg_types=[[inst_attr_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column_expression"],
        arg_names=[[None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.base.Mapped"
    assert t.args and isinstance(t.args[0], Instance) and t.args[0].type.fullname == "builtins.int"


def test_get_function_signature_hook_returns_none_for_missing_symbol() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.lookup_fully_qualified = lambda _full: None  # type: ignore[method-assign]
    assert p.get_function_signature_hook("m.User") is None


def test_get_function_signature_hook_skips_base_and_non_sqlmodel() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    sqlmodel_base = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")
    sqlmodel_base.mro = [sqlmodel_base, obj_info]

    other = make_typeinfo("m.Other")
    other.mro = [other, obj_info]

    lookup_map = {
        plugin_mod.SQLMODEL_BASEMODEL_FULLNAME: sqlmodel_base,
        "m.Other": other,
    }
    p.lookup_fully_qualified = (  # type: ignore[method-assign]
        lambda full: SimpleNamespace(node=lookup_map[full]) if full in lookup_map else None
    )

    assert p.get_function_signature_hook(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME) is None
    assert p.get_function_signature_hook("m.Other") is None


def test_get_base_class_hook_returns_none_for_non_sqlmodel() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    cls = ClassDef("Other", Block([]))
    info = TypeInfo(SymbolTable(), cls, "m")
    cls.info = info
    info._fullname = "m.Other"
    info.mro = [info, make_typeinfo("builtins.object")]

    p.lookup_fully_qualified = lambda _full: SimpleNamespace(node=info)  # type: ignore[method-assign]
    assert p.get_base_class_hook("m.Other") is None


def test_get_method_signature_hook_selects_only_model_construct() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    assert p.get_method_signature_hook("m.User.other") is None
    hook = p.get_method_signature_hook("m.User.model_construct")
    assert hook is not None
    assert getattr(hook, "__self__", None) is p
    assert (
        getattr(hook, "__func__", None)
        is plugin_mod.SQLModelMypyPlugin._sqlmodel_model_construct_signature_callback
    )


def test_get_class_attribute_hook_guard_paths() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    assert p.get_class_attribute_hook("NoDot") is None
    assert p.get_class_attribute_hook("m.User._x") is None
    assert p.get_class_attribute_hook("m.User.model_config") is None

    # Owner not found.
    p.lookup_fully_qualified = lambda _full: None  # type: ignore[method-assign]
    assert p.get_class_attribute_hook("m.User.x") is None

    # Owner is SQLModel base.
    sqlmodel_base = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    sqlmodel_base.mro = [sqlmodel_base, make_typeinfo("builtins.object")]
    p.lookup_fully_qualified = lambda _full: SimpleNamespace(node=sqlmodel_base)  # type: ignore[method-assign]
    assert p.get_class_attribute_hook(f"{plugin_mod.SQLMODEL_BASEMODEL_FULLNAME}.x") is None

    # Owner is not a SQLModel subclass.
    non_model = make_typeinfo("m.Other")
    non_model.mro = [non_model, make_typeinfo("builtins.object")]
    p.lookup_fully_qualified = lambda _full: SimpleNamespace(node=non_model)  # type: ignore[method-assign]
    assert p.get_class_attribute_hook("m.Other.x") is None

    # Owner is a SQLModel subclass but member is not declared.
    model_info = make_sqlmodel_class("m.User")
    p.lookup_fully_qualified = (  # type: ignore[method-assign]
        lambda full: SimpleNamespace(node=model_info) if full == "m.User" else None
    )
    assert p.get_class_attribute_hook("m.User.x") is None


def test_declares_sqlmodel_member_false_for_untyped_or_missing_assignment() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    info = make_sqlmodel_class("m.User")

    int_t = Instance(make_typeinfo("builtins.int"), [])
    var = Var("x", int_t)
    info.names["x"] = SymbolTableNode(0, var)

    # Var exists but no assignment in class body.
    assert p._declares_sqlmodel_member(info, "x") is False

    # Assignment exists but is untyped -> still not a declared member for class attribute typing.
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = False
    info.defn.defs.body.append(stmt)
    assert p._declares_sqlmodel_member(info, "x") is False


def test_collect_fields_for_signature_reports_invalid_override() -> None:
    sqlmodel_base = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_base, obj_info]

    model_info = make_typeinfo("m.User")
    model_info.mro = [model_info, base_info, sqlmodel_base, obj_info]

    int_t = Instance(make_typeinfo("builtins.int"), [])

    # Base: x: int = Field()
    base_var = Var("x", int_t)
    base_info.names["x"] = SymbolTableNode(0, base_var)
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    base_info.defn.defs.body.append(stmt)

    # Derived overrides `x` with a non-Var (invalid override).
    model_info.names["x"] = SymbolTableNode(0, FuncDef("x", [], Block([])))

    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    fields = p._collect_fields_for_signature(model_info, api)
    assert {f.name for f in fields} == set()
    assert any("SQLModel field may only be overridden" in msg for msg, _ in api.failed)


def test_collect_member_from_stmt_untyped_and_skip_paths() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    info = make_sqlmodel_class("m.User")

    # Untyped field -> warn.
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = False
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None
    assert ("Untyped fields disallowed", plugin_mod.ERROR_FIELD) in api.failed

    # Untyped relationship -> warn.
    api.failed.clear()
    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("rel")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt.new_syntax = False
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None
    assert ("Untyped fields disallowed", plugin_mod.ERROR_FIELD) in api.failed

    # Typed but lvalue is not NameExpr.
    stmt = plugin_mod.AssignmentStmt(
        [make_call("m.Call")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None

    # Typed but name is ignored.
    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("model_config")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None

    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("_hidden")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None

    # Typed but no symbol table entry -> ignored.
    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("missing")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None

    # Typed but symbol is not Var -> ignored.
    info.names["bad"] = SymbolTableNode(0, FuncDef("bad", [], Block([])))
    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("bad")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None

    # Typed but classvar -> ignored.
    classvar = Var("cv", AnyType(TypeOfAny.explicit))
    classvar.is_classvar = True
    info.names["cv"] = SymbolTableNode(0, classvar)
    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("cv")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api) is None


def test_model_construct_signature_callback_guard_paths_and_instance_receiver() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    default_sig = CallableType(
        [], [], [], AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.function"), [])
    )

    # Unknown receiver type -> no-op.
    ctx = MethodSigContext(
        type=AnyType(TypeOfAny.explicit),
        args=[],
        default_signature=default_sig,
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_model_construct_signature_callback(ctx) is default_sig  # type: ignore[arg-type]

    # Receiver is SQLModel base -> no-op.
    sqlmodel_base = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    sqlmodel_base.mro = [sqlmodel_base, make_typeinfo("builtins.object")]
    ctx = MethodSigContext(
        type=TypeType(Instance(sqlmodel_base, [])),
        args=[],
        default_signature=default_sig,
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_model_construct_signature_callback(ctx) is default_sig

    # Receiver is not a SQLModel subclass -> no-op.
    other = make_typeinfo("m.Other")
    other.mro = [other, make_typeinfo("builtins.object")]
    ctx = MethodSigContext(
        type=TypeType(Instance(other, [])),
        args=[],
        default_signature=default_sig,
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_model_construct_signature_callback(ctx) is default_sig

    # Instance receiver for an SQLModel subclass.
    model_info = make_sqlmodel_class("m.User")
    int_t = Instance(make_typeinfo("builtins.int"), [])
    model_info.names["x"] = SymbolTableNode(0, Var("x", int_t))
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    model_info.defn.defs.body.append(stmt)

    receiver = Instance(model_info, [])
    ctx = MethodSigContext(
        type=receiver,
        args=[],
        default_signature=default_sig,
        context=NameExpr("x"),
        api=api,
    )
    sig = p._sqlmodel_model_construct_signature_callback(ctx)
    assert isinstance(sig, CallableType)
    assert sig.ret_type == receiver


def test_class_attr_type_callback_guard_paths() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    int_t = Instance(make_typeinfo("builtins.int"), [])

    # Lvalue access: don't change type.
    ctx = AttributeContext(
        type=AnyType(TypeOfAny.explicit),
        default_attr_type=int_t,
        is_lvalue=True,
        context=NameExpr("x"),
        api=DummyCheckerAPI(),
    )
    assert p._sqlmodel_class_attr_type_callback(ctx) == int_t  # type: ignore[arg-type]

    # Already Mapped[T]: no change.
    mapped_t = Instance(make_typeinfo("sqlalchemy.orm.base.Mapped"), [int_t])
    ctx = AttributeContext(
        type=AnyType(TypeOfAny.explicit),
        default_attr_type=mapped_t,
        is_lvalue=False,
        context=NameExpr("x"),
        api=DummyCheckerAPI(),
    )
    assert p._sqlmodel_class_attr_type_callback(ctx) == mapped_t  # type: ignore[arg-type]

    # Already InstrumentedAttribute[T]: no change.
    inst_attr_t = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute"), [int_t]
    )
    ctx = AttributeContext(
        type=AnyType(TypeOfAny.explicit),
        default_attr_type=inst_attr_t,
        is_lvalue=False,
        context=NameExpr("x"),
        api=DummyCheckerAPI(),
    )
    assert p._sqlmodel_class_attr_type_callback(ctx) == inst_attr_t  # type: ignore[arg-type]

    # InstrumentedAttribute type not found -> no change.
    p.lookup_fully_qualified = lambda _full: None  # type: ignore[method-assign]
    ctx = AttributeContext(
        type=AnyType(TypeOfAny.explicit),
        default_attr_type=int_t,
        is_lvalue=False,
        context=NameExpr("x"),
        api=DummyCheckerAPI(),
    )
    assert p._sqlmodel_class_attr_type_callback(ctx) == int_t  # type: ignore[arg-type]


def test_col_return_type_callback_guard_paths() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    default_ret = AnyType(TypeOfAny.explicit)

    # No arguments -> no-op.
    ctx = FunctionContext(
        arg_types=[],
        arg_kinds=[],
        callee_arg_names=[],
        arg_names=[],
        default_return_type=default_ret,
        args=[],
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_col_return_type_callback(ctx) == default_ret

    # Non-Mapped / non-InstrumentedAttribute -> no-op.
    int_t = Instance(make_typeinfo("builtins.int"), [])
    ctx = FunctionContext(
        arg_types=[[int_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column_expression"],
        arg_names=[[None]],
        default_return_type=default_ret,
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_col_return_type_callback(ctx) == default_ret

    # Mapped[T] argument -> returns Mapped[T].
    mapped_info = make_typeinfo("sqlalchemy.orm.base.Mapped")
    mapped_t = Instance(mapped_info, [int_t])
    p.lookup_fully_qualified = lambda full: SimpleNamespace(  # type: ignore[method-assign]
        node={"sqlalchemy.orm.base.Mapped": mapped_info}.get(full)
    )
    ctx = FunctionContext(
        arg_types=[[mapped_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column_expression"],
        arg_names=[[None]],
        default_return_type=default_ret,
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    t = p._sqlmodel_col_return_type_callback(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.base.Mapped"

    # Mapped type not found -> no-op.
    p.lookup_fully_qualified = lambda _full: None  # type: ignore[method-assign]
    inst_attr_t = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute"), [int_t]
    )
    ctx = FunctionContext(
        arg_types=[[inst_attr_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column_expression"],
        arg_names=[[None]],
        default_return_type=default_ret,
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_col_return_type_callback(ctx) == default_ret


def test_get_function_hook_returns_none_for_other_functions() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    assert p.get_function_hook("sqlmodel.sql.expression.not_col") is None


def test_metaclass_callback_handles_missing_declared_metaclass() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    ctx = SimpleNamespace(cls=SimpleNamespace(info=SimpleNamespace(declared_metaclass=None)))
    p._sqlmodel_metaclass_callback(ctx)  # type: ignore[arg-type]
