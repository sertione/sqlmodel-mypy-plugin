from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR,
    ARG_STAR2,
    Block,
    CallExpr,
    ClassDef,
    FuncDef,
    IfStmt,
    MemberExpr,
    NameExpr,
    StrExpr,
    SymbolTable,
    SymbolTableNode,
    TempNode,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import (
    AttributeContext,
    FunctionContext,
    FunctionSigContext,
    MethodContext,
    MethodSigContext,
)
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    LiteralType,
    NoneType,
    Overloaded,
    TupleType,
    Type,
    TypeOfAny,
    TypeType,
    UnionType,
    get_proper_type,
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


def test_is_table_model_uses_persisted_table_metadata() -> None:
    info = make_sqlmodel_class("m.User")
    info.metadata[plugin_mod.METADATA_KEY] = {"table": True}

    assert plugin_mod._is_table_model(info) is True


def test_is_table_model_inherits_from_base_metadata() -> None:
    sqlmodel_info = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_info, obj_info]
    base_info.metadata[plugin_mod.METADATA_KEY] = {"is_table_model": True}

    info = make_typeinfo("m.User")
    info.mro = [info, base_info, sqlmodel_info, obj_info]
    info.metadata[plugin_mod.METADATA_KEY] = {"fields": {}, "relationships": {}}

    assert plugin_mod._is_table_model(info) is True


def test_is_table_model_detects_model_config_call_table_true() -> None:
    info = make_sqlmodel_class("m.User")
    true_expr = NameExpr("True")
    true_expr.fullname = "builtins.True"
    config_call = CallExpr(NameExpr("ConfigDict"), [true_expr], [ARG_NAMED], ["table"])
    stmt = plugin_mod.AssignmentStmt([NameExpr("model_config")], config_call)
    info.defn.defs.body.append(stmt)

    assert plugin_mod._is_table_model(info) is True


def test_is_table_model_detects_model_config_call_table_false() -> None:
    info = make_sqlmodel_class("m.User")
    false_expr = NameExpr("False")
    false_expr.fullname = "builtins.False"
    config_call = CallExpr(NameExpr("ConfigDict"), [false_expr], [ARG_NAMED], ["table"])
    stmt = plugin_mod.AssignmentStmt([NameExpr("model_config")], config_call)
    info.defn.defs.body.append(stmt)

    assert plugin_mod._is_table_model(info) is False


def test_is_table_model_detects_model_config_dict_table_true() -> None:
    info = make_sqlmodel_class("m.User")
    true_expr = NameExpr("True")
    true_expr.fullname = "builtins.True"
    config_dict = plugin_mod.DictExpr([(StrExpr("table"), true_expr)])
    stmt = plugin_mod.AssignmentStmt([NameExpr("model_config")], config_dict)
    info.defn.defs.body.append(stmt)

    assert plugin_mod._is_table_model(info) is True


def test_is_table_model_detects_model_config_dict_table_false() -> None:
    info = make_sqlmodel_class("m.User")
    false_expr = NameExpr("False")
    false_expr.fullname = "builtins.False"
    config_dict = plugin_mod.DictExpr(
        [(StrExpr("extra"), StrExpr("forbid")), (StrExpr("table"), false_expr)]
    )
    stmt = plugin_mod.AssignmentStmt([NameExpr("model_config")], config_dict)
    info.defn.defs.body.append(stmt)

    assert plugin_mod._is_table_model(info) is False


def test_is_table_model_inherits_from_base_model_config() -> None:
    sqlmodel_info = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_info, obj_info]
    true_expr = NameExpr("True")
    true_expr.fullname = "builtins.True"
    config_call = CallExpr(NameExpr("ConfigDict"), [true_expr], [ARG_NAMED], ["table"])
    base_info.defn.defs.body.append(
        plugin_mod.AssignmentStmt([NameExpr("model_config")], config_call)
    )

    info = make_typeinfo("m.User")
    info.mro = [info, base_info, sqlmodel_info, obj_info]

    assert plugin_mod._is_table_model(info) is True


def test_is_bool_nameexpr_returns_false_for_non_nameexpr() -> None:
    assert plugin_mod._is_bool_nameexpr(StrExpr("x"), True) is False


def test_is_table_model_inherits_from_base_table_keyword() -> None:
    sqlmodel_info = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_info, obj_info]
    base_info.defn.keywords["table"] = NameExpr("True")

    info = make_typeinfo("m.User")
    info.mro = [info, base_info, sqlmodel_info, obj_info]

    assert plugin_mod._is_table_model(info) is True


def test_table_value_from_model_config_skips_empty_lvalues() -> None:
    info = make_sqlmodel_class("m.User")
    dummy = plugin_mod.AssignmentStmt([NameExpr("x")], NameExpr("y"))
    dummy.lvalues = []  # type: ignore[assignment]
    info.defn.defs.body.append(dummy)

    true_expr = NameExpr("True")
    true_expr.fullname = "builtins.True"
    config_call = CallExpr(NameExpr("ConfigDict"), [true_expr], [ARG_NAMED], ["table"])
    info.defn.defs.body.append(plugin_mod.AssignmentStmt([NameExpr("model_config")], config_call))

    assert plugin_mod._is_table_model(info) is True


def test_plugin_config_defaults_when_no_config_file() -> None:
    options = Options()
    options.config_file = None
    cfg = plugin_mod.SQLModelPluginConfig(options)
    assert cfg.init_typed is False
    assert cfg.init_forbid_extra is False
    assert cfg.warn_untyped_fields is True
    assert cfg.typed_execute is False
    assert cfg.debug_dataclass_transform is False


def test_plugin_config_reads_toml(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.sqlmodel-mypy]
init_typed = true
init_forbid_extra = true
warn_untyped_fields = false
typed_execute = true
debug_dataclass_transform = true
""".lstrip()
    )

    options = Options()
    options.config_file = str(path)
    cfg = plugin_mod.SQLModelPluginConfig(options)
    assert cfg.init_typed is True
    assert cfg.init_forbid_extra is True
    assert cfg.warn_untyped_fields is False
    assert cfg.typed_execute is True
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
typed_execute = true
debug_dataclass_transform = true
""".lstrip()
    )
    options = Options()
    options.config_file = str(path)

    cfg = plugin_mod.SQLModelPluginConfig(options)
    assert cfg.init_typed is True
    assert cfg.init_forbid_extra is False
    assert cfg.warn_untyped_fields is False
    assert cfg.typed_execute is True
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

    def named_type(fullname: str, args: list[object] | None = None) -> object:
        # Minimal stand-in for SemanticAnalyzerPluginInterface.named_type.
        return {"fullname": fullname, "args": args or []}

    class DummyTransformer:
        def __init__(self, cls: object, reason: object, api: object, plugin_config: object) -> None:
            called["init"] = (cls, reason, api, plugin_config)

        def transform(self) -> None:
            called["transform"] = True

    monkeypatch.setattr(plugin_mod, "SQLModelTransformer", DummyTransformer)

    # Provide the minimal shape expected by the plugin callback.
    p.lookup_fully_qualified = lambda _fullname: None  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        cls=SimpleNamespace(
            info=SimpleNamespace(names={}, defn=SimpleNamespace(keywords={}), mro=[])
        ),
        reason=object(),
        api=SimpleNamespace(named_type=named_type),
    )
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


def test_constructor_signature_hook_includes_relationship_kwargs_for_table_model() -> None:
    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")

    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])

    team_info = make_typeinfo("m.Team")
    team_t = Instance(team_info, [])

    # name: int = Field()
    name_var = Var("name", int_t)
    model_info.names["name"] = SymbolTableNode(0, name_var)
    stmt_name = plugin_mod.AssignmentStmt(
        [NameExpr("name")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt_name.new_syntax = True
    model_info.defn.defs.body.append(stmt_name)

    # team: Team = Relationship()
    team_var = Var("team", team_t)
    model_info.names["team"] = SymbolTableNode(0, team_var)
    stmt_team = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True
    model_info.defn.defs.body.append(stmt_team)

    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.plugin_config.init_typed = True
    p.plugin_config.init_forbid_extra = True
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
    assert sig.arg_names == ["name", "team"]
    assert sig.arg_kinds == [ARG_NAMED, ARG_NAMED_OPT]

    assert isinstance(sig.arg_types[0], Instance)
    assert sig.arg_types[0].type.fullname == "builtins.int"
    assert isinstance(sig.arg_types[1], Instance)
    assert sig.arg_types[1].type.fullname == "m.Team"


def test_model_construct_signature_hook_includes_relationship_kwargs_for_table_model() -> None:
    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")

    int_info = make_typeinfo("builtins.int")
    int_t = Instance(int_info, [])

    team_info = make_typeinfo("m.Team")
    team_t = Instance(team_info, [])

    var = Var("x", int_t)
    model_info.names["x"] = SymbolTableNode(0, var)
    stmt = plugin_mod.AssignmentStmt([NameExpr("x")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    model_info.defn.defs.body.append(stmt)

    team_var = Var("team", team_t)
    model_info.names["team"] = SymbolTableNode(0, team_var)
    stmt_team = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True
    model_info.defn.defs.body.append(stmt_team)

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
    assert sig.arg_names[:3] == ["_fields_set", "x", "team"]
    assert sig.arg_kinds[:3] == [ARG_OPT, ARG_NAMED, ARG_NAMED_OPT]


def test_select_signature_hook_supports_more_than_4_entities() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    p.lookup_fully_qualified = lambda full: (  # type: ignore[method-assign]
        SimpleNamespace(node=select_info) if full == "sqlalchemy.sql.selectable.Select" else None
    )

    default_sig = CallableType(
        [], [], [], AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.function"), [])
    )

    call = CallExpr(
        NameExpr("select"),
        [NameExpr("a"), NameExpr("b"), NameExpr("c"), NameExpr("d"), NameExpr("e")],
        [ARG_POS, ARG_POS, ARG_POS, ARG_POS, ARG_POS],
        [None, None, None, None, None],
    )

    hook = p.get_function_signature_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME)
    assert hook is not None

    api = DummyCheckerAPI()
    sig = hook(FunctionSigContext(args=[], default_signature=default_sig, context=call, api=api))
    assert isinstance(sig, CallableType)
    assert sig.arg_kinds == [ARG_POS, ARG_POS, ARG_POS, ARG_POS, ARG_POS]
    assert sig.arg_names == ["__ent0", "__ent1", "__ent2", "__ent3", "__ent4"]

    ret = get_proper_type(sig.ret_type)
    assert isinstance(ret, Instance)
    assert ret.type.fullname == "sqlalchemy.sql.selectable.Select"
    assert ret.args
    row = get_proper_type(ret.args[0])
    assert isinstance(row, TupleType)
    assert len(row.items) == 5


def test_select_return_type_callback_recovers_tuple_item_types_for_5plus() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    hero_info = make_typeinfo("m.Hero")
    team_info = make_typeinfo("m.Team")
    select_info = make_typeinfo(plugin_mod.SQLMODEL_SELECT_CLS_FULLNAME)

    any_t = AnyType(TypeOfAny.explicit)
    tuple_info = make_typeinfo("builtins.tuple")
    tuple_fallback = Instance(tuple_info, [any_t])
    default_row = TupleType([any_t, any_t, any_t, any_t, any_t], tuple_fallback)
    default_return = Instance(select_info, [default_row])

    int_t = Instance(make_typeinfo("builtins.int"), [])
    str_t = Instance(make_typeinfo("builtins.str"), [])

    inst_attr_int = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute"), [int_t]
    )
    inst_attr_str = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute"), [str_t]
    )

    call = CallExpr(
        NameExpr("select"),
        [
            NameExpr("Hero"),
            NameExpr("Team"),
            NameExpr("HeroId"),
            NameExpr("TeamId"),
            NameExpr("HeroName"),
        ],
        [ARG_POS, ARG_POS, ARG_POS, ARG_POS, ARG_POS],
        [None, None, None, None, None],
    )
    ctx = FunctionContext(
        arg_types=[
            [TypeType(Instance(hero_info, []))],
            [TypeType(Instance(team_info, []))],
            [inst_attr_int],
            [inst_attr_int],
            [inst_attr_str],
        ],
        arg_kinds=[[ARG_POS], [ARG_POS], [ARG_POS], [ARG_POS], [ARG_POS]],
        callee_arg_names=["__ent0", "__ent1", "__ent2", "__ent3", "__ent4"],
        arg_names=[[None], [None], [None], [None], [None]],
        default_return_type=default_return,
        args=[],
        context=call,
        api=api,
    )
    out = p._sqlmodel_select_return_type_callback(ctx)
    proper = get_proper_type(out)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == plugin_mod.SQLMODEL_SELECT_CLS_FULLNAME
    assert proper.args
    row = get_proper_type(proper.args[0])
    assert isinstance(row, TupleType)
    assert len(row.items) == 5
    assert isinstance(get_proper_type(row.items[0]), Instance)
    assert get_proper_type(row.items[0]).type.fullname == "m.Hero"  # type: ignore[union-attr]
    assert isinstance(get_proper_type(row.items[1]), Instance)
    assert get_proper_type(row.items[1]).type.fullname == "m.Team"  # type: ignore[union-attr]
    assert isinstance(get_proper_type(row.items[2]), Instance)
    assert get_proper_type(row.items[2]).type.fullname == "builtins.int"  # type: ignore[union-attr]
    assert isinstance(get_proper_type(row.items[4]), Instance)
    assert get_proper_type(row.items[4]).type.fullname == "builtins.str"  # type: ignore[union-attr]


def test_select_join_isouter_true_makes_joined_entity_optional_in_return_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_typeinfo("m.Hero")
    team_info = make_typeinfo("m.Team")

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(hero_info, []), Instance(team_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    team_expr = NameExpr("Team")
    team_expr.node = team_info

    true_expr = NameExpr("True")

    call = CallExpr(
        NameExpr("join"),
        [team_expr, true_expr],
        [ARG_POS, ARG_NAMED],
        [None, "isouter"],
    )

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.join")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[
            [TypeType(Instance(team_info, []))],
            [Instance(make_typeinfo("builtins.bool"), [])],
        ],
        arg_kinds=[[ARG_POS], [ARG_NAMED]],
        callee_arg_names=["target", "isouter"],
        arg_names=[[None], ["isouter"]],
        default_return_type=select_inst,
        args=[[team_expr], [true_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.sql.selectable.Select"
    assert proper.args
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    assert len(tp2.items) == 2
    assert isinstance(get_proper_type(tp2.items[0]), Instance)
    second = get_proper_type(tp2.items[1])
    assert isinstance(second, UnionType)
    assert any(isinstance(get_proper_type(it), NoneType) for it in second.items)


def test_typeinfo_from_join_target_expr_supports_typed_variables() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    team_info = make_typeinfo("m.Team")
    alias_var = Var("team_alias")
    alias_var.type = TypeType(Instance(team_info, []))

    expr = NameExpr("team_alias")
    expr.node = alias_var

    assert p._typeinfo_from_join_target_expr(expr, api) is team_info


def test_select_join_isouter_true_with_relationship_target_makes_joined_entity_optional_in_return_type() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_sqlmodel_class("m.Hero")
    hero_info.defn.keywords["table"] = NameExpr("True")

    team_info = make_sqlmodel_class("m.Team")
    team_info.defn.keywords["table"] = NameExpr("True")

    team_t = Instance(team_info, [])
    rel_t = UnionType.make_union([team_t, NoneType()])
    hero_info.names["team"] = SymbolTableNode(0, Var("team", rel_t))
    stmt_team = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True
    hero_info.defn.defs.body.append(stmt_team)

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(hero_info, []), Instance(team_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    hero_expr = NameExpr("Hero")
    hero_expr.node = hero_info
    target_expr = MemberExpr(hero_expr, "team")

    true_expr = NameExpr("True")
    call = CallExpr(
        NameExpr("join"),
        [target_expr, true_expr],
        [ARG_POS, ARG_NAMED],
        [None, "isouter"],
    )

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.join")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[
            [AnyType(TypeOfAny.explicit)],
            [Instance(make_typeinfo("builtins.bool"), [])],
        ],
        arg_kinds=[[ARG_POS], [ARG_NAMED]],
        callee_arg_names=["target", "isouter"],
        arg_names=[[None], ["isouter"]],
        default_return_type=select_inst,
        args=[[target_expr], [true_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    second = get_proper_type(tp2.items[1])
    assert isinstance(second, UnionType)
    assert any(isinstance(get_proper_type(it), NoneType) for it in second.items)


def test_select_join_isouter_true_with_relationship_target_list_makes_joined_entity_optional_in_return_type() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_sqlmodel_class("m.Hero")
    hero_info.defn.keywords["table"] = NameExpr("True")

    team_info = make_sqlmodel_class("m.Team")
    team_info.defn.keywords["table"] = NameExpr("True")

    list_info = make_typeinfo("builtins.list")
    rel_t = Instance(list_info, [Instance(hero_info, [])])
    team_info.names["heroes"] = SymbolTableNode(0, Var("heroes", rel_t))
    stmt_heroes = plugin_mod.AssignmentStmt(
        [NameExpr("heroes")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_heroes.new_syntax = True
    team_info.defn.defs.body.append(stmt_heroes)

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(team_info, []), Instance(hero_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    team_expr = NameExpr("Team")
    team_expr.node = team_info
    target_expr = MemberExpr(team_expr, "heroes")

    true_expr = NameExpr("True")
    call = CallExpr(
        NameExpr("join"),
        [target_expr, true_expr],
        [ARG_POS, ARG_NAMED],
        [None, "isouter"],
    )

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.join")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[
            [AnyType(TypeOfAny.explicit)],
            [Instance(make_typeinfo("builtins.bool"), [])],
        ],
        arg_kinds=[[ARG_POS], [ARG_NAMED]],
        callee_arg_names=["target", "isouter"],
        arg_names=[[None], ["isouter"]],
        default_return_type=select_inst,
        args=[[target_expr], [true_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    second = get_proper_type(tp2.items[1])
    assert isinstance(second, UnionType)
    assert any(isinstance(get_proper_type(it), NoneType) for it in second.items)


def test_typeinfo_from_relationship_join_target_returns_none_when_target_is_not_member_expr() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    assert p._typeinfo_from_relationship_join_target(NameExpr("x"), api) is None


def test_typeinfo_from_relationship_join_target_returns_none_when_owner_is_unresolved() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    expr = MemberExpr(NameExpr("Hero"), "team")
    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_typeinfo_from_relationship_join_target_returns_none_when_owner_is_sqlmodel_base() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    sqlmodel_info = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    sqlmodel_expr = NameExpr("SQLModel")
    sqlmodel_expr.node = sqlmodel_info
    expr = MemberExpr(sqlmodel_expr, "team")

    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_typeinfo_from_relationship_join_target_returns_none_when_owner_is_not_sqlmodel() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    owner_info = make_typeinfo("m.NotSQLModel")
    obj_info = make_typeinfo("builtins.object")
    owner_info.mro = [owner_info, obj_info]

    owner_expr = NameExpr("Hero")
    owner_expr.node = owner_info
    expr = MemberExpr(owner_expr, "team")

    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_typeinfo_from_relationship_join_target_returns_none_when_owner_is_not_table_model() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    owner_info = make_sqlmodel_class("m.Hero")
    owner_expr = NameExpr("Hero")
    owner_expr.node = owner_info
    expr = MemberExpr(owner_expr, "team")

    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_typeinfo_from_relationship_join_target_returns_none_when_relationship_missing() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    owner_info = make_sqlmodel_class("m.Hero")
    owner_info.defn.keywords["table"] = NameExpr("True")

    owner_expr = NameExpr("Hero")
    owner_expr.node = owner_info
    expr = MemberExpr(owner_expr, "team")

    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_typeinfo_from_relationship_join_target_returns_none_for_ambiguous_union_relationship_type() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    owner_info = make_sqlmodel_class("m.Hero")
    owner_info.defn.keywords["table"] = NameExpr("True")

    team_info = make_sqlmodel_class("m.Team")
    team_info.defn.keywords["table"] = NameExpr("True")

    other_info = make_sqlmodel_class("m.Other")
    other_info.defn.keywords["table"] = NameExpr("True")

    rel_t = UnionType.make_union([Instance(team_info, []), Instance(other_info, []), NoneType()])
    owner_info.names["team"] = SymbolTableNode(0, Var("team", rel_t))
    stmt_team = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True
    owner_info.defn.defs.body.append(stmt_team)

    owner_expr = NameExpr("Hero")
    owner_expr.node = owner_info
    expr = MemberExpr(owner_expr, "team")

    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_typeinfo_from_relationship_join_target_returns_none_for_non_sqlmodel_entity_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    owner_info = make_sqlmodel_class("m.Hero")
    owner_info.defn.keywords["table"] = NameExpr("True")

    not_model_info = make_typeinfo("m.NotModel")
    obj_info = make_typeinfo("builtins.object")
    not_model_info.mro = [not_model_info, obj_info]

    rel_t = UnionType.make_union([Instance(not_model_info, []), NoneType()])
    owner_info.names["team"] = SymbolTableNode(0, Var("team", rel_t))
    stmt_team = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True
    owner_info.defn.defs.body.append(stmt_team)

    owner_expr = NameExpr("Hero")
    owner_expr.node = owner_info
    expr = MemberExpr(owner_expr, "team")

    assert p._typeinfo_from_relationship_join_target(expr, api) is None


def test_select_join_without_isouter_keeps_return_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_typeinfo("m.Hero")
    team_info = make_typeinfo("m.Team")

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(hero_info, []), Instance(team_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    team_expr = NameExpr("Team")
    team_expr.node = team_info

    call = CallExpr(NameExpr("join"), [team_expr], [ARG_POS], [None])

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.join")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[[TypeType(Instance(team_info, []))]],
        arg_kinds=[[ARG_POS]],
        callee_arg_names=["target"],
        arg_names=[[None]],
        default_return_type=select_inst,
        args=[[team_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    assert isinstance(get_proper_type(tp2.items[1]), Instance)


def test_select_join_isouter_false_keeps_return_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_typeinfo("m.Hero")
    team_info = make_typeinfo("m.Team")

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(hero_info, []), Instance(team_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    team_expr = NameExpr("Team")
    team_expr.node = team_info

    false_expr = NameExpr("False")

    call = CallExpr(
        NameExpr("join"),
        [team_expr, false_expr],
        [ARG_POS, ARG_NAMED],
        [None, "isouter"],
    )

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.join")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[
            [TypeType(Instance(team_info, []))],
            [Instance(make_typeinfo("builtins.bool"), [])],
        ],
        arg_kinds=[[ARG_POS], [ARG_NAMED]],
        callee_arg_names=["target", "isouter"],
        arg_names=[[None], ["isouter"]],
        default_return_type=select_inst,
        args=[[team_expr], [false_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    assert isinstance(get_proper_type(tp2.items[1]), Instance)


def test_select_join_from_isouter_true_makes_target_optional_in_return_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_typeinfo("m.Hero")
    team_info = make_typeinfo("m.Team")

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(hero_info, []), Instance(team_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    hero_expr = NameExpr("Hero")
    hero_expr.node = hero_info

    team_expr = NameExpr("Team")
    team_expr.node = team_info

    true_expr = NameExpr("True")

    call = CallExpr(
        NameExpr("join_from"),
        [hero_expr, team_expr, true_expr],
        [ARG_POS, ARG_POS, ARG_NAMED],
        [None, None, "isouter"],
    )

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.join_from")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[
            [TypeType(Instance(hero_info, []))],
            [TypeType(Instance(team_info, []))],
            [Instance(make_typeinfo("builtins.bool"), [])],
        ],
        arg_kinds=[[ARG_POS], [ARG_POS], [ARG_NAMED]],
        callee_arg_names=["from_", "target", "isouter"],
        arg_names=[[None], [None], ["isouter"]],
        default_return_type=select_inst,
        args=[[hero_expr], [team_expr], [true_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    second = get_proper_type(tp2.items[1])
    assert isinstance(second, UnionType)
    assert any(isinstance(get_proper_type(it), NoneType) for it in second.items)


def test_select_outerjoin_makes_target_optional_in_return_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    hero_info = make_typeinfo("m.Hero")
    team_info = make_typeinfo("m.Team")

    tuple_info = make_typeinfo("builtins.tuple")
    fallback = Instance(tuple_info, [AnyType(TypeOfAny.explicit)])
    tp = TupleType([Instance(hero_info, []), Instance(team_info, [])], fallback)

    select_info = make_typeinfo("sqlalchemy.sql.selectable.Select")
    select_inst = Instance(select_info, [tp])

    team_expr = NameExpr("Team")
    team_expr.node = team_info

    call = CallExpr(NameExpr("outerjoin"), [team_expr], [ARG_POS], [None])

    hook = p.get_method_hook("sqlalchemy.sql.selectable.Select.outerjoin")
    assert hook is not None

    ctx = MethodContext(
        type=select_inst,
        arg_types=[[TypeType(Instance(team_info, []))]],
        arg_kinds=[[ARG_POS]],
        callee_arg_names=["target"],
        arg_names=[[None]],
        default_return_type=select_inst,
        args=[[team_expr]],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    tp2 = get_proper_type(proper.args[0])
    assert isinstance(tp2, TupleType)
    second = get_proper_type(tp2.items[1])
    assert isinstance(second, UnionType)
    assert any(isinstance(get_proper_type(it), NoneType) for it in second.items)


def test_relationship_comparator_hook_coerces_any_return_type_for_sqlmodel_relationship() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    team_info = make_sqlmodel_class("m.Team")
    team_info.defn.keywords["table"] = NameExpr("True")

    hero_info = make_typeinfo("m.Hero")
    list_info = make_typeinfo("builtins.list")
    heroes_t = Instance(list_info, [Instance(hero_info, [])])

    team_info.names["heroes"] = SymbolTableNode(0, Var("heroes", heroes_t))
    stmt_heroes = plugin_mod.AssignmentStmt(
        [NameExpr("heroes")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_heroes.new_syntax = True
    team_info.defn.defs.body.append(stmt_heroes)

    team_expr = NameExpr("Team")
    team_expr.node = team_info
    heroes_expr = MemberExpr(team_expr, "heroes")
    any_member = MemberExpr(heroes_expr, "any")
    call = CallExpr(any_member, [], [], [])

    hook = p.get_method_hook("sqlalchemy.orm.base.SQLORMOperations.any")
    assert hook is not None

    ctx = MethodContext(
        type=AnyType(TypeOfAny.explicit),
        arg_types=[],
        arg_kinds=[],
        callee_arg_names=[],
        arg_names=[],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[],
        context=call,
        api=DummyCheckerAPI(),
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.sql.elements.ColumnElement"
    assert proper.args
    arg0 = get_proper_type(proper.args[0])
    assert isinstance(arg0, Instance)
    assert arg0.type.fullname == "builtins.bool"


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
    model_info.defn.keywords["table"] = NameExpr("True")

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


def test_class_attribute_hook_skips_non_table_models() -> None:
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
    assert hook is None


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


def test_tuple_function_hook_returns_sqlalchemy_tuple_expression() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_hook("sqlmodel.sql.expression.tuple_")
    assert hook is not None

    api = DummyCheckerAPI()
    ctx = FunctionContext(
        arg_types=[],
        arg_kinds=[],
        callee_arg_names=[],
        arg_names=[],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[],
        context=NameExpr("x"),
        api=api,
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.sql.elements.Tuple"


def test_tuple_function_hook_falls_back_to_column_element_when_tuple_missing() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.lookup_fully_qualified = lambda _fullname: None  # type: ignore[method-assign]
    hook = p.get_function_hook("sqlmodel.sql.expression.tuple_")
    assert hook is not None

    class NoTupleAPI(DummyCheckerAPI):
        def named_generic_type(self, name: str, args: list[Type]) -> Instance:
            if name in {"sqlalchemy.sql.elements.Tuple", "sqlalchemy.sql.expression.Tuple"}:
                raise RuntimeError("missing")
            return super().named_generic_type(name, args)

    api = NoTupleAPI()
    ctx = FunctionContext(
        arg_types=[],
        arg_kinds=[],
        callee_arg_names=[],
        arg_names=[],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[],
        context=NameExpr("x"),
        api=api,
    )
    t = hook(ctx)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.sql.elements.ColumnElement"


def test_sqlalchemy_tuple_expr_type_falls_back_to_lookup_when_api_is_none() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    tuple_info = make_typeinfo("builtins.tuple")
    col_info = make_typeinfo("sqlalchemy.sql.elements.ColumnElement")

    lookup_map = {
        "builtins.tuple": tuple_info,
        "sqlalchemy.sql.elements.ColumnElement": col_info,
    }
    p.lookup_fully_qualified = (  # type: ignore[method-assign]
        lambda full: SimpleNamespace(node=lookup_map[full]) if full in lookup_map else None
    )

    t = p._sqlalchemy_tuple_expr_type(api=None)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.sql.elements.ColumnElement"


def test_tuple_signature_hook_updates_return_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook("sqlmodel.sql.expression.tuple_")
    assert hook is not None

    tuple_ret = Instance(make_typeinfo("builtins.tuple"), [AnyType(TypeOfAny.explicit)])
    default_sig = CallableType(
        [], [], [], tuple_ret, Instance(make_typeinfo("builtins.function"), [])
    )
    api = DummyCheckerAPI()
    sig = hook(
        FunctionSigContext(args=[], default_signature=default_sig, context=NameExpr("x"), api=api)
    )
    assert isinstance(sig, CallableType)
    proper = get_proper_type(sig.ret_type)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.sql.elements.Tuple"


def test_tuple_signature_hook_is_idempotent_for_expression_return_types() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook("sqlmodel.sql.expression.tuple_")
    assert hook is not None

    api = DummyCheckerAPI()

    sig_tuple = CallableType(
        [],
        [],
        [],
        Instance(make_typeinfo("sqlalchemy.sql.elements.Tuple"), []),
        Instance(make_typeinfo("builtins.function"), []),
    )
    assert (
        hook(
            FunctionSigContext(args=[], default_signature=sig_tuple, context=NameExpr("x"), api=api)
        )
        is sig_tuple
    )

    sig_col = CallableType(
        [],
        [],
        [],
        Instance(
            make_typeinfo("sqlalchemy.sql.elements.ColumnElement"), [AnyType(TypeOfAny.explicit)]
        ),
        Instance(make_typeinfo("builtins.function"), []),
    )
    assert (
        hook(FunctionSigContext(args=[], default_signature=sig_col, context=NameExpr("x"), api=api))
        is sig_col
    )


def test_getattr_function_hook_types_sqlmodel_table_members() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {"fields": {"id": {}}, "relationships": {}}

    int_t = Instance(make_typeinfo("builtins.int"), [])
    model_info.names["id"] = SymbolTableNode(0, Var("id", int_t))

    hook = p.get_function_hook(plugin_mod.BUILTINS_GETATTR_FULLNAME)
    assert hook is not None

    user_type = TypeType(Instance(model_info, []))
    ctx = FunctionContext(
        arg_types=[[user_type], [Instance(make_typeinfo("builtins.str"), [])]],
        arg_kinds=[[ARG_POS], [ARG_POS]],
        callee_arg_names=["object", "name"],
        arg_names=[[None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [StrExpr("id")]],
        context=NameExpr("x"),
        api=api,
    )
    t = hook(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.attributes.InstrumentedAttribute"
    assert t.args and isinstance(get_proper_type(t.args[0]), Instance)


def test_getattr_function_hook_supports_literal_name_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {
        "fields": {"id": {}, "name": {}},
        "relationships": {},
    }

    int_t = Instance(make_typeinfo("builtins.int"), [])
    str_t = Instance(make_typeinfo("builtins.str"), [])
    model_info.names["id"] = SymbolTableNode(0, Var("id", int_t))
    model_info.names["name"] = SymbolTableNode(0, Var("name", str_t))

    user_type = TypeType(Instance(model_info, []))
    name_type = LiteralType("id", Instance(make_typeinfo("builtins.str"), []))
    ctx = FunctionContext(
        arg_types=[[user_type], [name_type]],
        arg_kinds=[[ARG_POS], [ARG_POS]],
        callee_arg_names=["object", "name"],
        arg_names=[[None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [NameExpr("FIELD")]],
        context=NameExpr("x"),
        api=api,
    )

    t = p._sqlmodel_getattr_return_type_callback(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.attributes.InstrumentedAttribute"


def test_getattr_function_hook_unions_literal_name_types() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {
        "fields": {"id": {}, "name": {}},
        "relationships": {},
    }

    int_t = Instance(make_typeinfo("builtins.int"), [])
    str_t = Instance(make_typeinfo("builtins.str"), [])
    model_info.names["id"] = SymbolTableNode(0, Var("id", int_t))
    model_info.names["name"] = SymbolTableNode(0, Var("name", str_t))

    user_type = TypeType(Instance(model_info, []))
    lit_id = LiteralType("id", Instance(make_typeinfo("builtins.str"), []))
    lit_name = LiteralType("name", Instance(make_typeinfo("builtins.str"), []))
    name_type = UnionType.make_union([lit_id, lit_name])

    ctx = FunctionContext(
        arg_types=[[user_type], [name_type]],
        arg_kinds=[[ARG_POS], [ARG_POS]],
        callee_arg_names=["object", "name"],
        arg_names=[[None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [NameExpr("FIELD")]],
        context=NameExpr("x"),
        api=api,
    )

    t = p._sqlmodel_getattr_return_type_callback(ctx)
    assert isinstance(get_proper_type(t), UnionType)


def test_getattr_function_hook_handles_table_dunder_table() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {"fields": {"id": {}}, "relationships": {}}
    model_info.names["id"] = SymbolTableNode(
        0, Var("id", Instance(make_typeinfo("builtins.int"), []))
    )

    user_type = TypeType(Instance(model_info, []))
    ctx = FunctionContext(
        arg_types=[[user_type], [Instance(make_typeinfo("builtins.str"), [])]],
        arg_kinds=[[ARG_POS], [ARG_POS]],
        callee_arg_names=["object", "name"],
        arg_names=[[None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [StrExpr("__table__")]],
        context=NameExpr("x"),
        api=api,
    )
    t = p._sqlmodel_getattr_return_type_callback(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.sql.schema.Table"


def test_getattr_function_hook_types_expressionish_non_member() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {"fields": {}, "relationships": {}}

    int_t = Instance(make_typeinfo("builtins.int"), [])
    mapped_t = Instance(make_typeinfo("sqlalchemy.orm.base.Mapped"), [int_t])
    model_info.names["_ticketscount"] = SymbolTableNode(0, Var("_ticketscount", mapped_t))

    user_type = TypeType(Instance(model_info, []))
    ctx = FunctionContext(
        arg_types=[[user_type], [Instance(make_typeinfo("builtins.str"), [])]],
        arg_kinds=[[ARG_POS], [ARG_POS]],
        callee_arg_names=["object", "name"],
        arg_names=[[None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [StrExpr("_ticketscount")]],
        context=NameExpr("x"),
        api=api,
    )
    t = p._sqlmodel_getattr_return_type_callback(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.attributes.InstrumentedAttribute"


def test_column_property_return_type_callback_infers_scalar_select() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    int_t = Instance(make_typeinfo("builtins.int"), [])
    scalar_select_t = Instance(make_typeinfo("sqlalchemy.sql.selectable.ScalarSelect"), [int_t])
    ctx = FunctionContext(
        arg_types=[[scalar_select_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column"],
        arg_names=[[None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    t = p._sqlalchemy_column_property_return_type_callback(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.base.Mapped"
    assert t.args and isinstance(get_proper_type(t.args[0]), Instance)


def test_column_property_return_type_callback_infers_column_element() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    int_t = Instance(make_typeinfo("builtins.int"), [])
    col_t = Instance(make_typeinfo("sqlalchemy.sql.elements.ColumnElement"), [int_t])
    ctx = FunctionContext(
        arg_types=[[col_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column"],
        arg_names=[[None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    t = p._sqlalchemy_column_property_return_type_callback(ctx)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.orm.base.Mapped"


def test_column_property_return_type_callback_preserves_precise_mapped_default() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    int_t = Instance(make_typeinfo("builtins.int"), [])
    scalar_select_t = Instance(make_typeinfo("sqlalchemy.sql.selectable.ScalarSelect"), [int_t])
    default_mapped = Instance(make_typeinfo("sqlalchemy.orm.base.Mapped"), [int_t])
    ctx = FunctionContext(
        arg_types=[[scalar_select_t]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column"],
        arg_names=[[None]],
        default_return_type=default_mapped,
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlalchemy_column_property_return_type_callback(ctx) is default_mapped


def test_getattr_return_type_callback_returns_default_for_missing_member() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {"fields": {"id": {}}, "relationships": {}}
    model_info.names["id"] = SymbolTableNode(
        0, Var("id", Instance(make_typeinfo("builtins.int"), []))
    )

    user_type = TypeType(Instance(model_info, []))
    ctx = FunctionContext(
        arg_types=[[user_type], [Instance(make_typeinfo("builtins.str"), [])]],
        arg_kinds=[[ARG_POS], [ARG_POS]],
        callee_arg_names=["object", "name"],
        arg_names=[[None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [StrExpr("missing")]],
        context=NameExpr("x"),
        api=api,
    )
    assert p._sqlmodel_getattr_return_type_callback(ctx).type_of_any == TypeOfAny.explicit  # type: ignore[attr-defined]


def test_is_sqlalchemy_expressionish_covers_non_instance_and_other_fullnames() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    union = UnionType.make_union([Instance(make_typeinfo("builtins.int"), []), NoneType()])
    assert p._is_sqlalchemy_expressionish(union) is False

    inst_attr = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute"), [plugin_mod._plugin_any()]
    )
    assert p._is_sqlalchemy_expressionish(inst_attr) is True

    queryable = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.QueryableAttribute"), [plugin_mod._plugin_any()]
    )
    assert p._is_sqlalchemy_expressionish(queryable) is True

    col = Instance(
        make_typeinfo("sqlalchemy.sql.elements.ColumnElement"), [plugin_mod._plugin_any()]
    )
    assert p._is_sqlalchemy_expressionish(col) is True


def test_sqlalchemy_expr_type_for_class_attr_keeps_existing_expression_types() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    inst_attr = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.InstrumentedAttribute"), [plugin_mod._plugin_any()]
    )
    assert p._sqlalchemy_expr_type_for_class_attr(api, inst_attr) is inst_attr

    queryable = Instance(
        make_typeinfo("sqlalchemy.orm.attributes.QueryableAttribute"), [plugin_mod._plugin_any()]
    )
    assert p._sqlalchemy_expr_type_for_class_attr(api, queryable) is queryable

    col = Instance(
        make_typeinfo("sqlalchemy.sql.elements.ColumnElement"), [plugin_mod._plugin_any()]
    )
    assert p._sqlalchemy_expr_type_for_class_attr(api, col) is col


def test_sqlalchemy_expr_type_for_class_attr_falls_back_to_column_element() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    class NoInstrumentedAPI(DummyCheckerAPI):
        def named_generic_type(self, name: str, args: list[Type]) -> Instance:
            if name in {
                "sqlalchemy.orm.attributes.InstrumentedAttribute",
                "sqlalchemy.orm.InstrumentedAttribute",
            }:
                raise RuntimeError("missing")
            return super().named_generic_type(name, args)

    api = NoInstrumentedAPI()
    int_t = Instance(make_typeinfo("builtins.int"), [])
    t = p._sqlalchemy_expr_type_for_class_attr(api, int_t)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.sql.elements.ColumnElement"


def test_getattr_return_type_callback_unions_default_value() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    model_info = make_sqlmodel_class("m.User")
    model_info.defn.keywords["table"] = NameExpr("True")
    model_info.metadata[plugin_mod.METADATA_KEY] = {"fields": {"id": {}}, "relationships": {}}
    model_info.names["id"] = SymbolTableNode(
        0, Var("id", Instance(make_typeinfo("builtins.int"), []))
    )

    user_type = TypeType(Instance(model_info, []))
    ctx = FunctionContext(
        arg_types=[[user_type], [Instance(make_typeinfo("builtins.str"), [])], [NoneType()]],
        arg_kinds=[[ARG_POS], [ARG_POS], [ARG_OPT]],
        callee_arg_names=["object", "name", "default"],
        arg_names=[[None], [None], [None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("User")], [StrExpr("id")], [NameExpr("None")]],
        context=NameExpr("x"),
        api=api,
    )
    t = p._sqlmodel_getattr_return_type_callback(ctx)
    assert isinstance(get_proper_type(t), UnionType)


def test_column_property_return_type_callback_guard_and_value_type_none_paths() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    # No args -> no-op.
    ctx = FunctionContext(
        arg_types=[],
        arg_kinds=[],
        callee_arg_names=[],
        arg_names=[],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[],
        context=NameExpr("x"),
        api=api,
    )
    assert isinstance(p._sqlalchemy_column_property_return_type_callback(ctx), AnyType)

    # Unknown arg type -> value_type stays None -> no-op.
    ctx = FunctionContext(
        arg_types=[[Instance(make_typeinfo("builtins.int"), [])]],
        arg_kinds=[[ARG_OPT]],
        callee_arg_names=["column"],
        arg_names=[[None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("x")]],
        context=NameExpr("x"),
        api=api,
    )
    assert isinstance(p._sqlalchemy_column_property_return_type_callback(ctx), AnyType)


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


def test_get_method_signature_hook_selects_model_construct_and_exec() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    assert p.get_method_signature_hook("m.User.other") is None

    hook_exec = p.get_method_signature_hook(plugin_mod.SQLMODEL_SESSION_EXEC_FULLNAME)
    assert hook_exec is not None
    assert callable(hook_exec)

    hook_async_exec = p.get_method_signature_hook(plugin_mod.SQLMODEL_ASYNC_SESSION_EXEC_FULLNAME)
    assert hook_async_exec is not None
    assert callable(hook_async_exec)

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


def test_collect_member_from_stmt_prefers_metadata_for_default_and_aliases() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    info = make_sqlmodel_class("m.User")

    int_t = Instance(make_typeinfo("builtins.int"), [])
    info.names["x"] = SymbolTableNode(0, Var("x", int_t))

    # No RHS (TempNode) means we can't infer Field(...) metadata from stmt.rvalue.
    stmt = plugin_mod.AssignmentStmt(
        [NameExpr("x")], TempNode(AnyType(TypeOfAny.special_form), no_rhs=True)
    )
    stmt.new_syntax = True

    info.metadata[plugin_mod.METADATA_KEY] = {
        "fields": {"x": {"has_default": True, "aliases": ["alias_x"]}}
    }

    member = p._collect_member_from_stmt(stmt, defining_info=info, current_info=info, api=api)
    assert isinstance(member, plugin_mod._CollectedField)
    assert member.has_default is True
    assert member.aliases == ("alias_x",)


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


def test_constructor_signature_prefers_plugin_init() -> None:
    model_info = make_sqlmodel_class("m.User")

    function_type = Instance(make_typeinfo("builtins.function"), [])
    self_t = Instance(model_info, [])
    int_t = Instance(make_typeinfo("builtins.int"), [])

    # Simulate the plugin-generated `__init__` signature that mypy stores on the class.
    init_sig = CallableType(
        [self_t, int_t],
        [ARG_POS, ARG_NAMED],
        ["__sqlmodel_self__", "x"],
        NoneType(),
        function_type,
    )
    init_var = Var("__init__", init_sig)
    init_sym = SymbolTableNode(0, init_var)
    init_sym.plugin_generated = True
    model_info.names["__init__"] = init_sym

    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.lookup_fully_qualified = lambda full: SimpleNamespace(  # type: ignore[method-assign]
        node={"m.User": model_info}.get(full)
    )

    default_sig = CallableType([], [], [], Instance(model_info, []), function_type)
    hook = p.get_function_signature_hook("m.User")
    assert hook is not None
    sig = hook(
        FunctionSigContext(
            args=[], default_signature=default_sig, context=NameExpr("x"), api=DummyCheckerAPI()
        )
    )
    assert isinstance(sig, CallableType)
    assert sig.arg_names == ["x"]
    assert sig.arg_kinds == [ARG_NAMED]
    assert sig.ret_type == Instance(model_info, [])


def test_declares_sqlmodel_member_uses_metadata() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    info = make_sqlmodel_class("m.User")

    info.metadata[plugin_mod.METADATA_KEY] = {
        "fields": {"id": {}},
        "relationships": {"team": {}},
    }

    assert p._declares_sqlmodel_member(info, "id") is True
    assert p._declares_sqlmodel_member(info, "team") is True
    assert p._declares_sqlmodel_member(info, "missing") is False


def test_plugins_from_config_file_ini(tmp_path: Path) -> None:
    path = tmp_path / "mypy.ini"
    path.write_text(
        """
[mypy]
plugins = pydantic.mypy, sqlmodel_mypy.plugin
""".lstrip()
    )
    assert plugin_mod._plugins_from_config_file(str(path)) == [
        "pydantic.mypy",
        "sqlmodel_mypy.plugin",
    ]


def test_plugins_from_config_file_toml_list(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.mypy]
plugins = ["pydantic.mypy", "sqlmodel_mypy.plugin"]
""".lstrip()
    )
    assert plugin_mod._plugins_from_config_file(str(path)) == [
        "pydantic.mypy",
        "sqlmodel_mypy.plugin",
    ]


def test_plugins_from_config_file_toml_string(tmp_path: Path) -> None:
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.mypy]
plugins = "pydantic.mypy, sqlmodel_mypy.plugin"
""".lstrip()
    )
    assert plugin_mod._plugins_from_config_file(str(path)) == [
        "pydantic.mypy",
        "sqlmodel_mypy.plugin",
    ]


def test_named_type_helpers() -> None:
    int_t = Instance(make_typeinfo("builtins.int"), [])

    class NamedTypeAPI:
        def named_type(self, fullname: str, args: list[Type] | None = None) -> Instance:  # noqa: D401
            return Instance(make_typeinfo(fullname), args or [])

    class NamedGenericTypeAPI:
        def named_generic_type(self, fullname: str, args: list[Type]) -> Instance:  # noqa: D401
            return Instance(make_typeinfo(fullname), args)

    t = plugin_mod._named_type_or_none(NamedTypeAPI(), "builtins.int")
    assert isinstance(t, Instance)
    assert t.type.fullname == "builtins.int"

    t = plugin_mod._named_type_or_none(NamedGenericTypeAPI(), "builtins.int")
    assert isinstance(t, Instance)
    assert t.type.fullname == "builtins.int"

    t = plugin_mod._named_generic_type_or_none(NamedTypeAPI(), "builtins.list", [int_t])
    assert isinstance(t, Instance)
    assert t.type.fullname == "builtins.list"
    assert t.args == (int_t,)

    t = plugin_mod._named_generic_type_or_none(NamedGenericTypeAPI(), "builtins.list", [int_t])
    assert isinstance(t, Instance)
    assert t.type.fullname == "builtins.list"
    assert t.args == (int_t,)


def test_get_function_hook_returns_none_for_other_functions() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    assert p.get_function_hook("sqlmodel.sql.expression.not_col") is None


def test_metaclass_callback_handles_missing_declared_metaclass() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    ctx = SimpleNamespace(cls=SimpleNamespace(info=SimpleNamespace(declared_metaclass=None)))
    p._sqlmodel_metaclass_callback(ctx)  # type: ignore[arg-type]


def test_field_signature_hook_widens_validation_alias_and_sa_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    type_engine_info = make_typeinfo("sqlalchemy.sql.type_api.TypeEngine")
    alias_path_info = make_typeinfo("pydantic.aliases.AliasPath")
    alias_choices_info = make_typeinfo("pydantic.aliases.AliasChoices")
    lookup_map = {
        type_engine_info.fullname: type_engine_info,
        alias_path_info.fullname: alias_path_info,
        alias_choices_info.fullname: alias_choices_info,
    }
    p.lookup_fully_qualified = (  # type: ignore[method-assign]
        lambda full: SimpleNamespace(node=lookup_map[full]) if full in lookup_map else None
    )

    old_sa_type = Instance(make_typeinfo("builtins.type"), [])
    old_validation_alias = UnionType.make_union(
        [Instance(make_typeinfo("builtins.str"), []), NoneType()]
    )
    default_sig = CallableType(
        [old_sa_type, old_validation_alias],
        [ARG_OPT, ARG_OPT],
        ["sa_type", "validation_alias"],
        AnyType(TypeOfAny.explicit),
        Instance(make_typeinfo("builtins.function"), []),
    )
    out = p._sqlmodel_field_signature_callback(
        FunctionSigContext(
            args=[], default_signature=default_sig, context=NameExpr("x"), api=DummyCheckerAPI()
        )
    )
    assert isinstance(out, CallableType)
    assert out is not default_sig

    sa_type_arg = get_proper_type(out.arg_types[0])
    assert isinstance(sa_type_arg, UnionType)
    assert any(
        isinstance(get_proper_type(it), Instance)
        and get_proper_type(it).type.fullname == "sqlalchemy.sql.type_api.TypeEngine"  # type: ignore[union-attr]
        for it in sa_type_arg.items
    )

    validation_alias_arg = get_proper_type(out.arg_types[1])
    assert isinstance(validation_alias_arg, UnionType)
    assert any(
        isinstance(get_proper_type(it), Instance)
        and get_proper_type(it).type.fullname == "pydantic.aliases.AliasPath"  # type: ignore[union-attr]
        for it in validation_alias_arg.items
    )
    assert any(
        isinstance(get_proper_type(it), Instance)
        and get_proper_type(it).type.fullname == "pydantic.aliases.AliasChoices"  # type: ignore[union-attr]
        for it in validation_alias_arg.items
    )


def test_session_execute_method_hook_types_select_results_when_enabled() -> None:
    options = Options()
    options.config_file = None
    p = plugin_mod.SQLModelMypyPlugin(options)
    p.plugin_config.typed_execute = True

    api = DummyCheckerAPI()
    hero_info = make_typeinfo("m.Hero")

    # `select(Hero)` is typically a SelectOfScalar[Hero].
    select_scalar_info = make_typeinfo(plugin_mod.SQLMODEL_SELECT_OF_SCALAR_CLS_FULLNAME)
    stmt = Instance(select_scalar_info, [Instance(hero_info, [])])

    call = CallExpr(NameExpr("execute"), [NameExpr("stmt")], [ARG_POS], [None])
    ctx = MethodContext(
        type=Instance(make_typeinfo("sqlmodel.orm.session.Session"), []),
        arg_types=[[stmt]],
        arg_kinds=[[ARG_POS]],
        callee_arg_names=["statement"],
        arg_names=[[None]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[NameExpr("stmt")]],
        context=call,
        api=api,
    )
    t = p._sqlmodel_session_execute_return_type_callback(ctx, is_async=False)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.engine.Result"
    assert proper.args
    row = get_proper_type(proper.args[0])
    assert isinstance(row, TupleType)
    assert len(row.items) == 1

    # Async wrapper: preserve Coroutine[T] and only swap the inner result type.
    default_coro = Instance(
        make_typeinfo("typing.Coroutine"),
        [AnyType(TypeOfAny.explicit), AnyType(TypeOfAny.explicit), AnyType(TypeOfAny.explicit)],
    )
    ctx_async = MethodContext(
        type=Instance(make_typeinfo("sqlmodel.ext.asyncio.session.AsyncSession"), []),
        arg_types=[[stmt]],
        arg_kinds=[[ARG_POS]],
        callee_arg_names=["statement"],
        arg_names=[[None]],
        default_return_type=default_coro,
        args=[[NameExpr("stmt")]],
        context=call,
        api=api,
    )
    t2 = p._sqlmodel_session_execute_return_type_callback(ctx_async, is_async=True)
    proper2 = get_proper_type(t2)
    assert isinstance(proper2, Instance)
    assert proper2.type.fullname == "typing.Coroutine"
    assert len(proper2.args) == 3
    inner = get_proper_type(proper2.args[2])
    assert isinstance(inner, Instance)
    assert inner.type.fullname == "sqlalchemy.engine.Result"


def test_call_get_positional_returns_none_for_out_of_range() -> None:
    call = CallExpr(NameExpr("f"), [NameExpr("x")], [ARG_POS], [None])
    assert plugin_mod._call_get_positional(call, 1) is None


def test_typeinfo_from_ref_expr_none_and_non_typeinfo() -> None:
    assert plugin_mod._typeinfo_from_ref_expr(None) is None

    expr = NameExpr("x")
    expr.node = Var("x")
    assert plugin_mod._typeinfo_from_ref_expr(expr) is None


def test_named_type_helpers_return_none_when_api_missing_methods() -> None:
    int_t = Instance(make_typeinfo("builtins.int"), [])

    class EmptyAPI:
        pass

    assert plugin_mod._named_type_or_none(EmptyAPI(), "builtins.int") is None
    assert plugin_mod._named_generic_type_or_none(EmptyAPI(), "builtins.list", [int_t]) is None


def test_plugins_from_config_file_returns_none_for_invalid_shapes(tmp_path: Path) -> None:
    # TOML with unsupported plugins shape.
    path = tmp_path / "pyproject.toml"
    path.write_text(
        """
[tool.mypy]
plugins = 1
""".lstrip()
    )
    assert plugin_mod._plugins_from_config_file(str(path)) is None

    # INI without [mypy] section.
    ini1 = tmp_path / "no-mypy.ini"
    ini1.write_text(
        """
[sqlmodel-mypy]
init_typed = true
""".lstrip()
    )
    assert plugin_mod._plugins_from_config_file(str(ini1)) is None

    # INI with [mypy] but without plugins option.
    ini2 = tmp_path / "no-plugins.ini"
    ini2.write_text(
        """
[mypy]
strict = true
""".lstrip()
    )
    assert plugin_mod._plugins_from_config_file(str(ini2)) is None


def test_warm_sqlalchemy_typing_is_idempotent_and_skips_exceptions() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    class RaisingNamedTypeAPI:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def named_type(self, fullname: str, args: list[Type] | None = None) -> Instance:
            self.calls.append(fullname)
            if fullname == "builtins.bool":
                return Instance(make_typeinfo("builtins.bool"), [])
            raise RuntimeError("boom")

    api = RaisingNamedTypeAPI()
    p._warm_sqlalchemy_typing(api)  # should swallow exceptions in the loop
    assert p._warmed_sqlalchemy_typing is True

    # Second call is a no-op.
    before = len(api.calls)
    p._warm_sqlalchemy_typing(api)
    assert len(api.calls) == before


def test_declares_sqlmodel_member_accepts_relationship_list_metadata() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    info = make_sqlmodel_class("m.User")
    info.metadata[plugin_mod.METADATA_KEY] = {"fields": {}, "relationships": ["team"]}
    assert p._declares_sqlmodel_member(info, "team") is True


def test_get_method_hook_returns_execute_hook_when_enabled() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.plugin_config.typed_execute = True
    assert p.get_method_hook(plugin_mod.SQLMODEL_SESSION_EXECUTE_FULLNAME) is not None
    assert p.get_method_hook(plugin_mod.SQLMODEL_ASYNC_SESSION_EXECUTE_FULLNAME) is not None


def test_get_function_hook_returns_select_hook() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    assert p.get_function_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME) is not None


def test_session_execute_callback_supports_statement_keyword_argument() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.plugin_config.typed_execute = True
    api = DummyCheckerAPI()

    hero_info = make_typeinfo("m.Hero")
    select_scalar_info = make_typeinfo(plugin_mod.SQLMODEL_SELECT_OF_SCALAR_CLS_FULLNAME)
    stmt = Instance(select_scalar_info, [Instance(hero_info, [])])

    stmt_expr = NameExpr("stmt")
    call = CallExpr(NameExpr("execute"), [stmt_expr], [ARG_NAMED], ["statement"])
    ctx = MethodContext(
        type=Instance(make_typeinfo("sqlmodel.orm.session.Session"), []),
        arg_types=[[stmt]],
        arg_kinds=[[ARG_NAMED]],
        callee_arg_names=["statement"],
        arg_names=[["statement"]],
        default_return_type=AnyType(TypeOfAny.explicit),
        args=[[stmt_expr]],
        context=call,
        api=api,
    )
    t = p._sqlmodel_session_execute_return_type_callback(ctx, is_async=False)
    proper = get_proper_type(t)
    assert isinstance(proper, Instance)
    assert proper.type.fullname == "sqlalchemy.engine.Result"


def test_sqlalchemy_result_type_falls_back_to_lookup_without_named_generic() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    result_info = make_typeinfo("sqlalchemy.engine.Result")
    p.lookup_fully_qualified = (  # type: ignore[method-assign]
        lambda full: SimpleNamespace(node=result_info) if full == result_info.fullname else None
    )

    class NoGenericAPI:
        def named_type(self, fullname: str, args: list[Type] | None = None) -> Instance:  # noqa: D401
            raise RuntimeError("no named_type support")

    api = NoGenericAPI()
    any_t = AnyType(TypeOfAny.explicit)
    row = TupleType(
        [Instance(make_typeinfo("builtins.int"), [])],
        Instance(make_typeinfo("builtins.tuple"), [any_t]),
    )
    t = p._sqlalchemy_result_type(api, row)
    assert isinstance(t, Instance)
    assert t.type.fullname == "sqlalchemy.engine.Result"


def test_row_tuple_type_from_typed_statement_returns_none_for_non_select() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    stmt = Instance(
        make_typeinfo("sqlalchemy.sql.elements.ColumnElement"), [AnyType(TypeOfAny.explicit)]
    )
    assert p._row_tuple_type_from_typed_statement(stmt, api) is None


def test_collect_members_for_signature_skips_none_members_and_collects_inherited_relationships() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    sqlmodel_info = make_typeinfo(plugin_mod.SQLMODEL_BASEMODEL_FULLNAME)
    obj_info = make_typeinfo("builtins.object")

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_info, obj_info]

    user_info = make_typeinfo("m.User")
    user_info.mro = [user_info, base_info, sqlmodel_info, obj_info]

    # Inherited statement that's ignored -> `continue` branch.
    stmt_base_cfg = plugin_mod.AssignmentStmt([NameExpr("model_config")], NameExpr("x"))
    stmt_base_cfg.new_syntax = True
    base_info.defn.defs.body.append(stmt_base_cfg)

    # Inherited relationship with a known type so map_type_from_supertype runs.
    team_info = make_typeinfo("m.Team")
    team_t = Instance(team_info, [])
    rel_t = UnionType.make_union([team_t, NoneType()])
    base_info.names["team"] = SymbolTableNode(0, Var("team", rel_t))
    stmt_base_rel = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_base_rel.new_syntax = True
    base_info.defn.defs.body.append(stmt_base_rel)

    # Current-class ignored statement -> `continue` branch.
    stmt_user_cfg = plugin_mod.AssignmentStmt([NameExpr("model_config")], NameExpr("x"))
    stmt_user_cfg.new_syntax = True
    user_info.defn.defs.body.append(stmt_user_cfg)

    fields, rels = p._collect_members_for_signature(user_info, api)
    assert fields == []
    assert [r.name for r in rels] == ["team"]


def test_constructor_signature_callback_uses_decorator_wrapped_init_type() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()

    info = make_sqlmodel_class("m.User")
    init_sig = CallableType(
        [AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.int"), [])],
        [ARG_POS, ARG_NAMED],
        [None, "x"],
        NoneType(),
        Instance(make_typeinfo("builtins.function"), []),
    )
    init_func = FuncDef("__init__", arguments=[], body=Block([]))
    init_func.type = init_sig
    init_var = Var("__init__")
    init_node = plugin_mod.Decorator(init_func, decorators=[], var=init_var)

    sym = SymbolTableNode(0, init_node)
    sym.plugin_generated = True  # type: ignore[attr-defined]
    info.names["__init__"] = sym

    default_sig = CallableType(
        [], [], [], Instance(info, []), Instance(make_typeinfo("builtins.function"), [])
    )
    ctx = FunctionSigContext(args=[], default_signature=default_sig, context=NameExpr("x"), api=api)
    out = p._sqlmodel_constructor_signature_callback(ctx, info)
    assert isinstance(out, CallableType)
    # Drop `self` from the init signature.
    assert out.arg_names == ["x"]


def test_constructor_signature_callback_dedupes_field_aliases_and_types_relationship_kwargs() -> (
    None
):
    p = plugin_mod.SQLModelMypyPlugin(Options())
    p.plugin_config.init_typed = True
    api = DummyCheckerAPI()

    info = make_sqlmodel_class("m.User")
    info.defn.keywords["table"] = NameExpr("True")
    info.metadata[plugin_mod.METADATA_KEY] = {
        "fields": {
            "a": {"aliases": ["x"], "has_default": False},
            "b": {"aliases": ["x"], "has_default": False},
        },
        "relationships": {},
    }

    info.names["a"] = SymbolTableNode(0, Var("a", Instance(make_typeinfo("builtins.int"), [])))
    info.names["b"] = SymbolTableNode(0, Var("b", Instance(make_typeinfo("builtins.str"), [])))

    stmt_a = plugin_mod.AssignmentStmt(
        [NameExpr("a")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt_a.new_syntax = True
    info.defn.defs.body.append(stmt_a)

    stmt_b = plugin_mod.AssignmentStmt(
        [NameExpr("b")], make_call(plugin_mod.SQLMODEL_FIELD_FULLNAME)
    )
    stmt_b.new_syntax = True
    info.defn.defs.body.append(stmt_b)

    # Relationship with unknown type -> should be `Any` in typed signatures.
    stmt_rel = plugin_mod.AssignmentStmt(
        [NameExpr("team")], make_call(plugin_mod.SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_rel.new_syntax = True
    info.defn.defs.body.append(stmt_rel)

    default_sig = CallableType(
        [], [], [], Instance(info, []), Instance(make_typeinfo("builtins.function"), [])
    )
    ctx = FunctionSigContext(args=[], default_signature=default_sig, context=NameExpr("x"), api=api)
    out = p._sqlmodel_constructor_signature_callback(ctx, info)
    assert isinstance(out, CallableType)

    # Alias `x` should appear only once, even if multiple fields share it.
    assert out.arg_names.count("x") == 1
    assert "team" in out.arg_names


def test_get_method_hook_registers_outerjoin_from() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    assert p.get_method_hook(plugin_mod.SQLALCHEMY_SELECT_OUTERJOIN_FROM_FULLNAME) is not None
    assert p.get_method_hook(plugin_mod.SQLMODEL_SELECT_OUTERJOIN_FROM_FULLNAME) is not None


def test_select_signature_hook_returns_default_for_non_call_context() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME)
    assert hook is not None

    default_sig = CallableType(
        [], [], [], AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.function"), [])
    )
    api = DummyCheckerAPI()
    assert (
        hook(
            FunctionSigContext(
                args=[], default_signature=default_sig, context=NameExpr("x"), api=api
            )
        )
        is default_sig
    )


def test_select_signature_hook_returns_default_for_star_args() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME)
    assert hook is not None

    default_sig = CallableType(
        [], [], [], AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.function"), [])
    )
    call = CallExpr(NameExpr("select"), [NameExpr("xs")], [ARG_STAR], [None])
    api = DummyCheckerAPI()
    assert (
        hook(FunctionSigContext(args=[], default_signature=default_sig, context=call, api=api))
        is default_sig
    )


def test_select_signature_hook_returns_default_for_named_args() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME)
    assert hook is not None

    default_sig = CallableType(
        [], [], [], AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.function"), [])
    )
    call = CallExpr(NameExpr("select"), [NameExpr("x")], [ARG_NAMED], ["entity_0"])
    api = DummyCheckerAPI()
    assert (
        hook(FunctionSigContext(args=[], default_signature=default_sig, context=call, api=api))
        is default_sig
    )


def test_select_signature_hook_returns_default_for_non_positional_kinds() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME)
    assert hook is not None

    default_sig = CallableType(
        [], [], [], AnyType(TypeOfAny.explicit), Instance(make_typeinfo("builtins.function"), [])
    )
    call = CallExpr(NameExpr("select"), [NameExpr("x")], [ARG_OPT], [None])
    api = DummyCheckerAPI()
    assert (
        hook(FunctionSigContext(args=[], default_signature=default_sig, context=call, api=api))
        is default_sig
    )


def test_field_signature_hook_handles_missing_validation_alias_param() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())

    default_sig = CallableType(
        [AnyType(TypeOfAny.explicit)],
        [ARG_OPT],
        ["sa_type"],
        AnyType(TypeOfAny.explicit),
        Instance(make_typeinfo("builtins.function"), []),
    )
    out = p._sqlmodel_field_signature_callback(
        FunctionSigContext(
            args=[], default_signature=default_sig, context=NameExpr("x"), api=DummyCheckerAPI()
        )
    )
    assert out is default_sig


def test_field_signature_hook_returns_default_for_overloaded_signature() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    sig_item = CallableType(
        [],
        [],
        [],
        AnyType(TypeOfAny.explicit),
        Instance(make_typeinfo("builtins.function"), []),
    )
    default_sig = Overloaded([sig_item])
    out = p._sqlmodel_field_signature_callback(
        FunctionSigContext(
            args=[], default_signature=default_sig, context=NameExpr("x"), api=DummyCheckerAPI()
        )
    )
    assert out is default_sig


def test_select_signature_hook_is_idempotent_for_star_overload() -> None:
    p = plugin_mod.SQLModelMypyPlugin(Options())
    hook = p.get_function_signature_hook(plugin_mod.SQLMODEL_SELECT_GEN_FULLNAME)
    assert hook is not None

    default_sig = CallableType(
        [AnyType(TypeOfAny.explicit), AnyType(TypeOfAny.explicit)],
        [ARG_POS, ARG_STAR],
        [None, None],
        AnyType(TypeOfAny.explicit),
        Instance(make_typeinfo("builtins.function"), []),
    )
    call = CallExpr(
        NameExpr("select"),
        [NameExpr("a"), NameExpr("b"), NameExpr("c"), NameExpr("d"), NameExpr("e")],
        [ARG_POS, ARG_POS, ARG_POS, ARG_POS, ARG_POS],
        [None, None, None, None, None],
    )
    api = DummyCheckerAPI()
    assert (
        hook(FunctionSigContext(args=[], default_signature=default_sig, context=call, api=api))
        is default_sig
    )


def test_sqlalchemy_typing_helpers_fallback_to_any(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cover SQLAlchemy typing helper fallbacks (best-effort robustness)."""
    p = plugin_mod.SQLModelMypyPlugin(Options())
    any_t = AnyType(TypeOfAny.explicit)

    def _no_lookup(_plugin: object, _fullname: str) -> None:
        return None

    monkeypatch.setattr(plugin_mod, "_lookup_typeinfo", _no_lookup)

    assert isinstance(get_proper_type(p._sqlalchemy_table_type(None)), AnyType)
    assert isinstance(get_proper_type(p._sqlalchemy_mapper_type(None, any_t)), AnyType)
    assert isinstance(get_proper_type(p._sqlalchemy_instance_state_type(None, any_t)), AnyType)


def test_sqlalchemy_instance_state_type_uses_named_generic_type() -> None:
    """Cover the fast-path branch for InstanceState typing."""
    p = plugin_mod.SQLModelMypyPlugin(Options())
    api = DummyCheckerAPI()
    any_t = AnyType(TypeOfAny.explicit)

    out = p._sqlalchemy_instance_state_type(api, any_t)
    assert isinstance(get_proper_type(out), Instance)
