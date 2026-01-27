from __future__ import annotations

from types import SimpleNamespace

from mypy.nodes import (
    ARG_NAMED,
    ARG_POS,
    INVARIANT,
    MDEF,
    Argument,
    AssignmentStmt,
    Block,
    CallExpr,
    ClassDef,
    Decorator,
    EllipsisExpr,
    FuncDef,
    IfStmt,
    NameExpr,
    PlaceholderNode,
    SymbolTable,
    SymbolTableNode,
    TempNode,
    TypeAlias,
    TypeInfo,
    Var,
)
from mypy.types import AnyType, Instance, NoneType, TypeOfAny, TypeVarId, TypeVarType

from sqlmodel_mypy.transform import (
    ERROR_FIELD,
    METADATA_KEY,
    SQLMODEL_FIELD_FULLNAME,
    SQLMODEL_RELATIONSHIP_FULLNAME,
    ForceInvariantTypeVars,
    SQLModelField,
    SQLModelTransformer,
    _callee_fullname,
    add_method,
)


class DummyAPI:
    """Minimal SemanticAnalyzerPluginInterface stub for unit tests."""

    def __init__(self) -> None:
        self.options = SimpleNamespace(strict_optional=True)
        self.failed: list[tuple[str, object | None]] = []
        self.deferred = False
        self.dependencies: list[str] = []

        self._builtins: dict[str, TypeInfo] = {}
        for name in ("function", "str", "set"):
            cls = ClassDef(name, Block([]))
            info = TypeInfo(SymbolTable(), cls, "builtins")
            cls.info = info
            info._fullname = f"builtins.{name}"
            self._builtins[info._fullname] = info

    def named_type(self, fullname: str, args: list[object] | None = None) -> Instance:
        info = self._builtins.get(fullname)
        if info is None:
            cls = ClassDef(fullname.split(".")[-1], Block([]))
            info = TypeInfo(SymbolTable(), cls, fullname.rsplit(".", 1)[0])
            cls.info = info
            info._fullname = fullname
            self._builtins[fullname] = info
        return Instance(info, list(args or []))

    def fail(self, msg: str, ctx: object, *, code: object | None = None) -> None:
        self.failed.append((msg, code))

    def defer(self) -> None:
        self.deferred = True

    def add_plugin_dependency(self, trigger: str) -> None:
        self.dependencies.append(trigger)


def make_typeinfo(fullname: str) -> TypeInfo:
    module, name = fullname.rsplit(".", 1)
    cls = ClassDef(name, Block([]))
    info = TypeInfo(SymbolTable(), cls, module)
    cls.info = info
    info._fullname = fullname

    obj_cls = ClassDef("object", Block([]))
    obj_info = TypeInfo(SymbolTable(), obj_cls, "builtins")
    obj_cls.info = obj_info
    obj_info._fullname = "builtins.object"
    info.mro = [info, obj_info]

    return info


def make_sqlmodel_class(fullname: str = "m.User") -> ClassDef:
    info = make_typeinfo(fullname)
    cls = info.defn
    assert isinstance(cls, ClassDef)
    cls.info = info
    return cls


def make_field_call(
    *, fullname: str, args: list[object] | None = None, arg_names: list[str | None] | None = None
) -> CallExpr:
    callee = NameExpr("callable")
    callee.fullname = fullname
    args = list(args or [])
    arg_names = list(arg_names or [])
    return CallExpr(callee, args, [ARG_POS] * len(args), arg_names)


def test_callee_fullname() -> None:
    call = make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)
    assert _callee_fullname(call) == SQLMODEL_FIELD_FULLNAME


def test_get_has_default_covers_common_cases() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=False, init_forbid_extra=False, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    # TempNode -> required
    stmt = AssignmentStmt([NameExpr("x")], TempNode(AnyType(TypeOfAny.explicit)))
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is False

    # x: int = Field() -> required
    stmt = AssignmentStmt([NameExpr("x")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is False

    # x: int = Field(default=...) -> required
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[EllipsisExpr()],
            arg_names=["default"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is False

    # x: int = Field(default=None) -> optional
    none_expr = NameExpr("None")
    none_expr.fullname = "builtins.None"
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[none_expr],
            arg_names=["default"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(default_factory=None) -> required (no factory)
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[none_expr],
            arg_names=["default_factory"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is False

    # x: int = Field(default_factory=somefunc) -> optional
    fn = NameExpr("factory")
    fn.fullname = "m.factory"
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[fn],
            arg_names=["default_factory"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = ... -> required
    stmt = AssignmentStmt([NameExpr("x")], EllipsisExpr())
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is False


def test_collect_field_from_stmt_untyped_warns() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=False, init_forbid_extra=False, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    # x = Field(...)
    stmt = AssignmentStmt([NameExpr("x")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = False
    assert transformer.collect_field_from_stmt(stmt) is None
    assert ("Untyped fields disallowed", ERROR_FIELD) in api.failed

    # Relationship() is also disallowed when untyped
    api.failed.clear()
    stmt = AssignmentStmt(
        [NameExpr("rel")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt.new_syntax = False
    assert transformer.collect_field_from_stmt(stmt) is None
    assert ("Untyped fields disallowed", ERROR_FIELD) in api.failed


def test_collect_field_from_stmt_typed_collects_and_skips() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=False, init_forbid_extra=False, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    # model_config and private names are ignored
    stmt = AssignmentStmt(
        [NameExpr("model_config")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)
    )
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None

    stmt = AssignmentStmt([NameExpr("_hidden")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None

    # Relationships are ignored
    stmt = AssignmentStmt(
        [NameExpr("team")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None

    # Non-NameExpr lvalues are ignored
    stmt = AssignmentStmt(
        [make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)],
        make_field_call(fullname=SQLMODEL_FIELD_FULLNAME),
    )
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None

    # A proper typed field is collected
    any_t = AnyType(TypeOfAny.explicit)
    var = Var("x", any_t)
    cls.info.names["x"] = SymbolTableNode(MDEF, var)
    stmt = AssignmentStmt([NameExpr("x")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    field = transformer.collect_field_from_stmt(stmt)
    assert field is not None
    assert field.name == "x"
    assert field.type is any_t

    # ClassVar fields are ignored
    var2 = Var("y", any_t)
    var2.is_classvar = True
    cls.info.names["y"] = SymbolTableNode(MDEF, var2)
    stmt = AssignmentStmt([NameExpr("y")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None

    # PlaceholderNode and TypeAlias nodes are ignored
    cls.info.names["p"] = SymbolTableNode(MDEF, PlaceholderNode("m.p", NameExpr("p"), 1))
    stmt = AssignmentStmt([NameExpr("p")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None

    alias = TypeAlias(AnyType(TypeOfAny.explicit), "m.alias", "m", 1, 0)
    cls.info.names["alias"] = SymbolTableNode(MDEF, alias)
    stmt = AssignmentStmt([NameExpr("alias")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_field_from_stmt(stmt) is None


def test_add_method_generates_and_replaces_methods() -> None:
    api = DummyAPI()
    cls = make_sqlmodel_class()

    any_t = AnyType(TypeOfAny.explicit)
    arg = Argument(Var("x", any_t), any_t, None, ARG_NAMED)

    # First generation
    add_method(api, cls, "foo", args=[arg], return_type=NoneType())
    assert "foo" in cls.info.names
    assert cls.info.names["foo"].plugin_generated is True

    # Re-entry: previously generated method gets removed, old symbol kept under a new name
    add_method(api, cls, "foo", args=[arg], return_type=NoneType())
    assert "foo" in cls.info.names
    assert any(name.startswith("foo") and name != "foo" for name in cls.info.names)

    # Existing (non-plugin) definition is kept and renamed for semantic analysis
    existing = FuncDef("bar", [], Block([]))
    existing.info = cls.info
    cls.info.names["bar"] = SymbolTableNode(MDEF, existing)
    add_method(api, cls, "bar", args=[arg], return_type=NoneType())
    assert "bar" in cls.info.names
    assert any(name.startswith("bar") and name != "bar" for name in cls.info.names)

    # Classmethod path
    add_method(
        api, cls, "cm", args=[arg], return_type=AnyType(TypeOfAny.explicit), is_classmethod=True
    )
    assert "cm" in cls.info.names
    assert isinstance(cls.info.names["cm"].node, Decorator)


def test_transform_defer_paths() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=False, init_forbid_extra=False, warn_untyped_fields=True
    )

    class DeferTransformer(SQLModelTransformer):
        def collect_fields(self) -> list[SQLModelField] | None:  # noqa: D401
            return None

    t = DeferTransformer(cls, cls, api, plugin_config)
    assert t.transform() is False
    assert api.deferred is True

    # If any field has no type, we defer as well.
    api = DummyAPI()
    t2 = SQLModelTransformer(cls, cls, api, plugin_config)
    t2.collect_fields = lambda: [
        SQLModelField(name="x", has_default=False, line=1, column=0, type=None, info=cls.info)
    ]  # type: ignore[method-assign]
    assert t2.transform() is False
    assert api.deferred is True


def test_transform_generates_init_and_model_construct_and_metadata() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=True, init_forbid_extra=False, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    any_t = AnyType(TypeOfAny.explicit)
    fields = [
        SQLModelField(name="a", has_default=False, line=1, column=0, type=any_t, info=cls.info),
        SQLModelField(name="b", has_default=True, line=1, column=0, type=any_t, info=cls.info),
    ]
    transformer.collect_fields = lambda: fields  # type: ignore[method-assign]
    assert transformer.transform() is True

    # Plugin-generated methods are registered.
    assert "__init__" in cls.info.names
    assert "model_construct" in cls.info.names

    # Metadata is JSON-serializable.
    assert METADATA_KEY in cls.info.metadata
    assert set(cls.info.metadata[METADATA_KEY]["fields"].keys()) == {"a", "b"}


def test_field_expand_type_self_type_branch_and_invariant_translation() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()

    # Fake a "Self" type variable on the TypeInfo to exercise expand_type mapping.
    self_tv = TypeVarType(
        "SelfT",
        "SelfT",
        TypeVarId(1),
        [],
        AnyType(TypeOfAny.explicit),
        AnyType(TypeOfAny.explicit),
        INVARIANT,
    )
    cls.info.self_type = self_tv  # type: ignore[attr-defined]

    field = SQLModelField(
        name="x",
        has_default=False,
        line=1,
        column=0,
        type=self_tv,
        info=cls.info,
    )
    expanded = field.expand_type(cls.info, api, force_typevars_invariant=True)
    assert isinstance(expanded, Instance)

    # ForceInvariantTypeVars makes non-invariant type vars invariant.
    tv = TypeVarType(
        "T",
        "T",
        TypeVarId(2),
        [],
        AnyType(TypeOfAny.explicit),
        AnyType(TypeOfAny.explicit),
        1,  # not INVARIANT
    )
    assert tv.variance != INVARIANT
    tv2 = tv.accept(ForceInvariantTypeVars())
    assert isinstance(tv2, TypeVarType)
    assert tv2.variance == INVARIANT


def test_collect_fields_includes_if_blocks_and_inherited_metadata() -> None:
    # Build a base class with serialized metadata and a derived class that inherits it.
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=False, init_forbid_extra=True, warn_untyped_fields=True
    )

    base_cls = make_sqlmodel_class("m.Base")
    # NOTE: `SQLModelField.deserialize(...)` requires a real mypy `SemanticAnalyzerPluginInterface`,
    # so we keep the inherited metadata empty here and focus on exercising the traversal and
    # dependency registration paths.
    base_any = AnyType(TypeOfAny.explicit)
    base_cls.info.metadata[METADATA_KEY] = {"fields": {}}

    derived_cls = make_sqlmodel_class("m.Derived")
    derived_cls.info.mro = [derived_cls.info, base_cls.info, make_typeinfo("builtins.object")]

    # Add a typed field under an if-statement to exercise block traversal.
    field_var = Var("name", base_any)
    derived_cls.info.names["name"] = SymbolTableNode(MDEF, field_var)
    if_body = Block(
        [
            AssignmentStmt([NameExpr("name")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)),
        ]
    )
    # Ensure the assignment is treated as typed (new syntax).
    if_body.body[0].new_syntax = True  # type: ignore[attr-defined]

    if_stmt = IfStmt([NameExpr("True")], [if_body], None)
    derived_cls.defs.body.append(if_stmt)

    transformer = SQLModelTransformer(derived_cls, derived_cls, api, plugin_config)
    fields = transformer.collect_fields()
    assert fields is not None
    assert {f.name for f in fields} == {"name"}

    # init_forbid_extra=True -> no kwargs in signatures
    transformer.add_initializer(fields)
    init_node = derived_cls.info.names["__init__"].node
    assert isinstance(init_node, FuncDef)
    assert all(arg.variable.name != "kwargs" for arg in init_node.arguments)

    transformer.add_model_construct(fields)
    mc_node = derived_cls.info.names["model_construct"].node
    assert isinstance(mc_node, Decorator)
    assert isinstance(mc_node.func, FuncDef)

    # Ensure we recorded plugin dependency on the base
    assert any("m.Base" in dep for dep in api.dependencies)
