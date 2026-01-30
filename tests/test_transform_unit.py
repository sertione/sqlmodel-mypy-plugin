from __future__ import annotations

from types import SimpleNamespace

from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_POS,
    INVARIANT,
    MDEF,
    Argument,
    AssignmentStmt,
    Block,
    CallExpr,
    ClassDef,
    Decorator,
    DictExpr,
    EllipsisExpr,
    FuncDef,
    IfStmt,
    NameExpr,
    PlaceholderNode,
    StrExpr,
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
    SQLModelRelationship,
    SQLModelTransformer,
    _callee_fullname,
    _parse_annotated_field_metadata_by_line,
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

    # x: int = 1 -> optional
    stmt = AssignmentStmt([NameExpr("x")], NameExpr("y"))
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(nullable=True) -> optional
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[NameExpr("True")],
            arg_names=["nullable"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(sa_column=Column(..., nullable=True)) -> optional
    col_nullable = make_field_call(
        fullname="sqlalchemy.Column",
        args=[NameExpr("True")],
        arg_names=["nullable"],
    )
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[col_nullable],
            arg_names=["sa_column"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(sa_column=Column(Computed(...))) -> optional
    computed = make_field_call(
        fullname="sqlalchemy.Computed",
        args=[StrExpr("x")],
        arg_names=[None],
    )
    col_computed = make_field_call(
        fullname="sqlalchemy.Column",
        args=[computed],
        arg_names=[None],
    )
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[col_computed],
            arg_names=["sa_column"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(sa_column=Column(server_default=...)) -> optional
    col_server_default = make_field_call(
        fullname="sqlalchemy.Column",
        args=[NameExpr("x")],
        arg_names=["server_default"],
    )
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[col_server_default],
            arg_names=["sa_column"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(sa_column_kwargs={...}) -> optional (nullable/server_default hints)
    sa_column_kwargs = DictExpr([(StrExpr("nullable"), NameExpr("True"))])
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[sa_column_kwargs],
            arg_names=["sa_column_kwargs"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is True

    # x: int = Field(sa_column=Column()) -> required (no nullability/default hints)
    col_plain = make_field_call(fullname="sqlalchemy.Column")
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[col_plain],
            arg_names=["sa_column"],
        ),
    )
    stmt.new_syntax = True
    assert transformer.get_has_default(stmt) is False


def test_get_field_aliases_covers_common_cases() -> None:
    # alias=...
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[StrExpr("full_name")],
            arg_names=["alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == ["full_name"]

    # alias + validation_alias override
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[StrExpr("a"), StrExpr("va")],
            arg_names=["alias", "validation_alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == ["a", "va"]

    # schema_extra validation alias
    schema_extra = DictExpr([(StrExpr("validation_alias"), StrExpr("sa"))])
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[StrExpr("a"), schema_extra],
            arg_names=["alias", "schema_extra"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == ["a", "sa"]

    # Ignore invalid keyword names.
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[StrExpr("full-name")],
            arg_names=["alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # Ignore non-literal values.
    dyn = NameExpr("ALIAS")
    dyn.fullname = "m.ALIAS"
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[dyn],
            arg_names=["alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # Non-Field calls -> no aliases.
    stmt = AssignmentStmt([NameExpr("x")], NameExpr("y"))
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # Explicit alias=None -> treated as absent.
    none_expr = NameExpr("None")
    none_expr.fullname = "builtins.None"
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[none_expr],
            arg_names=["alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # serialization_alias is parsed (but is not a constructor kwarg).
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[StrExpr("ser")],
            arg_names=["serialization_alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # schema_extra serialization alias is also parsed (but is not a constructor kwarg).
    schema_extra = DictExpr([(StrExpr("serialization_alias"), StrExpr("ser"))])
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[schema_extra],
            arg_names=["schema_extra"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # Ignore schema_extra keys that are not string literals.
    schema_extra = DictExpr([(NameExpr("validation_alias"), StrExpr("sa"))])
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[schema_extra],
            arg_names=["schema_extra"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []

    # Ignore Python keywords as argument names.
    stmt = AssignmentStmt(
        [NameExpr("x")],
        make_field_call(
            fullname=SQLMODEL_FIELD_FULLNAME,
            args=[StrExpr("class")],
            arg_names=["alias"],
        ),
    )
    stmt.new_syntax = True
    assert SQLModelTransformer.get_field_aliases(stmt) == []


def test_parse_annotated_field_metadata_by_line_covers_common_cases() -> None:
    src = """
from __future__ import annotations

from typing import Annotated, Optional

from sqlmodel import Field, SQLModel


class Model(SQLModel):
    id: Annotated[int | None, Field(default=None, primary_key=True)]
    name: Annotated[str, Field(alias="full_name")]
    v: Optional[Annotated[str, Field(default=None, alias="a", validation_alias="va")]]
""".lstrip()

    out = _parse_annotated_field_metadata_by_line(src)

    lines = src.splitlines()
    id_line = next(i for i, line in enumerate(lines, start=1) if "id:" in line)
    name_line = next(i for i, line in enumerate(lines, start=1) if "name:" in line)
    v_line = next(i for i, line in enumerate(lines, start=1) if "v:" in line)

    assert out[id_line] == (True, [])
    assert out[name_line] == (False, ["full_name"])
    assert out[v_line] == (True, ["a", "va"])


def test_parse_annotated_field_metadata_by_line_covers_defaultish_hints_and_union_syntax() -> None:
    src = """
from __future__ import annotations

import sqlmodel
from typing import Annotated

from sqlalchemy import Column, Computed, String, text


class Model(sqlmodel.SQLModel):
    a: Annotated[str, sqlmodel.Field(alias="attr_alias")]
    b: Annotated[str, sqlmodel.Field(alias="a", schema_extra={"validation_alias": "sa"})]
    c: Annotated[int, sqlmodel.Field(default=...)]
    d: Annotated[int, sqlmodel.Field(default_factory=None)]
    e: Annotated[int, sqlmodel.Field(default_factory=lambda: 1)]
    f: Annotated[str, sqlmodel.Field(nullable=True)]
    g: Annotated[str, sqlmodel.Field(sa_column=Column("g", String(), nullable=True))]
    h: Annotated[str | None, sqlmodel.Field(sa_column=Column("h", String(), Computed("1", persisted=True)))]
    i: Annotated[str, sqlmodel.Field(sa_column=Column("i", String(), server_default=text("'x'")))]
    j: Annotated[str, sqlmodel.Field(sa_column_kwargs={"server_default": "x"})]
    k: Annotated[str, sqlmodel.Field(alias="k")] | None
""".lstrip()

    out = _parse_annotated_field_metadata_by_line(src)
    lines = src.splitlines()

    def ln(prefix: str) -> int:
        return next(i for i, line in enumerate(lines, start=1) if line.strip().startswith(prefix))

    assert out[ln("a:")] == (False, ["attr_alias"])
    assert out[ln("b:")] == (False, ["a", "sa"])
    assert out[ln("c:")] == (False, [])
    assert out[ln("d:")] == (False, [])
    assert out[ln("e:")] == (True, [])
    assert out[ln("f:")] == (True, [])
    assert out[ln("g:")] == (True, [])
    assert out[ln("h:")] == (True, [])
    assert out[ln("i:")] == (True, [])
    assert out[ln("j:")] == (True, [])
    assert out[ln("k:")] == (False, ["k"])


def test_parse_annotated_field_metadata_by_line_exercises_edge_paths() -> None:
    assert _parse_annotated_field_metadata_by_line("class C(") == {}

    src = """
from __future__ import annotations

from typing import Annotated

from sqlmodel import Field, SQLModel


class Model(SQLModel):
    # alias=None should be ignored
    a: Annotated[str, Field(alias=None)]
    # Python keyword aliases should be ignored
    b: Annotated[str, Field(alias="class")]
    # Annotated metadata with no Field(...) should be ignored
    c: Annotated[str, "meta"]

    def method(self) -> None:
        self.x = 1
""".lstrip()

    out = _parse_annotated_field_metadata_by_line(src)
    lines = src.splitlines()
    a_line = next(i for i, line in enumerate(lines, start=1) if line.strip().startswith("a:"))
    b_line = next(i for i, line in enumerate(lines, start=1) if line.strip().startswith("b:"))
    c_line = next(i for i, line in enumerate(lines, start=1) if line.strip().startswith("c:"))

    assert out[a_line] == (False, [])
    assert out[b_line] == (False, [])
    assert c_line not in out


def test_collect_member_from_stmt_untyped_warns() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=False, init_forbid_extra=False, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    # x = Field(...)
    stmt = AssignmentStmt([NameExpr("x")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = False
    assert transformer.collect_member_from_stmt(stmt) is None
    assert ("Untyped fields disallowed", ERROR_FIELD) in api.failed

    # Relationship() is also disallowed when untyped
    api.failed.clear()
    stmt = AssignmentStmt(
        [NameExpr("rel")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt.new_syntax = False
    assert transformer.collect_member_from_stmt(stmt) is None
    assert ("Untyped fields disallowed", ERROR_FIELD) in api.failed


def test_collect_member_from_stmt_typed_collects_and_skips() -> None:
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
    assert transformer.collect_member_from_stmt(stmt) is None

    stmt = AssignmentStmt([NameExpr("_hidden")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_member_from_stmt(stmt) is None

    # Relationships are tracked (but excluded from constructor args)
    stmt = AssignmentStmt(
        [NameExpr("team")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt.new_syntax = True
    rel = transformer.collect_member_from_stmt(stmt)
    assert isinstance(rel, SQLModelRelationship)
    assert rel.name == "team"

    # Non-NameExpr lvalues are ignored
    stmt = AssignmentStmt(
        [make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)],
        make_field_call(fullname=SQLMODEL_FIELD_FULLNAME),
    )
    stmt.new_syntax = True
    assert transformer.collect_member_from_stmt(stmt) is None

    # A proper typed field is collected
    any_t = AnyType(TypeOfAny.explicit)
    var = Var("x", any_t)
    cls.info.names["x"] = SymbolTableNode(MDEF, var)
    stmt = AssignmentStmt([NameExpr("x")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    field = transformer.collect_member_from_stmt(stmt)
    assert field is not None
    assert isinstance(field, SQLModelField)
    assert field.name == "x"
    assert field.type is any_t

    # ClassVar fields are ignored
    var2 = Var("y", any_t)
    var2.is_classvar = True
    cls.info.names["y"] = SymbolTableNode(MDEF, var2)
    stmt = AssignmentStmt([NameExpr("y")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_member_from_stmt(stmt) is None

    # PlaceholderNode and TypeAlias nodes are ignored
    cls.info.names["p"] = SymbolTableNode(MDEF, PlaceholderNode("m.p", NameExpr("p"), 1))
    stmt = AssignmentStmt([NameExpr("p")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_member_from_stmt(stmt) is None

    alias = TypeAlias(AnyType(TypeOfAny.explicit), "m.alias", "m", 1, 0)
    cls.info.names["alias"] = SymbolTableNode(MDEF, alias)
    stmt = AssignmentStmt([NameExpr("alias")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME))
    stmt.new_syntax = True
    assert transformer.collect_member_from_stmt(stmt) is None


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
        def collect_members(  # noqa: D401
            self,
        ) -> tuple[list[SQLModelField], dict[str, SQLModelRelationship]] | None:
            return None

    t = DeferTransformer(cls, cls, api, plugin_config)
    assert t.transform() is False
    assert api.deferred is True

    # If any field has no type, we defer as well.
    api = DummyAPI()
    t2 = SQLModelTransformer(cls, cls, api, plugin_config)
    t2.collect_members = lambda: (  # type: ignore[method-assign]
        [SQLModelField(name="x", has_default=False, line=1, column=0, type=None, info=cls.info)],
        {},
    )
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
    transformer.collect_members = lambda: (fields, {})  # type: ignore[method-assign]
    assert transformer.transform() is True

    # Plugin-generated methods are registered.
    assert "__init__" in cls.info.names
    assert "model_construct" in cls.info.names

    # Metadata is JSON-serializable.
    assert METADATA_KEY in cls.info.metadata
    assert set(cls.info.metadata[METADATA_KEY]["fields"].keys()) == {"a", "b"}
    assert cls.info.metadata[METADATA_KEY]["relationships"] == {}


def test_transform_adds_field_alias_kwargs_to_signatures_and_metadata() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=True, init_forbid_extra=True, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    any_t = AnyType(TypeOfAny.explicit)
    fields = [
        SQLModelField(
            name="name",
            has_default=False,
            line=1,
            column=0,
            aliases=["full_name"],
            type=any_t,
            info=cls.info,
        )
    ]
    transformer.collect_members = lambda: (fields, {})  # type: ignore[method-assign]
    assert transformer.transform() is True

    init_node = cls.info.names["__init__"].node
    assert isinstance(init_node, FuncDef)
    init_args = {a.variable.name: a for a in init_node.arguments}
    assert "name" in init_args
    assert "full_name" in init_args
    assert init_args["name"].kind == ARG_NAMED_OPT
    assert init_args["full_name"].kind == ARG_NAMED_OPT

    mc_node = cls.info.names["model_construct"].node
    assert isinstance(mc_node, Decorator)
    assert isinstance(mc_node.func, FuncDef)
    mc_args = {a.variable.name: a for a in mc_node.func.arguments}
    assert "_fields_set" in mc_args
    assert "name" in mc_args
    assert "full_name" in mc_args
    assert mc_args["name"].kind == ARG_NAMED_OPT
    assert mc_args["full_name"].kind == ARG_NAMED_OPT

    meta = cls.info.metadata[METADATA_KEY]
    assert meta["fields"]["name"]["aliases"] == ["full_name"]


def test_transform_table_model_includes_relationship_kwargs_and_types() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=True, init_forbid_extra=True, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    # Treat this class as `table=True`.
    cls.info.defn.keywords["table"] = NameExpr("True")

    # Required field.
    any_t = AnyType(TypeOfAny.explicit)
    name_var = Var("name", any_t)
    cls.info.names["name"] = SymbolTableNode(MDEF, name_var)
    stmt_name = AssignmentStmt(
        [NameExpr("name")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)
    )
    stmt_name.new_syntax = True

    # Typed relationship.
    team_info = make_typeinfo("m.Team")
    team_t = Instance(team_info, [])
    team_var = Var("team", team_t)
    cls.info.names["team"] = SymbolTableNode(MDEF, team_var)
    stmt_team = AssignmentStmt(
        [NameExpr("team")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True

    # Relationship with missing type -> falls back to Any.
    zzz_var = Var("zzz", None)
    cls.info.names["zzz"] = SymbolTableNode(MDEF, zzz_var)
    stmt_zzz = AssignmentStmt(
        [NameExpr("zzz")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_zzz.new_syntax = True

    cls.defs.body.extend([stmt_name, stmt_team, stmt_zzz])
    assert transformer.transform() is True

    meta = cls.info.metadata[METADATA_KEY]
    assert set(meta["relationships"].keys()) == {"team", "zzz"}

    init_node = cls.info.names["__init__"].node
    assert isinstance(init_node, FuncDef)
    init_arg_names = [a.variable.name for a in init_node.arguments]
    assert "kwargs" not in init_arg_names
    assert "team" in init_arg_names
    assert "zzz" in init_arg_names

    team_arg = next(a for a in init_node.arguments if a.variable.name == "team")
    assert team_arg.kind == ARG_NAMED_OPT
    assert team_arg.type_annotation == team_t

    zzz_arg = next(a for a in init_node.arguments if a.variable.name == "zzz")
    assert isinstance(zzz_arg.type_annotation, AnyType)

    mc_node = cls.info.names["model_construct"].node
    assert isinstance(mc_node, Decorator)
    assert isinstance(mc_node.func, FuncDef)
    mc_arg_names = [a.variable.name for a in mc_node.func.arguments]
    assert "kwargs" not in mc_arg_names
    assert "team" in mc_arg_names
    assert "zzz" in mc_arg_names


def test_transform_non_table_model_does_not_accept_relationship_kwargs() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=True, init_forbid_extra=True, warn_untyped_fields=True
    )
    transformer = SQLModelTransformer(cls, cls, api, plugin_config)

    # Add SQLModel base to the MRO so `_is_table_model()` exercises the base-skip path.
    sqlmodel_info = make_typeinfo("sqlmodel.main.SQLModel")
    obj_info = sqlmodel_info.mro[-1]
    cls.info.mro = [cls.info, sqlmodel_info, obj_info]

    any_t = AnyType(TypeOfAny.explicit)
    name_var = Var("name", any_t)
    cls.info.names["name"] = SymbolTableNode(MDEF, name_var)
    stmt_name = AssignmentStmt(
        [NameExpr("name")], make_field_call(fullname=SQLMODEL_FIELD_FULLNAME)
    )
    stmt_name.new_syntax = True

    team_info = make_typeinfo("m.Team")
    team_t = Instance(team_info, [])
    team_var = Var("team", team_t)
    cls.info.names["team"] = SymbolTableNode(MDEF, team_var)
    stmt_team = AssignmentStmt(
        [NameExpr("team")], make_field_call(fullname=SQLMODEL_RELATIONSHIP_FULLNAME)
    )
    stmt_team.new_syntax = True

    cls.defs.body.extend([stmt_name, stmt_team])
    assert transformer.transform() is True

    # Relationship is recorded, but not accepted in signatures for non-table models.
    meta = cls.info.metadata[METADATA_KEY]
    assert set(meta["relationships"].keys()) == {"team"}

    init_node = cls.info.names["__init__"].node
    assert isinstance(init_node, FuncDef)
    init_arg_names = [a.variable.name for a in init_node.arguments]
    assert "team" not in init_arg_names

    mc_node = cls.info.names["model_construct"].node
    assert isinstance(mc_node, Decorator)
    assert isinstance(mc_node.func, FuncDef)
    mc_arg_names = [a.variable.name for a in mc_node.func.arguments]
    assert "team" not in mc_arg_names


def test_transform_inherits_relationships_from_old_metadata_list_format() -> None:
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=True, init_forbid_extra=True, warn_untyped_fields=True
    )

    sqlmodel_info = make_typeinfo("sqlmodel.main.SQLModel")
    obj_info = sqlmodel_info.mro[-1]

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_info, obj_info]
    base_info.metadata[METADATA_KEY] = {"fields": {}, "relationships": ["team"]}

    derived_cls = make_sqlmodel_class("m.User")
    derived_cls.info.mro = [derived_cls.info, base_info, sqlmodel_info, obj_info]
    derived_cls.info.defn.keywords["table"] = NameExpr("True")

    transformer = SQLModelTransformer(derived_cls, derived_cls, api, plugin_config)
    assert transformer.transform() is True

    init_node = derived_cls.info.names["__init__"].node
    assert isinstance(init_node, FuncDef)
    team_arg = next(a for a in init_node.arguments if a.variable.name == "team")
    assert isinstance(team_arg.type_annotation, AnyType)


def test_transform_defers_when_base_sqlmodel_subclass_not_processed() -> None:
    api = DummyAPI()
    plugin_config = SimpleNamespace(
        init_typed=True, init_forbid_extra=False, warn_untyped_fields=True
    )

    sqlmodel_info = make_typeinfo("sqlmodel.main.SQLModel")
    obj_info = sqlmodel_info.mro[-1]

    base_info = make_typeinfo("m.Base")
    base_info.mro = [base_info, sqlmodel_info, obj_info]
    # No METADATA_KEY on base_info.metadata -> should force deferral for SQLModel subclasses.

    derived_cls = make_sqlmodel_class("m.User")
    derived_cls.info.mro = [derived_cls.info, base_info, sqlmodel_info, obj_info]

    transformer = SQLModelTransformer(derived_cls, derived_cls, api, plugin_config)
    assert transformer.transform() is False
    assert api.deferred is True


def test_relationship_expand_type_self_type_branch() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()

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

    rel = SQLModelRelationship(name="rel", line=1, column=0)
    rel.type = self_tv
    rel.info = cls.info

    expanded = rel.expand_type(cls.info, api, force_typevars_invariant=True)
    assert isinstance(expanded, Instance)


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

    # Invariant vars are returned unchanged.
    tv_inv = TypeVarType(
        "TInv",
        "TInv",
        TypeVarId(3),
        [],
        AnyType(TypeOfAny.explicit),
        AnyType(TypeOfAny.explicit),
        INVARIANT,
    )
    tv_inv2 = tv_inv.accept(ForceInvariantTypeVars())
    assert tv_inv2 is tv_inv


def test_field_expand_type_returns_none_when_missing_type() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()

    field = SQLModelField(
        name="x",
        has_default=False,
        line=1,
        column=0,
        type=None,
        info=cls.info,
    )
    assert field.expand_type(cls.info, api) is None
    field.expand_typevar_from_subtype(cls.info, api)


def test_sqlmodelfield_deserialize_dedupes_aliases() -> None:
    cls = make_sqlmodel_class()
    api = DummyAPI()

    any_t = AnyType(TypeOfAny.explicit)
    field = SQLModelField(
        name="x",
        has_default=False,
        line=1,
        column=0,
        aliases=["a"],
        type=any_t,
        info=cls.info,
    )
    data = field.serialize()
    data["aliases"] = ["a", "a", 1]  # type: ignore[assignment]

    # `deserialize_and_fixup_type()` expects a real mypy `SemanticAnalyzerPluginInterface`
    # instance; patch it for this unit test so we can exercise the alias handling.
    from unittest.mock import patch

    with patch("sqlmodel_mypy.transform.deserialize_and_fixup_type", return_value=any_t):
        field2 = SQLModelField.deserialize(cls.info, data, api)  # type: ignore[arg-type]
    assert field2.aliases == ["a"]
    assert isinstance(field2.type, AnyType)


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
    members = transformer.collect_members()
    assert members is not None
    fields, relationships = members
    assert {f.name for f in fields} == {"name"}
    assert relationships == {}

    # init_forbid_extra=True -> no kwargs in signatures
    transformer.add_initializer(fields, [])
    init_node = derived_cls.info.names["__init__"].node
    assert isinstance(init_node, FuncDef)
    assert all(arg.variable.name != "kwargs" for arg in init_node.arguments)

    transformer.add_model_construct(fields, [])
    mc_node = derived_cls.info.names["model_construct"].node
    assert isinstance(mc_node, Decorator)
    assert isinstance(mc_node.func, FuncDef)

    # Ensure we recorded plugin dependency on the base
    assert any("m.Base" in dep for dep in api.dependencies)
