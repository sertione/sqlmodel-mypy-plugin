"""SQLModel-specific class transformation for mypy.

This module is intentionally small and focused:
- collect SQLModel fields (treating `sqlmodel.Field(...)` correctly for requiredness)
- include relationship kwargs for table-model constructors
- record relationship member names (and types) for later typing hooks
- synthesize `__init__` on SQLModel subclasses
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR2,
    INVARIANT,
    MDEF,
    Argument,
    AssignmentStmt,
    Block,
    CallExpr,
    ClassDef,
    Decorator,
    EllipsisExpr,
    Expression,
    FuncDef,
    IfStmt,
    JsonDict,
    NameExpr,
    PlaceholderNode,
    RefExpr,
    Statement,
    SymbolTableNode,
    TempNode,
    TypeAlias,
    TypeInfo,
    Var,
)
from mypy.plugin import SemanticAnalyzerPluginInterface
from mypy.plugins.common import deserialize_and_fixup_type
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.type_visitor import TypeTranslator
from mypy.typeops import map_type_from_supertype
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    NoneType,
    Type,
    TypeAliasType,
    TypeOfAny,
    TypeType,
    TypeVarType,
    UnionType,
    get_proper_type,
)
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name


class SQLModelPluginConfig(Protocol):
    init_typed: bool
    init_forbid_extra: bool
    warn_untyped_fields: bool


METADATA_KEY = "sqlmodel-mypy-metadata"

SQLMODEL_BASEMODEL_FULLNAME = "sqlmodel.main.SQLModel"
SQLMODEL_FIELD_FULLNAME = "sqlmodel.main.Field"
SQLMODEL_RELATIONSHIP_FULLNAME = "sqlmodel.main.Relationship"

SQLALCHEMY_MAPPED_FULLNAMES = {"sqlalchemy.orm.Mapped", "sqlalchemy.orm.base.Mapped"}

ERROR_FIELD = ErrorCode("sqlmodel-field", "SQLModel field error", "SQLModel")


class ForceInvariantTypeVars(TypeTranslator):
    def visit_type_var(self, t: TypeVarType) -> Type:  # noqa: D102
        if t.variance == INVARIANT:
            return t
        # `TypeVarType.copy_modified(variance=...)` does not reliably update the variance across mypy
        # versions, so mutate explicitly.
        modified = t.copy_modified()
        modified.variance = INVARIANT
        return modified

    def visit_type_alias_type(self, t: TypeAliasType) -> Type:  # noqa: D102
        # Expand type aliases and then continue translation.
        return get_proper_type(t).accept(self)


class SQLModelField:
    """A collected SQLModel field used for signature generation."""

    def __init__(
        self,
        *,
        name: str,
        has_default: bool,
        line: int,
        column: int,
        type: Type | None,
        info: TypeInfo,
    ) -> None:
        self.name = name
        self.has_default = has_default
        self.line = line
        self.column = column
        self.type = type
        self.info = info

    def expand_type(
        self,
        current_info: TypeInfo,
        api: SemanticAnalyzerPluginInterface,
        *,
        force_typevars_invariant: bool = False,
    ) -> Type | None:
        if self.type is None:
            return None

        typ = self.type
        if force_typevars_invariant and isinstance(typ, TypeVarType):
            modified = typ.copy_modified()
            modified.variance = INVARIANT
            typ = modified

        # This plugin runs very late in semantic analysis for the class, so types should be ready.
        if self.info.self_type is not None:
            with state.strict_optional_set(api.options.strict_optional):
                filled_with_typevars = fill_typevars(current_info)
                assert isinstance(filled_with_typevars, Instance)
                return expand_type(typ, {self.info.self_type.id: filled_with_typevars})
        return typ

    def serialize(self) -> JsonDict:
        assert self.type is not None
        return {
            "name": self.name,
            "has_default": self.has_default,
            "line": self.line,
            "column": self.column,
            "type": self.type.serialize(),
        }

    @classmethod
    def deserialize(
        cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface
    ) -> SQLModelField:
        data = data.copy()
        typ = deserialize_and_fixup_type(data.pop("type"), api)
        name = data.pop("name")
        return cls(name=name, type=typ, info=info, **data)

    def expand_typevar_from_subtype(
        self, sub_type: TypeInfo, api: SemanticAnalyzerPluginInterface
    ) -> None:
        if self.type is None:
            return
        with state.strict_optional_set(api.options.strict_optional):
            self.type = map_type_from_supertype(self.type, sub_type, self.info)


class SQLModelRelationship:
    """A collected SQLModel relationship.

    We store relationship member names in metadata so later hooks can adjust class attribute typing
    (e.g. `Hero.team` in SQLAlchemy expressions).
    """

    def __init__(self, *, name: str, line: int, column: int) -> None:
        self.name = name
        self.line = line
        self.column = column

        # Optional: available once semantic analysis resolves the annotation.
        self.type: Type | None = None
        self.info: TypeInfo | None = None

    def expand_type(
        self,
        current_info: TypeInfo,
        api: SemanticAnalyzerPluginInterface,
        *,
        force_typevars_invariant: bool = False,
    ) -> Type | None:
        if self.type is None or self.info is None:
            return None

        typ = self.type
        if force_typevars_invariant and isinstance(typ, TypeVarType):
            modified = typ.copy_modified()
            modified.variance = INVARIANT
            typ = modified

        # Mirror SQLModelField.expand_type() behavior.
        if self.info.self_type is not None:
            with state.strict_optional_set(api.options.strict_optional):
                filled_with_typevars = fill_typevars(current_info)
                assert isinstance(filled_with_typevars, Instance)
                return expand_type(typ, {self.info.self_type.id: filled_with_typevars})
        return typ

    def serialize(self) -> JsonDict:
        return {
            "name": self.name,
            "line": self.line,
            "column": self.column,
            "type": None if self.type is None else self.type.serialize(),
        }

    @classmethod
    def deserialize(
        cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface
    ) -> SQLModelRelationship:
        data = data.copy()
        typ_data = data.pop("type", None)
        name = data.pop("name")
        rel = cls(name=name, **data)
        rel.info = info
        if typ_data is not None:
            rel.type = deserialize_and_fixup_type(typ_data, api)
        return rel

    def expand_typevar_from_subtype(
        self, sub_type: TypeInfo, api: SemanticAnalyzerPluginInterface
    ) -> None:
        if self.type is None or self.info is None:
            return
        with state.strict_optional_set(api.options.strict_optional):
            self.type = map_type_from_supertype(self.type, sub_type, self.info)


class SQLModelTransformer:
    def __init__(
        self,
        cls: ClassDef,
        reason: Expression | Statement,
        api: SemanticAnalyzerPluginInterface,
        plugin_config: SQLModelPluginConfig,
    ) -> None:
        self._cls = cls
        self._reason = reason
        self._api = api
        self.plugin_config = plugin_config

    def transform(self) -> bool:
        info = self._cls.info

        members = self.collect_members()
        if members is None:
            self._api.defer()
            return False
        fields, relationships = members

        for f in fields:
            if f.type is None:
                self._api.defer()
                return False

        # Relationship types are nice-to-have but not required to proceed; if unknown,
        # relationship args fall back to Any in signatures.
        relationship_list = [relationships[k] for k in sorted(relationships)]

        self.add_initializer(fields, relationship_list)
        self.add_model_construct(fields, relationship_list)

        info.metadata[METADATA_KEY] = {
            "fields": {field.name: field.serialize() for field in fields},
            "relationships": {k: relationships[k].serialize() for k in sorted(relationships)},
        }
        return True

    def collect_members(self) -> tuple[list[SQLModelField], dict[str, SQLModelRelationship]] | None:
        cls = self._cls
        info = cls.info

        found_fields: dict[str, SQLModelField] = {}
        found_relationships: dict[str, SQLModelRelationship] = {}

        # 1) Inherited fields (base first).
        for base_info in reversed(info.mro[1:-1]):  # exclude current class and object
            if METADATA_KEY not in base_info.metadata:
                # If this is an SQLModel subclass we haven't processed yet, defer so we can include it.
                if base_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
                    continue
                if base_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
                    return None
                continue

            self._api.add_plugin_dependency(make_wildcard_trigger(base_info.fullname))
            base_metadata = base_info.metadata[METADATA_KEY]
            for name, data in base_metadata.get("fields", {}).items():
                sym_node = info.names.get(name)
                if sym_node and sym_node.node and not isinstance(sym_node.node, Var):
                    self._api.fail(
                        "SQLModel field may only be overridden by another field",
                        sym_node.node,
                        code=ERROR_FIELD,
                    )
                base_field = SQLModelField.deserialize(base_info, data, self._api)
                base_field.expand_typevar_from_subtype(info, self._api)
                found_fields[name] = base_field

            rels = base_metadata.get("relationships", {})
            if isinstance(rels, dict):
                for name, data in rels.items():
                    if not isinstance(name, str) or not isinstance(data, dict):
                        continue
                    base_rel = SQLModelRelationship.deserialize(base_info, data, self._api)
                    base_rel.expand_typevar_from_subtype(info, self._api)
                    found_relationships[name] = base_rel
            elif isinstance(rels, list):
                # Backwards compatibility with older metadata formats.
                for item in rels:
                    if isinstance(item, str):
                        found_relationships[item] = SQLModelRelationship(
                            name=item, line=base_info.line, column=base_info.column
                        )
                    elif isinstance(item, dict) and isinstance(item.get("name"), str):
                        base_rel = SQLModelRelationship.deserialize(base_info, item, self._api)
                        base_rel.expand_typevar_from_subtype(info, self._api)
                        found_relationships[base_rel.name] = base_rel

        # 2) Current class fields.
        for stmt in self._get_assignment_statements_from_block(cls.defs):
            member = self.collect_member_from_stmt(stmt)
            if member is None:
                continue
            if isinstance(member, SQLModelField):
                found_relationships.pop(member.name, None)
                found_fields[member.name] = member
            else:
                found_fields.pop(member.name, None)
                found_relationships[member.name] = member

        return list(found_fields.values()), found_relationships

    def _get_assignment_statements_from_if_statement(
        self, stmt: IfStmt
    ) -> Iterator[AssignmentStmt]:
        for body in stmt.body:
            if not body.is_unreachable:
                yield from self._get_assignment_statements_from_block(body)
        if stmt.else_body is not None and not stmt.else_body.is_unreachable:
            yield from self._get_assignment_statements_from_block(stmt.else_body)

    def _get_assignment_statements_from_block(self, block: Block) -> Iterator[AssignmentStmt]:
        for stmt in block.body:
            if isinstance(stmt, AssignmentStmt):
                yield stmt
            elif isinstance(stmt, IfStmt):
                yield from self._get_assignment_statements_from_if_statement(stmt)

    def collect_member_from_stmt(
        self, stmt: AssignmentStmt
    ) -> SQLModelField | SQLModelRelationship | None:
        # Untyped assignment (e.g. `x = Field(...)`).
        if not stmt.new_syntax:
            lhs = stmt.lvalues[0]
            if (
                self.plugin_config.warn_untyped_fields
                and isinstance(lhs, NameExpr)
                and isinstance(stmt.rvalue, CallExpr)
                and _callee_fullname(stmt.rvalue)
                in {SQLMODEL_FIELD_FULLNAME, SQLMODEL_RELATIONSHIP_FULLNAME}
            ):
                self._api.fail("Untyped fields disallowed", stmt, code=ERROR_FIELD)
            return None

        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr):
            return None

        name = lhs.name
        if name == "model_config" or name.startswith("_"):
            return None

        # Relationship: `foo: list[Bar] = Relationship(...)`
        if (
            isinstance(stmt.rvalue, CallExpr)
            and _callee_fullname(stmt.rvalue) == SQLMODEL_RELATIONSHIP_FULLNAME
        ):
            rel = SQLModelRelationship(name=name, line=stmt.line, column=stmt.column)
            rel.info = self._cls.info
            # Best-effort: keep the declared annotation type for typing relationship kwargs.
            sym = self._cls.info.names.get(name)
            if sym is not None and isinstance(sym.node, Var) and not sym.node.is_classvar:
                rel.type = sym.node.type
            return rel

        sym = self._cls.info.names.get(name)
        if sym is None or sym.node is None:
            return None

        node = sym.node
        if isinstance(node, PlaceholderNode):
            return None
        if isinstance(node, TypeAlias):
            return None
        if not isinstance(node, Var):
            return None
        if node.is_classvar:
            return None

        has_default = self.get_has_default(stmt)
        init_type = node.type

        return SQLModelField(
            name=name,
            has_default=has_default,
            line=stmt.line,
            column=stmt.column,
            type=init_type,
            info=self._cls.info,
        )

    @staticmethod
    def get_has_default(stmt: AssignmentStmt) -> bool:
        expr = stmt.rvalue
        if isinstance(expr, TempNode):
            return False

        # `x: int = Field(...)`
        if isinstance(expr, CallExpr) and _callee_fullname(expr) == SQLMODEL_FIELD_FULLNAME:
            for arg, arg_name in zip(expr.args, expr.arg_names, strict=True):
                if arg_name is None or arg_name == "default":
                    return arg.__class__ is not EllipsisExpr
                if arg_name == "default_factory":
                    return not (isinstance(arg, NameExpr) and arg.fullname == "builtins.None")
            return False

        # `x: int = ...` is required
        if isinstance(expr, EllipsisExpr):
            return False

        return True

    def add_initializer(
        self, fields: list[SQLModelField], relationships: list[SQLModelRelationship]
    ) -> None:
        info = self._cls.info

        if "__init__" in info.names and not info.names["__init__"].plugin_generated:
            return

        typed = self.plugin_config.init_typed

        args: list[Argument] = []
        for field in fields:
            expanded = field.expand_type(info, self._api, force_typevars_invariant=True)
            if expanded is not None:
                expanded = expanded.accept(ForceInvariantTypeVars())
                expanded = _unwrap_mapped_type(expanded)
            if typed and expanded is not None:
                type_annotation = expanded
            else:
                type_annotation = AnyType(TypeOfAny.explicit)

            variable = Var(field.name, type_annotation)
            args.append(
                Argument(
                    variable=variable,
                    type_annotation=type_annotation,
                    initializer=None,
                    kind=ARG_NAMED_OPT if field.has_default else ARG_NAMED,
                )
            )

        if _is_table_model(info):
            for rel in relationships:
                expanded = rel.expand_type(info, self._api, force_typevars_invariant=True)
                if expanded is not None:
                    expanded = expanded.accept(ForceInvariantTypeVars())
                    expanded = _unwrap_mapped_type(expanded)
                if typed and expanded is not None:
                    type_annotation = expanded
                else:
                    type_annotation = AnyType(TypeOfAny.explicit)
                variable = Var(rel.name, type_annotation)
                args.append(
                    Argument(
                        variable=variable,
                        type_annotation=type_annotation,
                        initializer=None,
                        kind=ARG_NAMED_OPT,
                    )
                )

        if not self.plugin_config.init_forbid_extra:
            kw = AnyType(TypeOfAny.explicit)
            kwargs_var = Var("kwargs", kw)
            args.append(Argument(kwargs_var, kw, None, ARG_STAR2))

        add_method(self._api, self._cls, "__init__", args=args, return_type=NoneType())

    def add_model_construct(
        self, fields: list[SQLModelField], relationships: list[SQLModelRelationship]
    ) -> None:
        info = self._cls.info

        if "model_construct" in info.names and not info.names["model_construct"].plugin_generated:
            return

        set_str = self._api.named_type("builtins.set", [self._api.named_type("builtins.str")])
        optional_set_str = UnionType([set_str, NoneType()])
        fields_set_argument = Argument(
            Var("_fields_set", optional_set_str), optional_set_str, None, ARG_OPT
        )

        args: list[Argument] = []
        for field in fields:
            expanded = field.expand_type(info, self._api, force_typevars_invariant=True)
            # `model_construct` bypasses validation, so this is always typed.
            if expanded is not None:
                expanded = expanded.accept(ForceInvariantTypeVars())
                expanded = _unwrap_mapped_type(expanded)
            type_annotation = expanded or AnyType(TypeOfAny.explicit)

            variable = Var(field.name, type_annotation)
            args.append(
                Argument(
                    variable=variable,
                    type_annotation=type_annotation,
                    initializer=None,
                    kind=ARG_NAMED_OPT if field.has_default else ARG_NAMED,
                )
            )

        if _is_table_model(info):
            for rel in relationships:
                expanded = rel.expand_type(info, self._api, force_typevars_invariant=True)
                if expanded is not None:
                    expanded = expanded.accept(ForceInvariantTypeVars())
                    expanded = _unwrap_mapped_type(expanded)
                type_annotation = expanded or AnyType(TypeOfAny.explicit)
                variable = Var(rel.name, type_annotation)
                args.append(
                    Argument(
                        variable=variable,
                        type_annotation=type_annotation,
                        initializer=None,
                        kind=ARG_NAMED_OPT,
                    )
                )

        if not self.plugin_config.init_forbid_extra:
            kw = AnyType(TypeOfAny.explicit)
            kwargs_var = Var("kwargs", kw)
            args.append(Argument(kwargs_var, kw, None, ARG_STAR2))

        add_method(
            self._api,
            self._cls,
            "model_construct",
            args=[fields_set_argument, *args],
            return_type=fill_typevars(self._cls.info),
            is_classmethod=True,
        )


def _callee_fullname(call: CallExpr) -> str | None:
    callee: Any = call.callee
    if isinstance(callee, RefExpr):
        return callee.fullname
    return None


def _unwrap_mapped_type(typ: Type) -> Type:
    """Convert SQLAlchemy `Mapped[T]` annotations to `T` for constructor signatures.

    SQLModel model attributes can be annotated as `Mapped[T]` for ORM mapping, but users pass plain `T`
    values into `__init__`/`model_construct` (not `Mapped[T]` descriptors).
    """
    proper = get_proper_type(typ)
    if isinstance(proper, Instance) and proper.type.fullname in SQLALCHEMY_MAPPED_FULLNAMES:
        if proper.args:
            return proper.args[0]
        return AnyType(TypeOfAny.from_omitted_generics)
    return typ


def _is_bool_nameexpr(expr: Expression, value: bool) -> bool:
    if not isinstance(expr, NameExpr):
        return False
    if value:
        return expr.fullname == "builtins.True" or expr.name == "True"
    return expr.fullname == "builtins.False" or expr.name == "False"


def _is_table_model(info: TypeInfo) -> bool:
    """Best-effort detection for `class Model(SQLModel, table=True)`."""
    kw = info.defn.keywords.get("table")
    if kw is not None:
        return _is_bool_nameexpr(kw, True)

    # Inherit `table=True` from bases if present.
    for base in info.mro[1:]:
        if base.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            continue
        kw = base.defn.keywords.get("table")
        if kw is not None and _is_bool_nameexpr(kw, True):
            return True
    return False


def add_method(
    api: SemanticAnalyzerPluginInterface,
    cls: ClassDef,
    name: str,
    args: list[Argument],
    return_type: Type,
    *,
    is_classmethod: bool = False,
) -> None:
    """Close to `mypy.plugins.common.add_method_to_class`, but avoids `self` name conflicts.

    SQLModel models can legally have a field named `self`; generating a signature using a parameter named
    `self` causes a `no-redef` error in mypy. Use a name that cannot conflict.
    """
    info = cls.info

    # Remove previously generated methods.
    if name in info.names:
        sym = info.names[name]
        if sym.plugin_generated and isinstance(sym.node, FuncDef):
            cls.defs.body.remove(sym.node)

    # Keep existing definition for semantic analysis by renaming it.
    if name in info.names:
        r_name = get_unique_redefinition_name(name, info.names)
        info.names[r_name] = info.names[name]

    function_type = api.named_type("builtins.function")

    if is_classmethod:
        cls_type = TypeType(fill_typevars(info))
        first = [Argument(Var("_cls"), cls_type, None, ARG_POS, True)]
    else:
        inst_type = fill_typevars(info)
        first = [Argument(Var("__sqlmodel_self__"), inst_type, None, ARG_POS)]

    full_args = first + args
    arg_types: list[Type] = []
    arg_names: list[str] = []
    arg_kinds = []
    for arg in full_args:
        assert arg.type_annotation is not None
        arg_types.append(arg.type_annotation)
        arg_names.append(arg.variable.name)
        arg_kinds.append(arg.kind)

    signature = CallableType(arg_types, arg_kinds, arg_names, return_type, function_type)

    from mypy.nodes import PassStmt

    func = FuncDef(name, full_args, Block([PassStmt()]))
    func.info = info
    func.type = set_callable_name(signature, func)
    func.is_class = is_classmethod
    func._fullname = info.fullname + "." + name
    func.line = info.line

    if is_classmethod:
        func.is_decorated = True
        v = Var(name, func.type)
        v.info = info
        v._fullname = func._fullname
        v.is_classmethod = True
        dec = Decorator(func, [NameExpr("classmethod")], v)
        dec.line = info.line
        sym = SymbolTableNode(MDEF, dec)
    else:
        sym = SymbolTableNode(MDEF, func)

    sym.plugin_generated = True
    info.names[name] = sym
    info.defn.defs.body.append(func)
