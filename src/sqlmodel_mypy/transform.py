"""SQLModel-specific class transformation for mypy.

This module is intentionally small and focused:
- collect SQLModel fields (treating `sqlmodel.Field(...)` correctly for requiredness)
- ignore `sqlmodel.Relationship(...)` for constructor generation
- synthesize `__init__` on SQLModel subclasses
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

from mypy.expandtype import expand_type
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_POS,
    ARG_STAR2,
    INVARIANT,
    MDEF,
    Argument,
    AssignmentStmt,
    Block,
    CallExpr,
    ClassDef,
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
from mypy.typeops import map_type_from_supertype
from mypy.types import (
    AnyType,
    CallableType,
    Instance,
    NoneType,
    Type,
    TypeOfAny,
    TypeType,
    TypeVarType,
)
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name


class SQLModelPluginConfig(Protocol):
    init_typed: bool
    init_forbid_extra: bool


METADATA_KEY = "sqlmodel-mypy-metadata"

SQLMODEL_BASEMODEL_FULLNAME = "sqlmodel.main.SQLModel"
SQLMODEL_FIELD_FULLNAME = "sqlmodel.main.Field"
SQLMODEL_RELATIONSHIP_FULLNAME = "sqlmodel.main.Relationship"


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

        fields = self.collect_fields()
        if fields is None:
            self._api.defer()
            return False

        for f in fields:
            if f.type is None:
                self._api.defer()
                return False

        self.add_initializer(fields)

        info.metadata[METADATA_KEY] = {
            "fields": {field.name: field.serialize() for field in fields},
        }
        return True

    def collect_fields(self) -> list[SQLModelField] | None:
        cls = self._cls
        info = cls.info

        found_fields: dict[str, SQLModelField] = {}

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
            for name, data in base_info.metadata[METADATA_KEY]["fields"].items():
                base_field = SQLModelField.deserialize(base_info, data, self._api)
                base_field.expand_typevar_from_subtype(info, self._api)
                found_fields[name] = base_field

        # 2) Current class fields.
        for stmt in self._get_assignment_statements_from_block(cls.defs):
            current_field = self.collect_field_from_stmt(stmt)
            if current_field is None:
                continue
            found_fields[current_field.name] = current_field

        return list(found_fields.values())

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

    def collect_field_from_stmt(self, stmt: AssignmentStmt) -> SQLModelField | None:
        # Only annotated assignments (`x: int = ...`); ignore untyped assignments.
        if not stmt.new_syntax:
            return None

        lhs = stmt.lvalues[0]
        if not isinstance(lhs, NameExpr):
            return None

        name = lhs.name
        if name == "model_config" or name.startswith("_"):
            return None

        # Skip relationships: `foo: list[Bar] = Relationship(...)`
        if (
            isinstance(stmt.rvalue, CallExpr)
            and _callee_fullname(stmt.rvalue) == SQLMODEL_RELATIONSHIP_FULLNAME
        ):
            return None

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

    def add_initializer(self, fields: list[SQLModelField]) -> None:
        info = self._cls.info

        if "__init__" in info.names and not info.names["__init__"].plugin_generated:
            return

        typed = self.plugin_config.init_typed

        args: list[Argument] = []
        for field in fields:
            expanded = field.expand_type(info, self._api, force_typevars_invariant=True)
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

        if not self.plugin_config.init_forbid_extra:
            kw = AnyType(TypeOfAny.explicit)
            kwargs_var = Var("kwargs", kw)
            args.append(Argument(kwargs_var, kw, None, ARG_STAR2))

        add_method(self._api, self._cls, "__init__", args=args, return_type=NoneType())


def _callee_fullname(call: CallExpr) -> str | None:
    callee: Any = call.callee
    if isinstance(callee, RefExpr):
        return callee.fullname
    return None


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

    sym = SymbolTableNode(MDEF, func)
    sym.plugin_generated = True
    info.names[name] = sym
    info.defn.defs.body.append(func)
