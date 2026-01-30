"""SQLModel-specific class transformation for mypy.

This module is intentionally small and focused:
- collect SQLModel fields (treating `sqlmodel.Field(...)` correctly for requiredness)
- include relationship kwargs for table-model constructors
- record relationship member names (and types) for later typing hooks
- synthesize `__init__` on SQLModel subclasses
"""

from __future__ import annotations

import ast
import keyword
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
    DictExpr,
    EllipsisExpr,
    Expression,
    FuncDef,
    IfStmt,
    JsonDict,
    NameExpr,
    PlaceholderNode,
    RefExpr,
    Statement,
    StrExpr,
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


def _plugin_any() -> AnyType:
    """Return an Any that should not trigger `disallow_any_explicit`."""
    return AnyType(TypeOfAny.implementation_artifact)


ERROR_FIELD = ErrorCode("sqlmodel-field", "SQLModel field error", "SQLModel")


def _parse_annotated_field_metadata_by_line(source: str) -> dict[int, tuple[bool, list[str]]]:
    """Return mapping of line -> (has_default, aliases) for Annotated Field metadata.

    Best-effort support for patterns like:
    - `x: Annotated[T, Field(...)]`
    - `x: Optional[Annotated[T, Field(...)] ]`

    Only handles class-body `AnnAssign` nodes with **no assignment value** (i.e. `x: ...`).
    """

    def _call_func_name(expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Name):
            return expr.id
        if isinstance(expr, ast.Attribute):
            return expr.attr
        return None

    def _call_get_kwarg(call: ast.Call, name: str) -> ast.expr | None:
        for kw in call.keywords:
            if kw.arg == name:
                return kw.value
        return None

    def _subscript_args(slice_expr: ast.expr) -> list[ast.expr]:
        if isinstance(slice_expr, ast.Tuple):
            return list(slice_expr.elts)
        return [slice_expr]

    def _is_none(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Constant) and expr.value is None

    def _is_true(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Constant) and expr.value is True

    def _is_ellipsis(expr: ast.expr) -> bool:
        return isinstance(expr, ast.Constant) and expr.value is Ellipsis

    def _literal_str_or_none(expr: ast.expr) -> str | None:
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if _is_none(expr):
            return None
        return None

    def _is_usable_kwarg_name(name: str) -> bool:
        return name.isidentifier() and not keyword.iskeyword(name)

    def _field_aliases_from_call(call: ast.Call) -> list[str]:
        alias_expr = _call_get_kwarg(call, "alias")
        alias_arg = _literal_str_or_none(alias_expr) if alias_expr is not None else None

        validation_alias_expr = _call_get_kwarg(call, "validation_alias")
        validation_alias_arg = (
            _literal_str_or_none(validation_alias_expr)
            if validation_alias_expr is not None
            else None
        )
        schema_validation_alias: str | None = None

        schema_extra_expr = _call_get_kwarg(call, "schema_extra")
        if isinstance(schema_extra_expr, ast.Dict):
            for key_expr, val_expr in zip(
                schema_extra_expr.keys, schema_extra_expr.values, strict=True
            ):
                if isinstance(key_expr, ast.Constant) and isinstance(key_expr.value, str):
                    if key_expr.value == "validation_alias":
                        schema_validation_alias = _literal_str_or_none(val_expr)

        validation_alias_final = validation_alias_arg or schema_validation_alias or alias_arg

        aliases: list[str] = []
        for candidate in (alias_arg, validation_alias_final):
            if candidate is None:
                continue
            if not _is_usable_kwarg_name(candidate):
                continue
            if candidate in aliases:
                continue
            aliases.append(candidate)
        return aliases

    def _column_expr_is_defaultish(expr: ast.expr) -> bool:
        if not isinstance(expr, ast.Call):
            return False

        callee_name = _call_func_name(expr.func)
        if callee_name is not None and not callee_name.endswith("Column"):
            return False

        for arg in expr.args:
            if isinstance(arg, ast.Call):
                computed_name = _call_func_name(arg.func)
                if computed_name is not None and computed_name.endswith("Computed"):
                    return True

        for kw in expr.keywords:
            if kw.arg == "nullable" and _is_true(kw.value):
                return True
            if kw.arg in {"server_default", "default", "insert_default"} and not _is_none(kw.value):
                return True
        return False

    def _sa_column_kwargs_are_defaultish(expr: ast.expr) -> bool:
        if not isinstance(expr, ast.Dict):
            return False
        for key_expr, val_expr in zip(expr.keys, expr.values, strict=True):
            if not (isinstance(key_expr, ast.Constant) and isinstance(key_expr.value, str)):
                continue
            if key_expr.value == "nullable" and _is_true(val_expr):
                return True
            if key_expr.value in {"server_default", "default", "insert_default"} and not _is_none(
                val_expr
            ):
                return True
        return False

    def _field_has_default_from_call(call: ast.Call) -> bool:
        # Explicit default always wins.
        default_expr = _call_get_kwarg(call, "default")
        if default_expr is None and call.args:
            default_expr = call.args[0]
        if default_expr is not None:
            return not _is_ellipsis(default_expr)

        factory_expr = _call_get_kwarg(call, "default_factory")
        if factory_expr is not None:
            return not _is_none(factory_expr)

        nullable_expr = _call_get_kwarg(call, "nullable")
        if nullable_expr is not None and _is_true(nullable_expr):
            return True

        sa_column_expr = _call_get_kwarg(call, "sa_column")
        if sa_column_expr is not None and _column_expr_is_defaultish(sa_column_expr):
            return True

        sa_column_kwargs_expr = _call_get_kwarg(call, "sa_column_kwargs")
        if sa_column_kwargs_expr is not None and _sa_column_kwargs_are_defaultish(
            sa_column_kwargs_expr
        ):
            return True

        return False

    def _find_field_call_in_annotation(expr: ast.expr) -> ast.Call | None:
        # PEP 604 unions: `X | None`
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            return _find_field_call_in_annotation(expr.left) or _find_field_call_in_annotation(
                expr.right
            )

        if isinstance(expr, ast.Subscript):
            base_name = _call_func_name(expr.value)
            args = _subscript_args(expr.slice)

            if base_name == "Annotated" and len(args) >= 2:
                for meta in args[1:]:
                    if isinstance(meta, ast.Call) and _call_func_name(meta.func) == "Field":
                        return meta
                return None

            if base_name in {"Optional", "Union"}:
                for item in args:
                    found = _find_field_call_in_annotation(item)
                    if found is not None:
                        return found

        return None

    out: dict[int, tuple[bool, list[str]]] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return out

    class _Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_depth = 0

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.class_depth += 1
            self.generic_visit(node)
            self.class_depth -= 1

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            # Don't descend into methods / nested functions.
            return

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            return

        def visit_Lambda(self, node: ast.Lambda) -> None:  # noqa: N802
            return

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
            if self.class_depth <= 0:
                return
            if node.value is not None:
                return
            if not isinstance(node.target, ast.Name):
                return

            field_call = _find_field_call_in_annotation(node.annotation)
            if field_call is None:
                return

            out[node.lineno] = (
                _field_has_default_from_call(field_call),
                _field_aliases_from_call(field_call),
            )

    _Visitor().visit(tree)
    return out


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
        aliases: list[str] | None = None,
        type: Type | None,
        info: TypeInfo,
    ) -> None:
        self.name = name
        self.has_default = has_default
        self.line = line
        self.column = column
        self.aliases = list(aliases or [])
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
            "aliases": self.aliases,
            "type": self.type.serialize(),
        }

    @classmethod
    def deserialize(
        cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface
    ) -> SQLModelField:
        data = data.copy()
        aliases = data.pop("aliases", [])
        if isinstance(aliases, list):
            deduped: list[str] = []
            seen = set()
            for alias in aliases:
                if not isinstance(alias, str):
                    continue
                if alias in seen:
                    continue
                seen.add(alias)
                deduped.append(alias)
            aliases = deduped
        else:
            aliases = []
        typ = deserialize_and_fixup_type(data.pop("type"), api)
        name = data.pop("name")
        return cls(name=name, type=typ, aliases=aliases, info=info, **data)

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
        self._annotated_field_metadata_by_line: dict[int, tuple[bool, list[str]]] | None = None

    def _get_annotated_field_metadata_by_line(self) -> dict[int, tuple[bool, list[str]]]:
        if self._annotated_field_metadata_by_line is not None:
            return self._annotated_field_metadata_by_line

        # Best-effort: only available when mypy provides module path info.
        path = ""
        try:
            mod = self._api.modules.get(self._api.cur_mod_id)
            if mod is not None:
                path = mod.path
        except Exception:
            path = ""

        if not path:
            self._annotated_field_metadata_by_line = {}
            return self._annotated_field_metadata_by_line

        try:
            with open(path, encoding="utf-8") as f:
                src = f.read()
        except OSError:
            self._annotated_field_metadata_by_line = {}
            return self._annotated_field_metadata_by_line

        self._annotated_field_metadata_by_line = _parse_annotated_field_metadata_by_line(src)
        return self._annotated_field_metadata_by_line

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
            "is_table_model": _is_table_model(info),
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
        aliases = self.get_field_aliases(stmt)
        if isinstance(stmt.rvalue, TempNode):
            annotated = self._get_annotated_field_metadata_by_line().get(stmt.line)
            if annotated is not None:
                has_default, aliases = annotated
        init_type = node.type

        return SQLModelField(
            name=name,
            has_default=has_default,
            line=stmt.line,
            column=stmt.column,
            aliases=aliases,
            type=init_type,
            info=self._cls.info,
        )

    @staticmethod
    def get_has_default(stmt: AssignmentStmt) -> bool:
        expr = stmt.rvalue

        # `x: int = Field(...)`
        if isinstance(expr, CallExpr) and _callee_fullname(expr) == SQLMODEL_FIELD_FULLNAME:

            def _is_none_expr(value: Expression) -> bool:
                return isinstance(value, NameExpr) and (
                    value.fullname == "builtins.None" or value.name == "None"
                )

            def _column_expr_is_defaultish(value: Expression) -> bool:
                """Best-effort detection of SQLAlchemy-generated/defaultable columns.

                We treat these as "optional" constructor kwargs:
                - nullable=True
                - server_default / default / insert_default
                - Computed(...) columns
                """
                if not isinstance(value, CallExpr):
                    return False

                callee_fullname = _callee_fullname(value)
                if callee_fullname is not None and not callee_fullname.endswith(".Column"):
                    # Be conservative: avoid interpreting non-Column calls.
                    return False

                for c_arg, c_arg_name in zip(value.args, value.arg_names, strict=True):
                    # Positional Computed(...) (common in declarative).
                    if c_arg_name is None and isinstance(c_arg, CallExpr):
                        computed_fullname = _callee_fullname(c_arg)
                        if computed_fullname is not None and computed_fullname.endswith(
                            ".Computed"
                        ):
                            return True

                    # Keyword hints.
                    if c_arg_name == "nullable" and _is_bool_nameexpr(c_arg, True):
                        return True
                    if c_arg_name in {"server_default", "default", "insert_default"}:
                        if not _is_none_expr(c_arg):
                            return True

                return False

            def _sa_column_kwargs_are_defaultish(value: Expression) -> bool:
                """Best-effort for `Field(sa_column_kwargs={...})`."""
                if not isinstance(value, DictExpr):
                    return False

                for key_expr, val_expr in value.items:
                    if not isinstance(key_expr, StrExpr):
                        continue
                    if key_expr.value == "nullable" and _is_bool_nameexpr(val_expr, True):
                        return True
                    if key_expr.value in {"server_default", "default", "insert_default"}:
                        if not _is_none_expr(val_expr):
                            return True
                return False

            saw_nullable_true = False
            sa_column_expr: Expression | None = None
            sa_column_kwargs_expr: Expression | None = None

            for arg, arg_name in zip(expr.args, expr.arg_names, strict=True):
                # Explicit default always wins.
                if arg_name is None or arg_name == "default":
                    return arg.__class__ is not EllipsisExpr
                if arg_name == "default_factory":
                    return not _is_none_expr(arg)

                if arg_name == "nullable" and _is_bool_nameexpr(arg, True):
                    saw_nullable_true = True
                elif arg_name == "sa_column":
                    sa_column_expr = arg
                elif arg_name == "sa_column_kwargs":
                    sa_column_kwargs_expr = arg

            if saw_nullable_true:
                return True
            if sa_column_expr is not None and _column_expr_is_defaultish(sa_column_expr):
                return True
            if sa_column_kwargs_expr is not None and _sa_column_kwargs_are_defaultish(
                sa_column_kwargs_expr
            ):
                return True

            return False

        if isinstance(expr, TempNode):
            return False

        # `x: int = ...` is required
        if isinstance(expr, EllipsisExpr):
            return False

        return True

    @staticmethod
    def get_field_aliases(stmt: AssignmentStmt) -> list[str]:
        """Best-effort collection of Field() alias names usable as constructor kwargs.

        Mirrors SQLModel's precedence for validation aliases:
        `validation_alias` > `schema_extra["validation_alias"]` > `alias`.

        We currently only expose aliases that are statically known string literals and are valid
        Python keyword names.
        """
        expr = stmt.rvalue
        if not (isinstance(expr, CallExpr) and _callee_fullname(expr) == SQLMODEL_FIELD_FULLNAME):
            return []

        alias_arg: str | None = None
        validation_alias_arg: str | None = None
        serialization_alias_arg: str | None = None
        schema_validation_alias: str | None = None
        schema_serialization_alias: str | None = None

        def _literal_str_or_none(value: Expression) -> str | None:
            if isinstance(value, StrExpr):
                return value.value
            if isinstance(value, NameExpr) and (
                value.fullname == "builtins.None" or value.name == "None"
            ):
                return None
            return None

        def _is_usable_kwarg_name(name: str) -> bool:
            return name.isidentifier() and not keyword.iskeyword(name)

        for arg, arg_name in zip(expr.args, expr.arg_names, strict=True):
            if arg_name == "alias":
                alias_arg = _literal_str_or_none(arg)
            elif arg_name == "validation_alias":
                validation_alias_arg = _literal_str_or_none(arg)
            elif arg_name == "serialization_alias":
                serialization_alias_arg = _literal_str_or_none(arg)
            elif arg_name == "schema_extra" and isinstance(arg, DictExpr):
                for key_expr, value_expr in arg.items:
                    if not isinstance(key_expr, StrExpr):
                        continue
                    if key_expr.value == "validation_alias":
                        schema_validation_alias = _literal_str_or_none(value_expr)
                    elif key_expr.value == "serialization_alias":
                        schema_serialization_alias = _literal_str_or_none(value_expr)

        validation_alias_final = validation_alias_arg or schema_validation_alias or alias_arg
        _serialization_alias_final = (
            serialization_alias_arg or schema_serialization_alias or alias_arg
        )

        aliases: list[str] = []
        for candidate in (alias_arg, validation_alias_final):
            if candidate is None:
                continue
            if not _is_usable_kwarg_name(candidate):
                continue
            if candidate in aliases:
                continue
            aliases.append(candidate)
        return aliases

    def add_initializer(
        self, fields: list[SQLModelField], relationships: list[SQLModelRelationship]
    ) -> None:
        info = self._cls.info

        if "__init__" in info.names and not info.names["__init__"].plugin_generated:
            return

        typed = self.plugin_config.init_typed
        table_model = _is_table_model(info)

        canonical_names = {field.name for field in fields}
        if table_model:
            canonical_names.update(rel.name for rel in relationships)
        if not self.plugin_config.init_forbid_extra:
            canonical_names.add("kwargs")

        args: list[Argument] = []
        field_params: list[tuple[SQLModelField, Type, list[str]]] = []
        for field in fields:
            expanded = field.expand_type(info, self._api, force_typevars_invariant=True)
            if expanded is not None:
                expanded = expanded.accept(ForceInvariantTypeVars())
                expanded = _unwrap_mapped_type(expanded)
            if typed and expanded is not None:
                type_annotation = expanded
            else:
                type_annotation = _plugin_any()

            field_aliases = [
                alias
                for alias in field.aliases
                if alias != field.name and alias not in canonical_names
            ]
            variable = Var(field.name, type_annotation)
            args.append(
                Argument(
                    variable=variable,
                    type_annotation=type_annotation,
                    initializer=None,
                    kind=ARG_NAMED_OPT if field.has_default or field_aliases else ARG_NAMED,
                )
            )
            field_params.append((field, type_annotation, field_aliases))

        if table_model:
            for rel in relationships:
                expanded = rel.expand_type(info, self._api, force_typevars_invariant=True)
                if expanded is not None:
                    expanded = expanded.accept(ForceInvariantTypeVars())
                    expanded = _unwrap_mapped_type(expanded)
                if typed and expanded is not None:
                    type_annotation = expanded
                else:
                    type_annotation = _plugin_any()
                variable = Var(rel.name, type_annotation)
                args.append(
                    Argument(
                        variable=variable,
                        type_annotation=type_annotation,
                        initializer=None,
                        kind=ARG_NAMED_OPT,
                    )
                )

        used_names = set(canonical_names)
        for _field, type_annotation, field_aliases in field_params:
            for alias in field_aliases:
                if alias in used_names:
                    continue
                used_names.add(alias)
                variable = Var(alias, type_annotation)
                args.append(
                    Argument(
                        variable=variable,
                        type_annotation=type_annotation,
                        initializer=None,
                        kind=ARG_NAMED_OPT,
                    )
                )

        if not self.plugin_config.init_forbid_extra:
            kw = _plugin_any()
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

        table_model = _is_table_model(info)

        canonical_names = {field.name for field in fields}
        if table_model:
            canonical_names.update(rel.name for rel in relationships)
        canonical_names.add("_fields_set")
        if not self.plugin_config.init_forbid_extra:
            canonical_names.add("kwargs")

        args: list[Argument] = []
        field_params: list[tuple[SQLModelField, Type, list[str]]] = []
        for field in fields:
            expanded = field.expand_type(info, self._api, force_typevars_invariant=True)
            # `model_construct` bypasses validation, so this is always typed.
            if expanded is not None:
                expanded = expanded.accept(ForceInvariantTypeVars())
                expanded = _unwrap_mapped_type(expanded)
            type_annotation = expanded or _plugin_any()

            field_aliases = [
                alias
                for alias in field.aliases
                if alias != field.name and alias not in canonical_names
            ]
            variable = Var(field.name, type_annotation)
            args.append(
                Argument(
                    variable=variable,
                    type_annotation=type_annotation,
                    initializer=None,
                    kind=ARG_NAMED_OPT if field.has_default or field_aliases else ARG_NAMED,
                )
            )
            field_params.append((field, type_annotation, field_aliases))

        if table_model:
            for rel in relationships:
                expanded = rel.expand_type(info, self._api, force_typevars_invariant=True)
                if expanded is not None:
                    expanded = expanded.accept(ForceInvariantTypeVars())
                    expanded = _unwrap_mapped_type(expanded)
                type_annotation = expanded or _plugin_any()
                variable = Var(rel.name, type_annotation)
                args.append(
                    Argument(
                        variable=variable,
                        type_annotation=type_annotation,
                        initializer=None,
                        kind=ARG_NAMED_OPT,
                    )
                )

        used_names = set(canonical_names)
        for _field, type_annotation, field_aliases in field_params:
            for alias in field_aliases:
                if alias in used_names:
                    continue
                used_names.add(alias)
                variable = Var(alias, type_annotation)
                args.append(
                    Argument(
                        variable=variable,
                        type_annotation=type_annotation,
                        initializer=None,
                        kind=ARG_NAMED_OPT,
                    )
                )

        if not self.plugin_config.init_forbid_extra:
            kw = _plugin_any()
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

    # Prefer persisted metadata when the class-body AST isn't available
    # (common in incremental mode when loading modules from cache).
    md = info.metadata.get(METADATA_KEY)
    if isinstance(md, dict):
        md_table = md.get("is_table_model")
        if isinstance(md_table, bool):
            return md_table
        # Backwards-compatible alias for older metadata shapes.
        md_table = md.get("table")
        if isinstance(md_table, bool):
            return md_table

    # Inherit `table=True` from bases if present.
    for base in info.mro[1:]:
        if base.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            continue
        kw = base.defn.keywords.get("table")
        if kw is not None and _is_bool_nameexpr(kw, True):
            return True
        base_md = base.metadata.get(METADATA_KEY)
        if isinstance(base_md, dict) and base_md.get("is_table_model") is True:
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
