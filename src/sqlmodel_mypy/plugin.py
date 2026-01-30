"""Mypy plugin entry point for SQLModel."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from configparser import ConfigParser
from typing import Any, NamedTuple, cast

from mypy.errorcodes import ErrorCode
from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_POS,
    ARG_STAR,
    ARG_STAR2,
    MDEF,
    ArgKind,
    AssignmentStmt,
    Block,
    CallExpr,
    Decorator,
    Expression,
    FuncDef,
    IfStmt,
    MemberExpr,
    NameExpr,
    RefExpr,
    StrExpr,
    SymbolTableNode,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import (
    AttributeContext,
    ClassDefContext,
    FunctionContext,
    FunctionSigContext,
    MethodContext,
    MethodSigContext,
    Plugin,
    ReportConfigContext,
)
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import (
    AnyType,
    CallableType,
    FunctionLike,
    Instance,
    NoneType,
    TupleType,
    Type,
    TypeOfAny,
    TypeType,
    UnionType,
    get_proper_type,
)
from mypy.typevars import fill_typevars

from .transform import (
    ERROR_FIELD,
    METADATA_KEY,
    SQLALCHEMY_MAPPED_FULLNAMES,
    SQLMODEL_FIELD_FULLNAME,
    SQLMODEL_RELATIONSHIP_FULLNAME,
    ForceInvariantTypeVars,
    SQLModelTransformer,
    _callee_fullname,
    _unwrap_mapped_type,
)

CONFIGFILE_KEY = "sqlmodel-mypy"
SQLMODEL_BASEMODEL_FULLNAME = "sqlmodel.main.SQLModel"
SQLMODEL_METACLASS_FULLNAME = "sqlmodel.main.SQLModelMetaclass"

ERROR_PLUGIN_ORDER = ErrorCode(
    "sqlmodel-plugin-order",
    "SQLModel plugin order error",
    "SQLModel",
)

BUILTINS_GETATTR_FULLNAME = "builtins.getattr"

SQLMODEL_COL_FULLNAME = "sqlmodel.sql.expression.col"
# SQLModel's `select()` is implemented in a generated module and re-exported.
SQLMODEL_SELECT_GEN_FULLNAME = "sqlmodel.sql._expression_select_gen.select"
SQLMODEL_SELECT_FULLNAMES = {
    SQLMODEL_SELECT_GEN_FULLNAME,
    "sqlmodel.sql.expression.select",
    "sqlmodel.select",
}

SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES = {
    "sqlalchemy.orm.InstrumentedAttribute",
    "sqlalchemy.orm.attributes.InstrumentedAttribute",
}

SQLALCHEMY_COLUMN_PROPERTY_FULLNAMES = {
    # Imported from `sqlalchemy.orm` (common user import).
    "sqlalchemy.orm.column_property",
    # May resolve to the internal defining module depending on stubs/version.
    "sqlalchemy.orm._orm_constructors.column_property",
}

SQLALCHEMY_SELECT_FULLNAME = "sqlalchemy.sql.selectable.Select"
SQLALCHEMY_SELECT_JOIN_FULLNAME = f"{SQLALCHEMY_SELECT_FULLNAME}.join"
SQLALCHEMY_SELECT_JOIN_FROM_FULLNAME = f"{SQLALCHEMY_SELECT_FULLNAME}.join_from"
SQLALCHEMY_SELECT_OUTERJOIN_FULLNAME = f"{SQLALCHEMY_SELECT_FULLNAME}.outerjoin"
SQLALCHEMY_SELECT_OUTERJOIN_FROM_FULLNAME = f"{SQLALCHEMY_SELECT_FULLNAME}.outerjoin_from"

# SQLModel wraps SQLAlchemy's Select in its own generic Select class.
SQLMODEL_SELECT_CLS_FULLNAME = "sqlmodel.sql._expression_select_cls.Select"
SQLMODEL_SELECT_JOIN_FULLNAME = f"{SQLMODEL_SELECT_CLS_FULLNAME}.join"
SQLMODEL_SELECT_JOIN_FROM_FULLNAME = f"{SQLMODEL_SELECT_CLS_FULLNAME}.join_from"
SQLMODEL_SELECT_OUTERJOIN_FULLNAME = f"{SQLMODEL_SELECT_CLS_FULLNAME}.outerjoin"
SQLMODEL_SELECT_OUTERJOIN_FROM_FULLNAME = f"{SQLMODEL_SELECT_CLS_FULLNAME}.outerjoin_from"

# Relationship / comparator methods used in query expressions.
#
# These are intentionally a small set of best-effort targets; mypy may resolve
# calls against different bases depending on SQLAlchemy version and stubs.
SQLALCHEMY_RELATIONSHIP_COMPARATOR_METHOD_FULLNAMES = {
    # ORM-level helpers (typed on SQLAlchemy's side under TYPE_CHECKING).
    "sqlalchemy.orm.base.SQLORMOperations.any",
    "sqlalchemy.orm.base.SQLORMOperations.has",
    # Comparator base (also typed under TYPE_CHECKING).
    "sqlalchemy.orm.interfaces.PropComparator.any",
    "sqlalchemy.orm.interfaces.PropComparator.has",
    # Instrumented attribute access (mypy may resolve against the concrete type).
    "sqlalchemy.orm.InstrumentedAttribute.any",
    "sqlalchemy.orm.InstrumentedAttribute.has",
    "sqlalchemy.orm.InstrumentedAttribute.contains",
    "sqlalchemy.orm.attributes.InstrumentedAttribute.any",
    "sqlalchemy.orm.attributes.InstrumentedAttribute.has",
    "sqlalchemy.orm.attributes.InstrumentedAttribute.contains",
    "sqlalchemy.orm.attributes.QueryableAttribute.any",
    "sqlalchemy.orm.attributes.QueryableAttribute.has",
    "sqlalchemy.orm.attributes.QueryableAttribute.contains",
    # Relationship comparator implementation.
    "sqlalchemy.orm.relationships.RelationshipProperty.Comparator.any",
    "sqlalchemy.orm.relationships.RelationshipProperty.Comparator.has",
    "sqlalchemy.orm.relationships.RelationshipProperty.Comparator.contains",
    # SQLCoreOperations / ColumnOperators surface `.contains(...)` for strings and
    # relationships; for relationships, SQLAlchemy dispatches to the relationship
    # comparator at runtime.
    "sqlalchemy.sql.elements.SQLCoreOperations.contains",
    "sqlalchemy.sql.operators.ColumnOperators.contains",
}

SQLMODEL_SESSION_EXEC_FULLNAME = "sqlmodel.orm.session.Session.exec"
SQLMODEL_ASYNC_SESSION_EXEC_FULLNAME = "sqlmodel.ext.asyncio.session.AsyncSession.exec"

# Increment when plugin changes should invalidate mypy cache.
__version__ = 20


class _CollectedField(NamedTuple):
    name: str
    aliases: tuple[str, ...]
    has_default: bool
    line: int
    column: int
    type: Type | None


class _CollectedRelationship(NamedTuple):
    name: str
    line: int
    column: int
    type: Type | None


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
    metadata = getattr(info, "metadata", None)
    md = metadata.get(METADATA_KEY) if isinstance(metadata, dict) else None
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
        base_metadata = getattr(base, "metadata", None)
        base_md = base_metadata.get(METADATA_KEY) if isinstance(base_metadata, dict) else None
        if isinstance(base_md, dict) and base_md.get("is_table_model") is True:
            return True
    return False


def _call_get_kwarg(call: CallExpr, name: str) -> Expression | None:
    for arg_name, arg_expr in zip(call.arg_names, call.args, strict=True):
        if arg_name == name:
            return arg_expr
    return None


def _call_get_positional(call: CallExpr, index: int) -> Expression | None:
    pos = 0
    for arg_name, arg_expr in zip(call.arg_names, call.args, strict=True):
        if arg_name is None:
            if pos == index:
                return arg_expr
            pos += 1
    return None


def _call_get_arg(call: CallExpr, name: str, positional_index: int) -> Expression | None:
    return _call_get_kwarg(call, name) or _call_get_positional(call, positional_index)


def _typeinfo_from_ref_expr(expr: Expression | None) -> TypeInfo | None:
    if expr is None:
        return None
    if isinstance(expr, RefExpr) and isinstance(expr.node, TypeInfo):
        return expr.node
    return None


def plugin(version: str) -> type[Plugin]:
    # `version` is the mypy version string; keep the signature required by mypy.
    return SQLModelMypyPlugin


def _iter_assignment_statements_from_if_statement(stmt: IfStmt) -> Iterator[AssignmentStmt]:
    for body in stmt.body:
        if not body.is_unreachable:
            yield from _iter_assignment_statements_from_block(body)
    if stmt.else_body is not None and not stmt.else_body.is_unreachable:
        yield from _iter_assignment_statements_from_block(stmt.else_body)


def _iter_assignment_statements_from_block(block: Block) -> Iterator[AssignmentStmt]:
    for stmt in block.body:
        if isinstance(stmt, AssignmentStmt):
            yield stmt
        elif isinstance(stmt, IfStmt):
            yield from _iter_assignment_statements_from_if_statement(stmt)


def _lookup_typeinfo(plugin: Plugin, fullname: str) -> TypeInfo | None:
    sym = plugin.lookup_fully_qualified(fullname)
    if sym and isinstance(sym.node, TypeInfo):
        return sym.node
    return None


def _plugin_any() -> AnyType:
    """Return an Any that should not trigger `disallow_any_explicit`."""
    return AnyType(TypeOfAny.implementation_artifact)


def _named_type_or_none(api: Any, fullname: str) -> Instance | None:
    """Best-effort `api.named_type`/`api.named_generic_type` wrapper.

    Prefer `named_type` (semantic analyzer) but fall back to `named_generic_type` (checker).
    """
    try:
        if hasattr(api, "named_type"):
            return cast(Instance, api.named_type(fullname))
        if hasattr(api, "named_generic_type"):
            return cast(Instance, api.named_generic_type(fullname, []))
    except Exception:
        return None
    return None


def _named_generic_type_or_none(api: Any, fullname: str, args: list[Type]) -> Instance | None:
    """Best-effort wrapper for generic named types across mypy plugin APIs."""
    try:
        if hasattr(api, "named_generic_type"):
            return cast(Instance, api.named_generic_type(fullname, args))
        if hasattr(api, "named_type"):
            return cast(Instance, api.named_type(fullname, args))
    except Exception:
        return None
    return None


def _plugins_from_config_file(config_file: str) -> list[str] | None:
    """Best-effort extraction of `[mypy] plugins = ...` from mypy config files."""
    toml_config = parse_toml(config_file)
    if toml_config is not None:
        plugins = toml_config.get("tool", {}).get("mypy", {}).get("plugins")
        if isinstance(plugins, str):
            return [p.strip() for p in plugins.split(",") if p.strip()]
        if isinstance(plugins, list) and all(isinstance(p, str) for p in plugins):
            return plugins
        return None

    parser = ConfigParser()
    parser.read(config_file)
    if not parser.has_section("mypy"):
        return None
    if not parser.has_option("mypy", "plugins"):
        return None
    raw = parser.get("mypy", "plugins")
    return [p.strip() for p in raw.split(",") if p.strip()]


class SQLModelMypyPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        self.plugin_config = SQLModelPluginConfig(options)
        self._plugin_data = {"__version__": __version__, **self.plugin_config.to_data()}
        self._plugin_order_ok = True
        self._reported_plugin_order = False
        self._warmed_sqlalchemy_typing = False
        plugins: list[str] | None = None
        if options.config_file is not None:
            plugins = _plugins_from_config_file(options.config_file)
        if plugins is None:
            raw_plugins = getattr(options, "plugins", None)
            if isinstance(raw_plugins, (list, tuple)):
                plugins = list(raw_plugins)
        if plugins is not None:
            try:
                sqlmodel_idx = plugins.index("sqlmodel_mypy.plugin")
                pydantic_idx = plugins.index("pydantic.mypy")
            except ValueError:
                pass
            else:
                self._plugin_order_ok = sqlmodel_idx < pydantic_idx
        super().__init__(options)

    def _warm_sqlalchemy_typing(self, api: Any) -> None:
        """Best-effort preloading of SQLAlchemy typing symbols.

        Some mypy plugin APIs only resolve types from already-loaded modules. Ensure
        core SQLAlchemy types are loaded before type-checking hooks run.
        """
        if self._warmed_sqlalchemy_typing:
            return
        self._warmed_sqlalchemy_typing = True

        any_t = _plugin_any()
        bool_t = api.named_type("builtins.bool")
        for fullname, args in (
            ("sqlalchemy.orm.attributes.InstrumentedAttribute", [any_t]),
            ("sqlalchemy.orm.InstrumentedAttribute", [any_t]),
            ("sqlalchemy.orm.attributes.QueryableAttribute", [any_t]),
            ("sqlalchemy.orm.base.Mapped", [any_t]),
            ("sqlalchemy.orm.Mapped", [any_t]),
            ("sqlalchemy.sql.elements.ColumnElement", [bool_t]),
            ("sqlalchemy.sql.schema.Table", None),
            ("sqlalchemy.sql.schema.TableClause", None),
            ("sqlalchemy.sql.selectable.FromClause", None),
        ):
            try:
                if args is None:
                    api.named_type(fullname)
                else:
                    api.named_type(fullname, args)
            except Exception:
                continue

    def get_function_signature_hook(
        self, fullname: str
    ) -> Callable[[FunctionSigContext], FunctionLike] | None:
        """Adjust function call signatures (SQLModel constructors and helpers)."""
        if fullname == SQLMODEL_FIELD_FULLNAME:
            return self._sqlmodel_field_signature_callback
        if fullname in SQLMODEL_SELECT_FULLNAMES:
            return self._sqlmodel_select_signature_callback

        info = _lookup_typeinfo(self, fullname)
        if info is None:
            return None
        if info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            return None
        if not info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return None

        def _hook(ctx: FunctionSigContext) -> FunctionLike:
            return self._sqlmodel_constructor_signature_callback(ctx, info)

        return _hook

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

    def get_method_signature_hook(
        self, fullname: str
    ) -> Callable[[MethodSigContext], FunctionLike] | None:
        if fullname == SQLMODEL_SESSION_EXEC_FULLNAME:
            return lambda ctx: self._sqlmodel_session_exec_signature_callback(ctx, is_async=False)
        if fullname == SQLMODEL_ASYNC_SESSION_EXEC_FULLNAME:
            return lambda ctx: self._sqlmodel_session_exec_signature_callback(ctx, is_async=True)
        if fullname.endswith(".model_construct"):
            return self._sqlmodel_model_construct_signature_callback
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        if fullname in {SQLALCHEMY_SELECT_JOIN_FULLNAME, SQLMODEL_SELECT_JOIN_FULLNAME}:
            return self._sqlalchemy_select_join_return_type_callback
        if fullname in {SQLALCHEMY_SELECT_JOIN_FROM_FULLNAME, SQLMODEL_SELECT_JOIN_FROM_FULLNAME}:
            return self._sqlalchemy_select_join_from_return_type_callback
        if fullname in {SQLALCHEMY_SELECT_OUTERJOIN_FULLNAME, SQLMODEL_SELECT_OUTERJOIN_FULLNAME}:
            return self._sqlalchemy_select_outerjoin_return_type_callback
        if fullname in {
            SQLALCHEMY_SELECT_OUTERJOIN_FROM_FULLNAME,
            SQLMODEL_SELECT_OUTERJOIN_FROM_FULLNAME,
        }:
            return self._sqlalchemy_select_outerjoin_from_return_type_callback
        if (
            fullname in SQLALCHEMY_RELATIONSHIP_COMPARATOR_METHOD_FULLNAMES
            # Robust fallback: different SQLAlchemy/mypy combinations may resolve
            # these methods against different bases.
            or fullname.endswith(".any")
            or fullname.endswith(".has")
            or fullname.endswith(".contains")
        ):
            return self._sqlalchemy_relationship_comparator_return_type_callback
        return None

    def get_class_attribute_hook(self, fullname: str) -> Callable[[AttributeContext], Type] | None:
        """Type SQLModel class attributes as SQLAlchemy expressions (e.g. `User.id`)."""
        if "." not in fullname:
            return None
        owner_fullname, attr_name = fullname.rsplit(".", 1)
        if attr_name == "model_config":
            return None
        if attr_name.startswith("_") and attr_name != "__table__":
            return None

        owner_info = _lookup_typeinfo(self, owner_fullname)
        if owner_info is None:
            return None
        if owner_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            return None
        if not owner_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return None
        if not _is_table_model(owner_info):
            return None

        if attr_name == "__table__":
            return lambda ctx: self._sqlalchemy_table_type(ctx.api)

        if not self._declares_sqlmodel_member(owner_info, attr_name):
            return None

        def _hook(ctx: AttributeContext) -> Type:
            return self._sqlmodel_class_attr_type_callback(ctx)

        return _hook

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if fullname == SQLMODEL_COL_FULLNAME:
            return self._sqlmodel_col_return_type_callback
        if fullname == BUILTINS_GETATTR_FULLNAME:
            return self._sqlmodel_getattr_return_type_callback
        if fullname in SQLALCHEMY_COLUMN_PROPERTY_FULLNAMES:
            return self._sqlalchemy_column_property_return_type_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> dict[str, Any]:
        # Used by mypy to decide whether to invalidate incremental caches.
        return self._plugin_data

    def _declares_sqlmodel_member(self, info: TypeInfo, name: str) -> bool:
        """Return True if `name` is declared as a field/relationship on `info`."""
        # Preferred: metadata-driven. This avoids relying on class-body AST shape at hook time.
        md = info.metadata.get(METADATA_KEY)
        if isinstance(md, dict):
            fields = md.get("fields")
            if isinstance(fields, dict) and name in fields:
                return True
            rels = md.get("relationships")
            if isinstance(rels, dict) and name in rels:
                return True
            if isinstance(rels, list) and name in rels:
                return True

        # Fast path: if it isn't even a Var on the class, it's not a model member.
        sym = info.names.get(name)
        if sym is None or sym.node is None or not isinstance(sym.node, Var) or sym.node.is_classvar:
            return False

        # Confirm it's declared in the class body (including in `if` blocks).
        cls_def = info.defn
        for stmt in _iter_assignment_statements_from_block(cls_def.defs):
            lhs = stmt.lvalues[0]
            if isinstance(lhs, NameExpr) and lhs.name == name:
                if not stmt.new_syntax:
                    return False
                return True
        return False

    def _collect_fields_for_signature(
        self, model_info: TypeInfo, api: Any
    ) -> list[_CollectedField]:
        fields, _relationships = self._collect_members_for_signature(model_info, api)
        return fields

    def _collect_members_for_signature(
        self, model_info: TypeInfo, api: Any
    ) -> tuple[list[_CollectedField], list[_CollectedRelationship]]:
        """Collect SQLModel members for signature generation (including inherited ones)."""
        found_fields: dict[str, _CollectedField] = {}
        found_relationships: dict[str, _CollectedRelationship] = {}

        def _add_field(field: _CollectedField) -> None:
            found_relationships.pop(field.name, None)
            found_fields[field.name] = field

        def _add_relationship(rel: _CollectedRelationship) -> None:
            found_fields.pop(rel.name, None)
            found_relationships[rel.name] = rel

        # 1) Inherited members (base first, mirroring semantic-phase logic).
        for base_info in reversed(model_info.mro[1:-1]):  # exclude current class and object
            if base_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
                continue
            if not base_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
                continue

            for stmt in _iter_assignment_statements_from_block(base_info.defn.defs):
                member = self._collect_member_from_stmt(
                    stmt,
                    defining_info=base_info,
                    current_info=model_info,
                    api=api,
                )
                if member is None:
                    continue
                if isinstance(member, _CollectedField):
                    sym_node = model_info.names.get(member.name)
                    if sym_node and sym_node.node and not isinstance(sym_node.node, Var):
                        api.fail(
                            "SQLModel field may only be overridden by another field",
                            sym_node.node,
                            code=ERROR_FIELD,
                        )
                        continue
                    _add_field(member)
                else:
                    _add_relationship(member)

        # 2) Current class members.
        for stmt in _iter_assignment_statements_from_block(model_info.defn.defs):
            member = self._collect_member_from_stmt(
                stmt,
                defining_info=model_info,
                current_info=model_info,
                api=api,
            )
            if member is None:
                continue
            if isinstance(member, _CollectedField):
                _add_field(member)
            else:
                _add_relationship(member)

        return list(found_fields.values()), list(found_relationships.values())

    def _collect_member_from_stmt(
        self,
        stmt: AssignmentStmt,
        *,
        defining_info: TypeInfo,
        current_info: TypeInfo,
        api: Any,
    ) -> _CollectedField | _CollectedRelationship | None:
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
                api.fail("Untyped fields disallowed", stmt, code=ERROR_FIELD)
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
            rel_type: Type | None = None
            sym = defining_info.names.get(name)
            if sym is not None and isinstance(sym.node, Var) and not sym.node.is_classvar:
                rel_type = sym.node.type
                if rel_type is not None and defining_info is not current_info:
                    with state.strict_optional_set(api.options.strict_optional):
                        rel_type = map_type_from_supertype(rel_type, current_info, defining_info)
            return _CollectedRelationship(
                name=name,
                line=stmt.line,
                column=stmt.column,
                type=rel_type,
            )

        sym = defining_info.names.get(name)
        if sym is None or sym.node is None:
            return None
        if not isinstance(sym.node, Var):
            return None
        if sym.node.is_classvar:
            return None

        init_type = sym.node.type
        has_default = SQLModelTransformer.get_has_default(stmt)
        aliases: tuple[str, ...] = tuple(SQLModelTransformer.get_field_aliases(stmt))

        # Prefer semantic-phase metadata when available (keeps checker-time fallback
        # consistent with plugin-generated signatures, including Annotated Field metadata).
        md = defining_info.metadata.get(METADATA_KEY)
        if isinstance(md, dict):
            fields = md.get("fields")
            if isinstance(fields, dict):
                data = fields.get(name)
                if isinstance(data, dict):
                    md_has_default = data.get("has_default")
                    if isinstance(md_has_default, bool):
                        has_default = md_has_default
                    md_aliases = data.get("aliases")
                    if isinstance(md_aliases, list):
                        aliases = tuple(a for a in md_aliases if isinstance(a, str))
        if init_type is not None and defining_info is not current_info:
            with state.strict_optional_set(api.options.strict_optional):
                init_type = map_type_from_supertype(init_type, current_info, defining_info)

        return _CollectedField(
            name=name,
            aliases=aliases,
            has_default=has_default,
            line=stmt.line,
            column=stmt.column,
            type=init_type,
        )

    def _sqlmodel_constructor_signature_callback(
        self, ctx: FunctionSigContext, info: TypeInfo
    ) -> FunctionLike:
        # Preferred: derive the class-call signature from the plugin-generated `__init__`.
        # This avoids re-scanning the class body AST at type-check time (which is brittle,
        # especially with compiled mypy) and matches what users see in `reveal_type(Model.__init__)`.
        init_sym = info.names.get("__init__")
        if init_sym is not None and init_sym.plugin_generated:
            init_node = init_sym.node
            init_type: CallableType | None = None
            if isinstance(init_node, Var):
                if isinstance(init_node.type, CallableType):
                    init_type = init_node.type
            elif isinstance(init_node, FuncDef):
                if isinstance(init_node.type, CallableType):
                    init_type = init_node.type
            elif isinstance(init_node, Decorator):
                if isinstance(init_node.func.type, CallableType):
                    init_type = init_node.func.type

            if init_type is not None and len(init_type.arg_types) >= 1:
                return init_type.copy_modified(
                    arg_types=init_type.arg_types[1:],
                    arg_kinds=init_type.arg_kinds[1:],
                    arg_names=init_type.arg_names[1:],
                    ret_type=ctx.default_signature.ret_type,
                    fallback=ctx.default_signature.fallback,
                )

        fields, relationships = self._collect_members_for_signature(info, ctx.api)

        arg_types: list[Type] = []
        arg_names: list[str | None] = []
        arg_kinds: list[ArgKind] = []

        typed = self.plugin_config.init_typed
        table_model = _is_table_model(info)

        canonical_names = {f.name for f in fields}
        if table_model:
            canonical_names.update(rel.name for rel in relationships)
        if not self.plugin_config.init_forbid_extra:
            canonical_names.add("kwargs")

        field_params: list[tuple[Type, list[str]]] = []
        for f in fields:
            if typed and f.type is not None:
                t = f.type.accept(ForceInvariantTypeVars())
                t = _unwrap_mapped_type(t)
            else:
                t = _plugin_any()
            field_aliases = [
                alias for alias in f.aliases if alias != f.name and alias not in canonical_names
            ]
            arg_types.append(t)
            arg_names.append(f.name)
            arg_kinds.append(ARG_NAMED_OPT if f.has_default or field_aliases else ARG_NAMED)
            field_params.append((t, field_aliases))

        if table_model:
            for rel in relationships:
                if typed and rel.type is not None:
                    t = rel.type.accept(ForceInvariantTypeVars())
                    t = _unwrap_mapped_type(t)
                else:
                    t = _plugin_any()
                arg_types.append(t)
                arg_names.append(rel.name)
                arg_kinds.append(ARG_NAMED_OPT)

        used_names = set(canonical_names)
        for t, field_aliases in field_params:
            for alias in field_aliases:
                if alias in used_names:
                    continue
                used_names.add(alias)
                arg_types.append(t)
                arg_names.append(alias)
                arg_kinds.append(ARG_NAMED_OPT)

        if not self.plugin_config.init_forbid_extra:
            kw = _plugin_any()
            arg_types.append(kw)
            arg_names.append("kwargs")
            arg_kinds.append(ARG_STAR2)

        # Preserve mypy's inferred return type for the class call.
        ret_type = ctx.default_signature.ret_type
        fallback = ctx.default_signature.fallback
        return CallableType(arg_types, arg_kinds, arg_names, ret_type, fallback)

    def _sqlmodel_field_signature_callback(self, ctx: FunctionSigContext) -> FunctionLike:
        """Widen `Field(sa_type=...)` to accept TypeEngine instances.

        SQLAlchemy `Column` accepts both a TypeEngine class and a TypeEngine instance (e.g.
        `DateTime(timezone=True)`, `String(50)`), but SQLModel's stub history has not always
        reflected that. This hook keeps behavior idempotent when upstream typing is already
        widened.
        """
        default = ctx.default_signature
        if not isinstance(default, CallableType):
            return default

        try:
            sa_type_index = default.arg_names.index("sa_type")
        except ValueError:
            return default

        old_sa_type = default.arg_types[sa_type_index]
        proper_old_sa_type = get_proper_type(old_sa_type)
        if isinstance(proper_old_sa_type, AnyType):
            return default

        type_engine_info = _lookup_typeinfo(self, "sqlalchemy.sql.type_api.TypeEngine")
        if type_engine_info is None:
            return default

        def _contains_instance_fullname(tp: Type, fullname: str) -> bool:
            proper = get_proper_type(tp)
            if isinstance(proper, Instance):
                return proper.type.fullname == fullname
            if isinstance(proper, UnionType):
                return any(_contains_instance_fullname(item, fullname) for item in proper.items)
            return False

        # Idempotency: if upstream stubs already accept TypeEngine instances, keep unchanged.
        if _contains_instance_fullname(old_sa_type, type_engine_info.fullname):
            return default

        any_t = _plugin_any()
        type_engine_args = [any_t] * len(type_engine_info.defn.type_vars)
        type_engine_instance = Instance(type_engine_info, type_engine_args)

        new_arg_types = list(default.arg_types)
        new_arg_types[sa_type_index] = UnionType.make_union([old_sa_type, type_engine_instance])
        return default.copy_modified(arg_types=new_arg_types)

    def _sqlmodel_select_signature_callback(self, ctx: FunctionSigContext) -> FunctionLike:
        """Avoid `select()` overload ceiling by adding a 5+ args fallback.

        SQLModel's generated stubs traditionally cap `select()` overloads at 4 entities.
        For 5+ entities, fall back to a conservative signature that accepts the call and
        returns `Select[tuple[Any, ...]]` (represented as a fixed-length tuple of `Any`s).
        """
        call = ctx.context
        if not isinstance(call, CallExpr):
            return ctx.default_signature

        # Only handle straightforward positional calls; don't change behavior for
        # `*args` / `**kwargs` forwarding or named arguments.
        if any(k in {ARG_STAR, ARG_STAR2} for k in call.arg_kinds):
            return ctx.default_signature
        if any(name is not None for name in call.arg_names):
            return ctx.default_signature
        if any(k != ARG_POS for k in call.arg_kinds):
            return ctx.default_signature

        positional_count = len(call.arg_kinds)
        if positional_count <= 4:
            return ctx.default_signature

        # Idempotency: if upstream stubs already accept 5+ entities (e.g. via a varargs
        # overload), keep the default signature unchanged.
        def _callable_supports_n_positional_args(sig: CallableType, n: int) -> bool:
            kinds = sig.arg_kinds
            if ARG_STAR in kinds:
                star_index = kinds.index(ARG_STAR)
                required = sum(1 for k in kinds[:star_index] if k == ARG_POS)
                return n >= required
            if len(kinds) < n:
                return False
            return all(k in {ARG_POS, ARG_OPT} for k in kinds[:n])

        try:
            default_items = ctx.default_signature.items
        except Exception:
            return ctx.default_signature
        if any(
            isinstance(item, CallableType)
            and _callable_supports_n_positional_args(item, positional_count)
            for item in default_items
        ):
            return ctx.default_signature

        select_info = _lookup_typeinfo(self, SQLALCHEMY_SELECT_FULLNAME)
        if select_info is None:
            return ctx.default_signature

        any_t = _plugin_any()
        tuple_fallback = ctx.api.named_generic_type("builtins.tuple", [any_t])
        row_type = TupleType([any_t] * positional_count, tuple_fallback)
        ret_type = Instance(select_info, [row_type])

        fallback = default_items[0].fallback if default_items else tuple_fallback
        arg_types = [any_t] * positional_count
        arg_kinds: list[ArgKind] = [ARG_POS] * positional_count
        arg_names = [f"__ent{i}" for i in range(positional_count)]
        return CallableType(arg_types, arg_kinds, arg_names, ret_type, fallback)

    def _sqlmodel_model_construct_signature_callback(self, ctx: MethodSigContext) -> FunctionLike:
        receiver_info: TypeInfo | None = None
        receiver_instance: Instance | None = None

        ctx_type = ctx.type
        if isinstance(ctx_type, TypeType):
            item = get_proper_type(ctx_type.item)
            if isinstance(item, Instance):
                receiver_instance = item
                receiver_info = item.type
        elif isinstance(ctx_type, Instance):
            receiver_instance = ctx_type
            receiver_info = ctx_type.type

        if receiver_info is None:
            return ctx.default_signature
        if receiver_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            return ctx.default_signature
        if not receiver_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return ctx.default_signature

        fields, relationships = self._collect_members_for_signature(receiver_info, ctx.api)

        set_str = ctx.api.named_generic_type(
            "builtins.set", [ctx.api.named_generic_type("builtins.str", [])]
        )
        optional_set_str: Type = UnionType([set_str, NoneType()])

        arg_types: list[Type] = [optional_set_str]
        arg_names: list[str | None] = ["_fields_set"]
        arg_kinds: list[ArgKind] = [ARG_OPT]

        table_model = _is_table_model(receiver_info)
        canonical_names = {f.name for f in fields}
        if table_model:
            canonical_names.update(rel.name for rel in relationships)
        canonical_names.add("_fields_set")
        if not self.plugin_config.init_forbid_extra:
            canonical_names.add("kwargs")

        field_params: list[tuple[Type, list[str]]] = []
        for f in fields:
            t = f.type.accept(ForceInvariantTypeVars()) if f.type is not None else _plugin_any()
            t = _unwrap_mapped_type(t)
            field_aliases = [
                alias for alias in f.aliases if alias != f.name and alias not in canonical_names
            ]
            arg_types.append(t)
            arg_names.append(f.name)
            arg_kinds.append(ARG_NAMED_OPT if f.has_default or field_aliases else ARG_NAMED)
            field_params.append((t, field_aliases))

        if table_model:
            for rel in relationships:
                t = (
                    rel.type.accept(ForceInvariantTypeVars())
                    if rel.type is not None
                    else _plugin_any()
                )
                t = _unwrap_mapped_type(t)
                arg_types.append(t)
                arg_names.append(rel.name)
                arg_kinds.append(ARG_NAMED_OPT)

        used_names = set(canonical_names)
        for t, field_aliases in field_params:
            for alias in field_aliases:
                if alias in used_names:
                    continue
                used_names.add(alias)
                arg_types.append(t)
                arg_names.append(alias)
                arg_kinds.append(ARG_NAMED_OPT)

        if not self.plugin_config.init_forbid_extra:
            kw = _plugin_any()
            arg_types.append(kw)
            arg_names.append("kwargs")
            arg_kinds.append(ARG_STAR2)

        ret_type: Type = receiver_instance or fill_typevars(receiver_info)
        fallback = ctx.default_signature.fallback
        return CallableType(arg_types, arg_kinds, arg_names, ret_type, fallback)

    def _sqlmodel_session_exec_signature_callback(
        self, ctx: MethodSigContext, *, is_async: bool
    ) -> FunctionLike:
        """Broaden `Session.exec()` accepted statement types.

        SQLModel's stubs accept `sqlmodel.sql.base.Executable[...]`, but many common SQLAlchemy statements
        (e.g. `text(...)`) are only `sqlalchemy.sql.base.Executable`, causing mypy failures.
        """

        # NOTE: mypy applies method signature hooks *per overload item*; when the original
        # method is `Overloaded`, returning `Overloaded` here will trigger mypy internal errors
        # (it expects a `CallableType` per item). Keep this hook returning `CallableType`.
        default = ctx.default_signature
        if not isinstance(default, CallableType):
            return default
        if not default.arg_types:
            return default

        executable_info = _lookup_typeinfo(self, "sqlalchemy.sql.base.Executable")
        if executable_info is None:
            return default
        sqlalchemy_executable = Instance(executable_info, [])

        # Depending on SQLModel/SQLAlchemy versions, the "broad" overload may be expressed via
        # `sqlmodel.sql.base.Executable[T]` or `sqlalchemy.sql.dml.UpdateBase` (DML statements).
        sqlmodel_executable_info = _lookup_typeinfo(self, "sqlmodel.sql.base.Executable")
        update_base_info = _lookup_typeinfo(self, "sqlalchemy.sql.dml.UpdateBase")
        if sqlmodel_executable_info is None and update_base_info is None:
            return default

        def _contains_instance_fullname(tp: Type, fullname: str) -> bool:
            proper = get_proper_type(tp)
            if isinstance(proper, Instance):
                return proper.type.fullname == fullname
            if isinstance(proper, UnionType):
                return any(_contains_instance_fullname(item, fullname) for item in proper.items)
            return False

        # Only adjust the broad overload; keep Select/SelectOfScalar overloads precise.
        broad = False
        if sqlmodel_executable_info is not None and _contains_instance_fullname(
            default.arg_types[0], sqlmodel_executable_info.fullname
        ):
            broad = True
        if update_base_info is not None and _contains_instance_fullname(
            default.arg_types[0], update_base_info.fullname
        ):
            broad = True
        if not broad:
            return default

        # Idempotency: if upstream stubs already accept SQLAlchemy Executable, keep unchanged.
        if _contains_instance_fullname(default.arg_types[0], executable_info.fullname):
            return default

        new_arg_types = list(default.arg_types)
        new_arg_types[0] = UnionType.make_union([default.arg_types[0], sqlalchemy_executable])
        return default.copy_modified(arg_types=new_arg_types)

    def _sqlalchemy_select_join_return_type_callback(self, ctx: MethodContext) -> Type:
        # join(target, onclause=None, *, isouter: bool = False, full: bool = False)
        return self._sqlalchemy_select_join_like_return_type(
            ctx, target_positional_index=0, isouter_always=False
        )

    def _sqlalchemy_select_join_from_return_type_callback(self, ctx: MethodContext) -> Type:
        # join_from(from_, target, onclause=None, *, isouter: bool = False, full: bool = False)
        return self._sqlalchemy_select_join_like_return_type(
            ctx, target_positional_index=1, isouter_always=False
        )

    def _sqlalchemy_select_outerjoin_return_type_callback(self, ctx: MethodContext) -> Type:
        # outerjoin(target, onclause=None, *, full: bool = False)
        return self._sqlalchemy_select_join_like_return_type(
            ctx, target_positional_index=0, isouter_always=True
        )

    def _sqlalchemy_select_outerjoin_from_return_type_callback(self, ctx: MethodContext) -> Type:
        # outerjoin_from(from_, target, onclause=None, *, full: bool = False)
        return self._sqlalchemy_select_join_like_return_type(
            ctx, target_positional_index=1, isouter_always=True
        )

    def _sqlalchemy_relationship_comparator_return_type_callback(self, ctx: MethodContext) -> Type:
        """Ensure relationship comparator calls return a SQL boolean expression.

        SQLAlchemy's typing for relationship comparator helpers varies across versions and stubs.
        For SQLModel relationship attributes (declared via `Relationship(...)`), prefer returning
        `ColumnElement[bool]` when mypy would otherwise fall back to `Any` / `bool`.
        """
        call = ctx.context
        if not isinstance(call, CallExpr):
            return ctx.default_return_type

        # Identify `Model.relationship.<method>(...)` calls so we only affect SQLModel relationships.
        callee = call.callee
        if not isinstance(callee, MemberExpr):
            return ctx.default_return_type

        receiver = callee.expr
        if not isinstance(receiver, MemberExpr):
            return ctx.default_return_type

        rel_name = receiver.name
        owner_info = _typeinfo_from_ref_expr(receiver.expr)
        if owner_info is None:
            return ctx.default_return_type
        if not owner_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return ctx.default_return_type
        if not _is_table_model(owner_info):
            return ctx.default_return_type

        try:
            _fields, relationships = self._collect_members_for_signature(owner_info, ctx.api)
        except Exception:
            return ctx.default_return_type

        if rel_name not in {r.name for r in relationships}:
            return ctx.default_return_type

        default = ctx.default_return_type
        proper_default = get_proper_type(default)
        if isinstance(proper_default, AnyType):
            return self._sqlalchemy_column_element_bool_type(ctx)
        if isinstance(proper_default, Instance) and proper_default.type.fullname == "builtins.bool":
            return self._sqlalchemy_column_element_bool_type(ctx)
        return default

    @staticmethod
    def _sqlalchemy_column_element_bool_type(ctx: MethodContext) -> Type:
        """Return `sqlalchemy.sql.elements.ColumnElement[bool]` or `Any` as a fallback."""
        bool_t = ctx.api.named_generic_type("builtins.bool", [])
        try:
            return ctx.api.named_generic_type("sqlalchemy.sql.elements.ColumnElement", [bool_t])
        except Exception:
            return _plugin_any()

    def _typeinfo_from_join_target_expr(self, expr: Expression | None, api: Any) -> TypeInfo | None:
        """Return a best-effort `TypeInfo` for a join target expression.

        Supports direct model-class targets (e.g. `join(Team)`) and relationship
        attribute targets (e.g. `join(Hero.team)`).
        """
        info = _typeinfo_from_ref_expr(expr)
        if info is not None:
            return info
        return self._typeinfo_from_relationship_join_target(expr, api)

    def _typeinfo_from_relationship_join_target(
        self, expr: Expression | None, api: Any
    ) -> TypeInfo | None:
        """Infer join target model type from `Model.relationship` expressions."""
        if not isinstance(expr, MemberExpr):
            return None

        owner_info = _typeinfo_from_ref_expr(expr.expr)
        if owner_info is None:
            return None
        if owner_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            return None
        if not owner_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return None
        if not _is_table_model(owner_info):
            return None

        rel_name = expr.name
        try:
            _fields, relationships = self._collect_members_for_signature(owner_info, api)
        except Exception:
            return None

        rel_type: Type | None = None
        for rel in relationships:
            if rel.name == rel_name:
                rel_type = rel.type
                break
        if rel_type is None:
            return None

        entity_type = self._unwrap_relationship_entity_type(rel_type)
        if entity_type is None:
            return None
        proper = get_proper_type(entity_type)
        if not isinstance(proper, Instance):
            return None

        target_info = proper.type
        if not target_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return None
        return target_info

    @classmethod
    def _unwrap_relationship_entity_type(cls, typ: Type) -> Type | None:
        """Return the related entity type for relationship annotations.

        Examples:
        - `Team | None` -> `Team`
        - `list[Hero]` -> `Hero`
        - `Mapped[list[Hero]]` -> `Hero`
        """
        typ = _unwrap_mapped_type(typ)
        proper = get_proper_type(typ)

        if isinstance(proper, UnionType):
            non_none: list[Type] = []
            for item in proper.items:
                if isinstance(get_proper_type(item), NoneType):
                    continue
                non_none.append(item)
            if len(non_none) != 1:
                return None
            return cls._unwrap_relationship_entity_type(non_none[0])

        if isinstance(proper, Instance):
            # Relationship types may appear wrapped as ORM descriptor/expression types.
            if proper.type.fullname in SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES and proper.args:
                return cls._unwrap_relationship_entity_type(proper.args[0])
            if (
                proper.type.fullname == "sqlalchemy.orm.attributes.QueryableAttribute"
                and proper.args
            ):
                return cls._unwrap_relationship_entity_type(proper.args[0])

            # One-to-many / many-to-many relationships are commonly annotated as `list[T]`.
            if proper.args and proper.type.fullname in {
                "builtins.list",
                "builtins.set",
                "builtins.frozenset",
                "collections.abc.Sequence",
                "typing.Sequence",
                "typing.List",
                "typing.Set",
                "typing.FrozenSet",
            }:
                return cls._unwrap_relationship_entity_type(proper.args[0])
            return typ

        return None

    def _sqlalchemy_select_join_like_return_type(
        self, ctx: MethodContext, *, target_positional_index: int, isouter_always: bool
    ) -> Type:
        # Only handle "outer join" for now, and only when it's explicit/literal.
        call = ctx.context
        if not isinstance(call, CallExpr):
            return ctx.default_return_type

        if not isouter_always:
            isouter_expr = _call_get_kwarg(call, "isouter")
            if isouter_expr is None:
                return ctx.default_return_type
            if not _is_bool_nameexpr(isouter_expr, True):
                return ctx.default_return_type

        target_expr = _call_get_arg(call, "target", target_positional_index)
        target_info = self._typeinfo_from_join_target_expr(target_expr, ctx.api)
        if target_info is None:
            return ctx.default_return_type

        receiver = get_proper_type(ctx.type)
        if not isinstance(receiver, Instance):
            return ctx.default_return_type
        if receiver.type.fullname not in {
            SQLALCHEMY_SELECT_FULLNAME,
            SQLMODEL_SELECT_CLS_FULLNAME,
        } and not receiver.type.has_base(SQLALCHEMY_SELECT_FULLNAME):
            return ctx.default_return_type
        if not receiver.args:
            return ctx.default_return_type

        tp = get_proper_type(receiver.args[0])
        if not isinstance(tp, TupleType):
            return ctx.default_return_type

        changed = False
        new_items: list[Type] = []
        for item in tp.items:
            proper_item = get_proper_type(item)
            if (
                isinstance(proper_item, Instance)
                and proper_item.type.fullname == target_info.fullname
            ):
                new_items.append(UnionType.make_union([item, NoneType()]))
                changed = True
            else:
                new_items.append(item)

        if not changed:
            return ctx.default_return_type

        new_tp = tp.copy_modified(items=new_items)
        return Instance(receiver.type, [new_tp])

    def _sqlmodel_class_attr_type_callback(self, ctx: AttributeContext) -> Type:
        if ctx.is_lvalue:
            return ctx.default_attr_type

        default = ctx.default_attr_type
        proper = get_proper_type(default)

        if isinstance(proper, Instance):
            if proper.type.fullname in SQLALCHEMY_MAPPED_FULLNAMES:
                return default
            if proper.type.fullname in SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES:
                return default

        for fullname in (
            "sqlalchemy.orm.attributes.InstrumentedAttribute",
            "sqlalchemy.orm.InstrumentedAttribute",
        ):
            info = _lookup_typeinfo(self, fullname)
            if info is not None:
                return Instance(info, [default])

        # Fallback: if ORM attribute types aren't available, prefer a Core expression wrapper.
        # This still enables common operator methods like `.in_(...)` and `.like(...)`.
        for fullname in (
            "sqlalchemy.sql.elements.ColumnElement",
            "sqlalchemy.sql.expression.ColumnElement",
        ):
            info = _lookup_typeinfo(self, fullname)
            if info is not None:
                return Instance(info, [default])
        return default

    def _sqlmodel_col_return_type_callback(self, ctx: FunctionContext) -> Type:
        # `col()` is effectively a typed identity/cast helper.
        if not ctx.arg_types or not ctx.arg_types[0]:
            return ctx.default_return_type

        arg0 = get_proper_type(ctx.arg_types[0][0])
        value_type: Type | None = None

        if isinstance(arg0, Instance):
            if arg0.type.fullname in SQLALCHEMY_MAPPED_FULLNAMES and arg0.args:
                value_type = arg0.args[0]
            elif arg0.type.fullname in SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES and arg0.args:
                value_type = arg0.args[0]

        if value_type is None:
            return ctx.default_return_type

        for fullname in ("sqlalchemy.orm.base.Mapped", "sqlalchemy.orm.Mapped"):
            info = _lookup_typeinfo(self, fullname)
            if info is not None:
                return Instance(info, [value_type])
        return ctx.default_return_type

    @staticmethod
    def _lookup_var_in_mro(info: TypeInfo, name: str) -> Var | None:
        """Find a `Var` named `name` in `info` or its MRO (best-effort)."""
        for base in info.mro:
            sym = base.names.get(name)
            if sym is not None and isinstance(sym.node, Var):
                return sym.node
        return None

    @staticmethod
    def _is_sqlalchemy_expressionish(tp: Type) -> bool:
        """Return True if `tp` already looks like a SQLAlchemy expression/descriptor type."""
        proper = get_proper_type(tp)
        if not isinstance(proper, Instance):
            return False
        if proper.type.fullname in SQLALCHEMY_MAPPED_FULLNAMES:
            return True
        if proper.type.fullname in SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES:
            return True
        if proper.type.fullname == "sqlalchemy.orm.attributes.QueryableAttribute":
            return True
        if proper.type.fullname in {
            "sqlalchemy.sql.elements.ColumnElement",
            "sqlalchemy.sql.expression.ColumnElement",
        }:
            return True
        return False

    def _sqlalchemy_instrumented_attribute_type(self, api: Any, value_type: Type) -> Type | None:
        for fullname in (
            "sqlalchemy.orm.attributes.InstrumentedAttribute",
            "sqlalchemy.orm.InstrumentedAttribute",
        ):
            try:
                return cast(Type, api.named_generic_type(fullname, [value_type]))
            except Exception:
                continue
        return None

    def _sqlalchemy_column_element_type(self, api: Any, value_type: Type) -> Type | None:
        for fullname in (
            "sqlalchemy.sql.elements.ColumnElement",
            "sqlalchemy.sql.expression.ColumnElement",
        ):
            try:
                return cast(Type, api.named_generic_type(fullname, [value_type]))
            except Exception:
                continue
        return None

    def _sqlalchemy_expr_type_for_class_attr(self, api: Any, declared_type: Type) -> Type:
        """Return the type of `Model.attr` in SQL expressions, given declared instance type.

        This mirrors `_sqlmodel_class_attr_type_callback` but works from a `Type`.
        """
        proper = get_proper_type(declared_type)
        if isinstance(proper, Instance):
            # If the attribute is already typed as an ORM descriptor/expression, keep it.
            if proper.type.fullname in SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES:
                return declared_type
            if proper.type.fullname == "sqlalchemy.orm.attributes.QueryableAttribute":
                return declared_type
            if proper.type.fullname in {
                "sqlalchemy.sql.elements.ColumnElement",
                "sqlalchemy.sql.expression.ColumnElement",
            }:
                return declared_type

            # `Mapped[T]` is a descriptor; class-level access behaves like an instrumented attribute.
            if proper.type.fullname in SQLALCHEMY_MAPPED_FULLNAMES and proper.args:
                value_type = proper.args[0]
                inst = self._sqlalchemy_instrumented_attribute_type(api, value_type)
                if inst is not None:
                    return inst

        inst = self._sqlalchemy_instrumented_attribute_type(api, declared_type)
        if inst is not None:
            return inst
        col = self._sqlalchemy_column_element_type(api, declared_type)
        if col is not None:
            return col
        return declared_type

    def _sqlmodel_getattr_return_type_callback(self, ctx: FunctionContext) -> Type:
        """Type `getattr(Model, \"field\")` like `Model.field` for SQLModel table models."""
        if not ctx.args or len(ctx.args) < 2:
            return ctx.default_return_type
        if not ctx.args[0] or not ctx.args[1]:
            return ctx.default_return_type

        name_expr = ctx.args[1][0]
        if not isinstance(name_expr, StrExpr):
            return ctx.default_return_type
        attr_name = name_expr.value

        if not ctx.arg_types or not ctx.arg_types[0]:
            return ctx.default_return_type
        obj_type = get_proper_type(ctx.arg_types[0][0])

        owner_info: TypeInfo | None = None
        if isinstance(obj_type, TypeType):
            item = get_proper_type(obj_type.item)
            if isinstance(item, Instance):
                owner_info = item.type
        if owner_info is None:
            return ctx.default_return_type
        if owner_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            return ctx.default_return_type
        if not owner_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return ctx.default_return_type
        if not _is_table_model(owner_info):
            return ctx.default_return_type

        # Mirror `Model.__table__` support.
        if attr_name == "__table__":
            return self._sqlalchemy_table_type(ctx.api)

        # Prefer SQLModel members (fields/relationships): treat as SQLAlchemy expressions.
        var = self._lookup_var_in_mro(owner_info, attr_name)
        if (
            var is not None
            and var.type is not None
            and self._declares_sqlmodel_member(owner_info, attr_name)
        ):
            tp = self._sqlalchemy_expr_type_for_class_attr(ctx.api, var.type)
        # For non-SQLModel attributes, only special-case SQLAlchemy expression-like descriptors.
        elif (
            var is not None and var.type is not None and self._is_sqlalchemy_expressionish(var.type)
        ):
            tp = self._sqlalchemy_expr_type_for_class_attr(ctx.api, var.type)
        else:
            return ctx.default_return_type

        # If a default value is provided, union it in (best-effort).
        if len(ctx.arg_types) >= 3 and ctx.arg_types[2]:
            default_t = ctx.arg_types[2][0]
            return UnionType.make_union([tp, default_t])
        return tp

    def _sqlalchemy_mapped_type(self, api: Any, value_type: Type) -> Type | None:
        for fullname in ("sqlalchemy.orm.base.Mapped", "sqlalchemy.orm.Mapped"):
            try:
                return cast(Type, api.named_generic_type(fullname, [value_type]))
            except Exception:
                continue
        return None

    def _sqlalchemy_column_property_return_type_callback(self, ctx: FunctionContext) -> Type:
        """Best-effort typing for `sqlalchemy.orm.column_property(...)`.

        This is primarily used to support patterns like:
            `Model._ticketscount = column_property(select(...).scalar_subquery())`
        where users later want `getattr(Model, "_ticketscount")` to behave like an ORM attribute.
        """
        default = ctx.default_return_type
        if not ctx.arg_types or not ctx.arg_types[0]:
            return default

        # If SQLAlchemy stubs already provide a precise mapped type, keep it.
        proper_default = get_proper_type(default)
        if (
            isinstance(proper_default, Instance)
            and proper_default.type.fullname in SQLALCHEMY_MAPPED_FULLNAMES
            and proper_default.args
            and not isinstance(get_proper_type(proper_default.args[0]), AnyType)
        ):
            return default

        # Infer `T` from the first argument type.
        arg0 = get_proper_type(ctx.arg_types[0][0])
        value_type: Type | None = None
        if isinstance(arg0, Instance) and arg0.args:
            if arg0.type.fullname in {
                "sqlalchemy.sql.selectable.ScalarSelect",
                "sqlalchemy.sql.expression.ScalarSelect",
                "sqlalchemy.sql.selectable.ScalarSubquery",
            }:
                value_type = arg0.args[0]
            if arg0.type.fullname in {
                "sqlalchemy.sql.elements.ColumnElement",
                "sqlalchemy.sql.expression.ColumnElement",
            }:
                value_type = arg0.args[0]

        if value_type is None:
            return default

        mapped = self._sqlalchemy_mapped_type(ctx.api, value_type)
        return mapped or default

    def _sqlmodel_model_class_callback(self, ctx: ClassDefContext) -> None:
        # SQLModel is built on Pydantic v2; users commonly override `model_config`
        # with `ConfigDict(...)`. SQLModel's own config typing may be narrower
        # than Pydantic's `ConfigDict` and can trigger strict-mode assignment
        # errors. Widen to a compatible supertype so `model_config = ConfigDict(...)`
        # is accepted without `# type: ignore`.
        self._widen_model_config_type(ctx)
        self._warm_sqlalchemy_typing(ctx.api)
        transformer = SQLModelTransformer(ctx.cls, ctx.reason, ctx.api, self.plugin_config)
        transformer.transform()
        self._add_table_dunders(ctx)

    def _add_table_dunders(self, ctx: ClassDefContext) -> None:
        info = ctx.cls.info
        if not _is_table_model(info):
            return

        if "__table__" in info.names:
            return

        table_t = self._sqlalchemy_table_type(ctx.api)
        v = Var("__table__", table_t)
        v.info = info
        v._fullname = f"{info.fullname}.__table__"
        v.is_classvar = True
        sym = SymbolTableNode(MDEF, v)
        sym.plugin_generated = True
        info.names["__table__"] = sym

    def _sqlalchemy_table_type(self, api: Any | None = None) -> Type:
        """Return the best available `sqlalchemy` Table-ish type.

        Prefer `sqlalchemy.sql.schema.Table`, but fall back to broader interfaces if the
        installed SQLAlchemy stubs don't expose it.
        """
        for fullname in (
            "sqlalchemy.sql.schema.Table",
            "sqlalchemy.sql.schema.TableClause",
            "sqlalchemy.sql.selectable.FromClause",
        ):
            if api is not None:
                inst = _named_type_or_none(api, fullname)
                if inst is not None:
                    return inst
            info = _lookup_typeinfo(self, fullname)
            if info is not None:
                return Instance(info, [])
        return _plugin_any()

    def _widen_model_config_type(self, ctx: ClassDefContext) -> None:
        mapping_str_any = ctx.api.named_type(
            "typing.Mapping",
            [ctx.api.named_type("builtins.str"), _plugin_any()],
        )

        # Widen the base SQLModel attribute (used as the expected type for overrides).
        base_info = _lookup_typeinfo(self, SQLMODEL_BASEMODEL_FULLNAME)
        if base_info is not None:
            sym = base_info.names.get("model_config")
            if sym is not None and isinstance(sym.node, Var):
                sym.node.type = mapping_str_any

        # Also widen the current class attribute, if declared in this class body.
        sym = ctx.cls.info.names.get("model_config")
        if sym is not None and isinstance(sym.node, Var):
            sym.node.type = mapping_str_any

    def _sqlmodel_metaclass_callback(self, ctx: ClassDefContext) -> None:
        """Disable dataclass-transform handling for SQLModel metaclass.

        SQLModel decorates `SQLModelMetaclass` with `__dataclass_transform__`, but we want this plugin
        (not mypy's generic dataclass-transform logic) to generate the model signatures.
        """
        if not self._plugin_order_ok and not self._reported_plugin_order:
            self._reported_plugin_order = True
            ctx.api.fail(
                "Plugin order matters: list 'sqlmodel_mypy.plugin' before 'pydantic.mypy' "
                "(see sqlmodel-mypy-plugin README). With 'pydantic.mypy' first, mypy can "
                "claim SQLModel classes as plain Pydantic models and SQLModel typing becomes incorrect.",
                ctx.cls,
                code=ERROR_PLUGIN_ORDER,
            )
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
