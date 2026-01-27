"""Mypy plugin entry point for SQLModel."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from configparser import ConfigParser
from typing import Any, NamedTuple

from mypy.nodes import (
    ARG_NAMED,
    ARG_NAMED_OPT,
    ARG_OPT,
    ARG_STAR2,
    ArgKind,
    AssignmentStmt,
    Block,
    CallExpr,
    IfStmt,
    NameExpr,
    TypeInfo,
    Var,
)
from mypy.options import Options
from mypy.plugin import (
    AttributeContext,
    ClassDefContext,
    FunctionContext,
    FunctionSigContext,
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
    Type,
    TypeOfAny,
    TypeType,
    UnionType,
    get_proper_type,
)
from mypy.typevars import fill_typevars

from .transform import (
    ERROR_FIELD,
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

SQLMODEL_COL_FULLNAME = "sqlmodel.sql.expression.col"

SQLALCHEMY_INSTRUMENTED_ATTRIBUTE_FULLNAMES = {
    "sqlalchemy.orm.InstrumentedAttribute",
    "sqlalchemy.orm.attributes.InstrumentedAttribute",
}

# Increment when plugin changes should invalidate mypy cache.
__version__ = 3


class _CollectedField(NamedTuple):
    name: str
    has_default: bool
    line: int
    column: int
    type: Type | None


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


class SQLModelMypyPlugin(Plugin):
    def __init__(self, options: Options) -> None:
        self.plugin_config = SQLModelPluginConfig(options)
        self._plugin_data = {"__version__": __version__, **self.plugin_config.to_data()}
        super().__init__(options)

    def get_function_signature_hook(
        self, fullname: str
    ) -> Callable[[FunctionSigContext], FunctionLike] | None:
        """Make SQLModel constructors correct even if `pydantic.mypy` runs first."""
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
        if fullname.endswith(".model_construct"):
            return self._sqlmodel_model_construct_signature_callback
        return None

    def get_class_attribute_hook(self, fullname: str) -> Callable[[AttributeContext], Type] | None:
        """Type SQLModel class attributes as SQLAlchemy expressions (e.g. `User.id`)."""
        if "." not in fullname:
            return None
        owner_fullname, attr_name = fullname.rsplit(".", 1)
        if attr_name == "model_config" or attr_name.startswith("_"):
            return None

        owner_info = _lookup_typeinfo(self, owner_fullname)
        if owner_info is None:
            return None
        if owner_info.fullname == SQLMODEL_BASEMODEL_FULLNAME:
            return None
        if not owner_info.has_base(SQLMODEL_BASEMODEL_FULLNAME):
            return None

        if not self._declares_sqlmodel_member(owner_info, attr_name):
            return None

        def _hook(ctx: AttributeContext) -> Type:
            return self._sqlmodel_class_attr_type_callback(ctx)

        return _hook

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        if fullname == SQLMODEL_COL_FULLNAME:
            return self._sqlmodel_col_return_type_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> dict[str, Any]:
        # Used by mypy to decide whether to invalidate incremental caches.
        return self._plugin_data

    def _declares_sqlmodel_member(self, info: TypeInfo, name: str) -> bool:
        """Return True if `name` is declared as a field/relationship on `info`."""
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
        """Collect SQLModel fields for signature generation (including inherited ones)."""
        found_fields: dict[str, _CollectedField] = {}
        found_relationships: set[str] = set()

        def _add_field(field: _CollectedField) -> None:
            found_relationships.discard(field.name)
            found_fields[field.name] = field

        def _add_relationship(name: str) -> None:
            found_fields.pop(name, None)
            found_relationships.add(name)

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

        return list(found_fields.values())

    def _collect_member_from_stmt(
        self,
        stmt: AssignmentStmt,
        *,
        defining_info: TypeInfo,
        current_info: TypeInfo,
        api: Any,
    ) -> _CollectedField | str | None:
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
            return name

        sym = defining_info.names.get(name)
        if sym is None or sym.node is None:
            return None
        if not isinstance(sym.node, Var):
            return None
        if sym.node.is_classvar:
            return None

        has_default = SQLModelTransformer.get_has_default(stmt)
        init_type = sym.node.type
        if init_type is not None and defining_info is not current_info:
            with state.strict_optional_set(api.options.strict_optional):
                init_type = map_type_from_supertype(init_type, current_info, defining_info)

        return _CollectedField(
            name=name,
            has_default=has_default,
            line=stmt.line,
            column=stmt.column,
            type=init_type,
        )

    def _sqlmodel_constructor_signature_callback(
        self, ctx: FunctionSigContext, info: TypeInfo
    ) -> FunctionLike:
        fields = self._collect_fields_for_signature(info, ctx.api)

        arg_types: list[Type] = []
        arg_names: list[str | None] = []
        arg_kinds: list[ArgKind] = []

        typed = self.plugin_config.init_typed

        for f in fields:
            if typed and f.type is not None:
                t = f.type.accept(ForceInvariantTypeVars())
                t = _unwrap_mapped_type(t)
            else:
                t = AnyType(TypeOfAny.explicit)
            arg_types.append(t)
            arg_names.append(f.name)
            arg_kinds.append(ARG_NAMED_OPT if f.has_default else ARG_NAMED)

        if not self.plugin_config.init_forbid_extra:
            kw = AnyType(TypeOfAny.explicit)
            arg_types.append(kw)
            arg_names.append("kwargs")
            arg_kinds.append(ARG_STAR2)

        # Preserve mypy's inferred return type for the class call.
        ret_type = ctx.default_signature.ret_type
        fallback = ctx.default_signature.fallback
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

        fields = self._collect_fields_for_signature(receiver_info, ctx.api)

        set_str = ctx.api.named_generic_type(
            "builtins.set", [ctx.api.named_generic_type("builtins.str", [])]
        )
        optional_set_str: Type = UnionType([set_str, NoneType()])

        arg_types: list[Type] = [optional_set_str]
        arg_names: list[str | None] = ["_fields_set"]
        arg_kinds: list[ArgKind] = [ARG_OPT]

        for f in fields:
            t = (
                f.type.accept(ForceInvariantTypeVars())
                if f.type is not None
                else AnyType(TypeOfAny.explicit)
            )
            t = _unwrap_mapped_type(t)
            arg_types.append(t)
            arg_names.append(f.name)
            arg_kinds.append(ARG_NAMED_OPT if f.has_default else ARG_NAMED)

        if not self.plugin_config.init_forbid_extra:
            kw = AnyType(TypeOfAny.explicit)
            arg_types.append(kw)
            arg_names.append("kwargs")
            arg_kinds.append(ARG_STAR2)

        ret_type: Type = receiver_instance or fill_typevars(receiver_info)
        fallback = ctx.default_signature.fallback
        return CallableType(arg_types, arg_kinds, arg_names, ret_type, fallback)

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

        inst_attr_info = _lookup_typeinfo(
            self, "sqlalchemy.orm.attributes.InstrumentedAttribute"
        ) or _lookup_typeinfo(self, "sqlalchemy.orm.InstrumentedAttribute")
        if inst_attr_info is None:
            return default

        return Instance(inst_attr_info, [default])

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

        mapped_info = _lookup_typeinfo(self, "sqlalchemy.orm.base.Mapped") or _lookup_typeinfo(
            self, "sqlalchemy.orm.Mapped"
        )
        if mapped_info is None:
            return ctx.default_return_type

        return Instance(mapped_info, [value_type])

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
