"""GraphQL schema parser implementation.

Parses GraphQL schema files to generate Python data models including
objects, interfaces, enums, scalars, inputs, and union types.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    cast,
)

from graphql.language import ast as gql_ast
from graphql.validation.validate import validate_sdl
from typing_extensions import Unpack

from datamodel_code_generator import (
    InputFileType,
    InvalidFileFormatError,
    LiteralType,
    snooper_to_methods,
)
from datamodel_code_generator._format_types import DatetimeClassType
from datamodel_code_generator.model.enum import SPECIALIZED_ENUM_TYPE_MATCH, Enum, EnumMemberValue
from datamodel_code_generator.parser.base import (
    DataType,
    Parser,
)
from datamodel_code_generator.reference import ModelType, Reference
from datamodel_code_generator.types import Types

try:
    import graphql
except ImportError as exc:  # pragma: no cover
    msg = "Please run `$pip install 'datamodel-code-generator[graphql]`' to generate data-model from a GraphQL schema."
    raise Exception(msg) from exc  # noqa: TRY002


if TYPE_CHECKING:
    from pathlib import Path
    from urllib.parse import ParseResult

    from datamodel_code_generator._types import GraphQLParserConfigDict
    from datamodel_code_generator.config import GraphQLParserConfig
    from datamodel_code_generator.model import DataModel, DataModelFieldBase
    from datamodel_code_generator.parser.schema_version import JsonSchemaFeatures

# graphql-core >=3.2.7 removed TypeResolvers in favor of TypeFields.kind.
# Normalize to a single callable for resolving type kinds.
try:  # graphql-core < 3.2.7
    graphql_resolver_kind = graphql.type.introspection.TypeResolvers().kind  # ty: ignore[unresolved-attribute]
except AttributeError:
    graphql_resolver_kind = graphql.type.introspection.TypeFields.kind


@dataclass(frozen=True, slots=True)
class _GraphQLSourceSegment:
    """A stitched GraphQL source and the line range it occupies in the joined document."""

    name: str
    start_line: int
    line_count: int

    def locate(self, line: int) -> tuple[str, int]:
        """Map a joined-document line number back to this source and its local line."""
        if line > self.start_line + self.line_count - 1:
            return self.name, self.line_count + 1
        return self.name, line - self.start_line + 1


def _segment_for_line(
    line: int,
    segments: "tuple[_GraphQLSourceSegment, ...]",
) -> tuple[str, int] | None:
    """Resolve a joined-document line to the originating source segment."""
    if not segments:
        return None
    for segment in segments:
        if segment.start_line <= line < segment.start_line + segment.line_count:
            return segment.locate(line)
    return segments[-1].locate(line)


def _format_graphql_error_location(
    locations: "list[graphql.SourceLocation] | None",
    segments: "tuple[_GraphQLSourceSegment, ...]",
) -> str:
    """Render error locations, remapping stitched sources back to their original files."""
    if not locations:
        return ""
    rendered: list[str] = []
    for location in locations:
        resolved = _segment_for_line(location.line, segments)
        if resolved is None:
            rendered.append(f"{location.line}:{location.column}")
        else:
            source_name, local_line = resolved
            rendered.append(f"{source_name}:{local_line}:{location.column}")
    return ", ".join(rendered)


def _format_graphql_error(
    error: "graphql.GraphQLError",
    segments: "tuple[_GraphQLSourceSegment, ...]",
) -> str:
    """Render a single GraphQL error with source-mapped locations."""
    location_detail = _format_graphql_error_location(getattr(error, "locations", None), segments)
    message = getattr(error, "message", None) or str(error)
    return f"{location_detail}: {message}" if location_detail else message


def build_graphql_schema(
    schema_str: str,
    *,
    source: str | None = None,
    keep_directives: bool = False,  # noqa: FBT001, FBT002
    segments: "tuple[_GraphQLSourceSegment, ...]" = (),
) -> graphql.GraphQLSchema:
    """Build a graphql schema from a string.

    When ``keep_directives`` is enabled the SDL is parsed and validated before
    construction so unknown directives, duplicate directive arguments and
    stitched-schema failures can be reported with their originating source
    locations.  The legacy ``build_schema`` path is kept unchanged otherwise.
    """
    if not keep_directives:
        try:
            schema = graphql.build_schema(schema_str)
        except graphql.GraphQLSyntaxError as exc:
            raise InvalidFileFormatError(exc, InputFileType.GraphQL, source=source) from exc
        return graphql.lexicographic_sort_schema(schema)

    try:
        document = graphql.parse(schema_str)
    except graphql.GraphQLSyntaxError as exc:
        raise InvalidFileFormatError(
            ValueError(_format_graphql_error(exc, segments)),
            InputFileType.GraphQL,
            source=source,
        ) from exc

    errors = validate_sdl(document)
    if errors:
        detail = "\n".join(_format_graphql_error(error, segments) for error in errors)
        raise InvalidFileFormatError(ValueError(detail), InputFileType.GraphQL, source=source)

    try:
        schema = graphql.build_ast_schema(document, assume_valid=True)
    except Exception as exc:  # noqa: BLE001 - surface construction failures with source context
        node_locations = [
            graphql.SourceLocation(
                line=node.loc.start_token.line,
                column=node.loc.start_token.column,
            )
            for node in getattr(exc, "nodes", ()) or ()
            if getattr(node, "loc", None) is not None
        ]
        location_detail = _format_graphql_error_location(node_locations, segments)
        detail = f"{location_detail}: {exc}" if location_detail else str(exc)
        raise InvalidFileFormatError(ValueError(detail), InputFileType.GraphQL, source=source) from exc

    return graphql.lexicographic_sort_schema(schema)


def _directive_argument_value(node: gql_ast.ValueNode) -> Any:
    """Convert a GraphQL directive argument AST value into a plain Python value."""
    if isinstance(node, gql_ast.IntValueNode):
        return int(node.value)
    if isinstance(node, gql_ast.FloatValueNode):
        return float(node.value)
    if isinstance(node, gql_ast.NullValueNode):
        return None
    if isinstance(node, gql_ast.ListValueNode):
        return [_directive_argument_value(value) for value in node.values]
    if isinstance(node, gql_ast.ObjectValueNode):
        return {field.name.value: _directive_argument_value(field.value) for field in node.fields}
    # StringValueNode (incl. block strings), BooleanValueNode and EnumValueNode
    # all expose their lexical value as ``.value``; enum identifiers are kept
    # as plain strings.
    return cast("str", node.value)


def _build_directive_entries(location: str, *ast_nodes: gql_ast.Node | None) -> list[dict[str, Any]]:
    """Collect directive name/arguments/location entries from definition AST nodes."""
    entries: list[dict[str, Any]] = []
    for ast_node in ast_nodes:
        if ast_node is None:
            continue
        entries.extend(
            {
                "name": directive.name.value,
                "arguments": {
                    argument.name.value: _directive_argument_value(argument.value)
                    for argument in directive.arguments
                },
                "location": location,
            }
            for directive in ast_node.directives
        )
    return entries


@snooper_to_methods()
class GraphQLParser(Parser["GraphQLParserConfig", "JsonSchemaFeatures"]):
    """Parser for GraphQL schema files."""

    # raw graphql schema as `graphql-core` object
    raw_obj: graphql.GraphQLSchema

    @cached_property
    def schema_features(self) -> JsonSchemaFeatures:
        """Get schema features for GraphQL (uses default JSON Schema features)."""
        from datamodel_code_generator.enums import JsonSchemaVersion  # noqa: PLC0415
        from datamodel_code_generator.parser.schema_version import JsonSchemaFeatures  # noqa: PLC0415

        return JsonSchemaFeatures.from_version(JsonSchemaVersion.Draft202012)

    # all processed graphql objects
    # mapper from an object name (unique) to an object
    all_graphql_objects: dict[str, graphql.GraphQLNamedType]
    # a reference for each object
    # mapper from an object name to his reference
    references: dict[str, Reference]
    # mapper from graphql type to all objects with this type
    # `graphql.type.introspection.TypeKind` -- an enum with all supported types
    # `graphql.GraphQLNamedType` -- base type for each graphql object
    # see `graphql-core` for more details
    support_graphql_types: dict[graphql.type.introspection.TypeKind, list[graphql.GraphQLNamedType]]
    _typename_collisions: list[graphql.GraphQLObjectType | graphql.GraphQLInterfaceType] | None
    # graphql types order for render
    # may be as a parameter in the future
    parse_order: list[graphql.type.introspection.TypeKind] = [  # noqa: RUF012
        graphql.type.introspection.TypeKind.SCALAR,
        graphql.type.introspection.TypeKind.ENUM,
        graphql.type.introspection.TypeKind.INTERFACE,
        graphql.type.introspection.TypeKind.OBJECT,
        graphql.type.introspection.TypeKind.INPUT_OBJECT,
        graphql.type.introspection.TypeKind.UNION,
    ]

    _config_class_name: ClassVar[str] = "GraphQLParserConfig"

    def __init__(
        self,
        source: str | Path | ParseResult,
        *,
        config: GraphQLParserConfig | None = None,
        **options: Unpack[GraphQLParserConfigDict],
    ) -> None:
        """Initialize the GraphQL parser with configuration options."""
        if config is None and options.get("target_datetime_class") is None:
            options["target_datetime_class"] = DatetimeClassType.Datetime
        super().__init__(source=source, config=config, **options)

        self.references: dict[str, Reference] = {}
        self.all_graphql_objects: dict[str, graphql.GraphQLNamedType] = {}
        self.data_model_scalar_type = self.config.data_model_scalar_type
        self.data_model_union_type = self.config.data_model_union_type
        self.use_standard_collections = self.config.use_standard_collections
        self.use_union_operator = self.config.use_union_operator
        self.graphql_keep_directives = self.config.graphql_keep_directives

    def _resolve_types(self, paths: list[str], schema: graphql.GraphQLSchema) -> None:
        root_types = {schema.query_type, schema.mutation_type, schema.subscription_type}
        for type_ in schema.type_map.values():
            if isinstance(type_, graphql.GraphQLUnionType):
                root_types.difference_update(type_.types)
        for type_name, type_ in schema.type_map.items():
            if type_name.startswith("__"):
                continue

            if type_ in root_types:
                continue

            resolved_type = graphql_resolver_kind(type_, None)

            if resolved_type in self.support_graphql_types:  # pragma: no cover
                self.all_graphql_objects[type_.name] = type_
                graphql_model_type = "enum" if resolved_type == graphql.TypeKind.ENUM else "model"
                affixed_name = self.model_resolver.get_affixed_name(type_.name, model_type=graphql_model_type)
                self.references[type_.name] = Reference(
                    path=f"{paths!s}/{resolved_type.value}/{type_.name}",
                    name=affixed_name,
                    original_name=type_.name,
                )

                self.support_graphql_types[resolved_type].append(type_)

    def _typename_field(self, name: str, excludes: set[str]) -> DataModelFieldBase:
        field_name = "typename__"
        if field_name in excludes:
            field_name = self.model_resolver.get_valid_field_name(
                field_name, excludes=excludes, model_type=self.field_name_model_type
            )
        return self.data_model_field_type(
            name=field_name,
            data_type=DataType(
                literals=[name],
                use_union_operator=self.use_union_operator,
                use_standard_collections=self.use_standard_collections,
            ),
            default=name,
            use_annotated=self.use_annotated,
            required=False,
            alias="__typename",
            serialization_alias=self.get_serialization_alias("__typename", field_name, name),
            use_one_literal_as_default=True,
            use_default_kwarg=self.use_default_kwarg,
            has_default=True,
            use_serialization_alias=self.use_serialization_alias,
            **self._data_model_field_common_kwargs(),
        )

    def _get_default(  # noqa: PLR6301
        self,
        field: graphql.GraphQLField | graphql.GraphQLInputField,
        final_data_type: DataType,  # noqa: ARG002
        *,
        required: bool,  # noqa: ARG002
    ) -> Any:
        if isinstance(field, graphql.GraphQLInputField):
            if field.default_value == graphql.pyutils.Undefined:
                return None
            return field.default_value

        return None

    def _has_schema_default(  # noqa: PLR6301
        self, field: graphql.GraphQLField | graphql.GraphQLInputField
    ) -> bool:
        """Return whether a GraphQL input field defines a schema default."""
        return isinstance(field, graphql.GraphQLInputField) and field.default_value != graphql.pyutils.Undefined

    def _type_directive_entries(
        self,
        location: str,
        graphql_object: graphql.GraphQLNamedType,
    ) -> list[dict[str, Any]]:
        """Return directive entries declared on a type definition and its extensions."""
        if not self.graphql_keep_directives:
            return []
        return _build_directive_entries(
            location,
            getattr(graphql_object, "ast_node", None),
            *getattr(graphql_object, "extension_ast_nodes", ()) or (),
        )

    def _store_type_directives(self, type_name: str, entries: list[dict[str, Any]]) -> None:
        """Attach type-level directive metadata to the model's template data."""
        if entries:
            self.extra_template_data[self.references[type_name].path]["directives"] = entries

    def _field_directive_extras(
        self,
        field: graphql.GraphQLField | graphql.GraphQLInputField,
    ) -> dict[str, list[dict[str, Any]]]:
        """Build field extras carrying directives on a field or input field definition."""
        if not self.graphql_keep_directives or field.ast_node is None:
            return {}
        location = (
            "INPUT_FIELD_DEFINITION" if isinstance(field, graphql.GraphQLInputField) else "FIELD_DEFINITION"
        )
        entries = _build_directive_entries(location, field.ast_node)
        return {"directives": entries} if entries else {}

    def parse_scalar(self, scalar_graphql_object: graphql.GraphQLScalarType) -> None:
        """Parse a GraphQL scalar type and add it to results."""
        entries = self._type_directive_entries("SCALAR", scalar_graphql_object)
        self._store_type_directives(scalar_graphql_object.name, entries)
        self.generation_store.register_model(
            self.data_model_scalar_type(
                reference=self.references[scalar_graphql_object.name],
                fields=[],
                custom_template_dir=self.custom_template_dir,
                extra_template_data=self.extra_template_data,
                description=scalar_graphql_object.description,
            )
        )

    def should_parse_enum_as_literal(self, obj: graphql.GraphQLEnumType) -> bool:
        """Determine if an enum should be parsed as a literal type."""
        if self.enum_field_as_literal == LiteralType.All:
            return True
        if self.enum_field_as_literal == LiteralType.One:
            return len(obj.values) == 1
        return False

    def parse_enum(self, enum_object: graphql.GraphQLEnumType) -> None:
        """Parse a GraphQL enum type and add it to results."""
        if self.ignore_enum_constraints:
            return self.parse_enum_as_str_type(enum_object)
        if self.should_parse_enum_as_literal(enum_object):
            return self.parse_enum_as_literal(enum_object)
        return self.parse_enum_as_enum_class(enum_object)

    def parse_enum_as_str_type(self, enum_object: graphql.GraphQLEnumType) -> None:
        """Parse enum as a str type alias when ignoring enum constraints."""
        self._store_type_directives(
            enum_object.name,
            self._type_directive_entries("ENUM", enum_object),
        )
        data_type = self.data_type_manager.get_data_type(Types.string)
        data_model_type = self._create_data_model(
            model_type=self.data_model_root_type,
            reference=self.references[enum_object.name],
            fields=[
                self.data_model_field_type(
                    required=True,
                    data_type=data_type,
                    **self._data_model_field_common_kwargs(),
                )
            ],
            custom_base_class=self._resolve_base_class(enum_object.name),
            custom_template_dir=self.custom_template_dir,
            extra_template_data=self.extra_template_data,
            path=self.current_source_path,
            description=enum_object.description,
        )
        self.generation_store.register_model(data_model_type)

    def parse_enum_as_literal(self, enum_object: graphql.GraphQLEnumType) -> None:
        """Parse enum values as a Literal type."""
        self._store_type_directives(
            enum_object.name,
            self._type_directive_entries("ENUM", enum_object),
        )
        data_type = self.data_type(literals=list(enum_object.values.keys()))
        data_model_type = self._create_data_model(
            model_type=self.data_model_root_type,
            reference=self.references[enum_object.name],
            fields=[
                self.data_model_field_type(
                    required=True,
                    data_type=data_type,
                    **self._data_model_field_common_kwargs(),
                )
            ],
            custom_base_class=self._resolve_base_class(enum_object.name),
            custom_template_dir=self.custom_template_dir,
            extra_template_data=self.extra_template_data,
            path=self.current_source_path,
            description=enum_object.description,
        )
        self.generation_store.register_model(data_model_type)

    def parse_enum_as_enum_class(self, enum_object: graphql.GraphQLEnumType) -> None:
        """Parse enum values as an Enum class."""
        self._store_type_directives(
            enum_object.name,
            self._type_directive_entries("ENUM", enum_object),
        )
        enum_fields: list[DataModelFieldBase] = []
        exclude_field_names: set[str] = set()

        for value_name, value in enum_object.values.items():
            default = EnumMemberValue(value_name) if isinstance(value_name, str) else value_name

            field_name = self.model_resolver.get_valid_field_name(
                value_name, excludes=exclude_field_names, model_type=ModelType.ENUM
            )
            exclude_field_names.add(field_name)

            field_kwargs: dict[str, Any] = {}
            if self.graphql_keep_directives and value.ast_node is not None:
                member_entries = _build_directive_entries("ENUM_VALUE", value.ast_node)
                if member_entries:
                    field_kwargs["extras"] = {"directives": member_entries}

            enum_fields.append(
                self.data_model_field_type(
                    name=field_name,
                    data_type=self.data_type_manager.get_data_type(
                        Types.string,
                    ),
                    default=default,
                    required=True,
                    strip_default_none=self.strip_default_none,
                    has_default=True,
                    use_field_description=value.description is not None,
                    original_name=None,
                    **field_kwargs,
                    **self._data_model_field_common_kwargs(),
                )
            )

        enum_cls: type[Enum] = Enum
        if (
            self.target_python_version.has_strenum
            and self.use_specialized_enum
            and (specialized_type := SPECIALIZED_ENUM_TYPE_MATCH.get(Types.string))
        ):
            # If specialized enum is available in the target Python version, use it
            enum_cls = specialized_type

        enum: Enum = enum_cls(
            reference=self.references[enum_object.name],
            fields=enum_fields,
            path=self.current_source_path,
            description=enum_object.description,
            type_=Types.string if self.use_subclass_enum else None,
            custom_template_dir=self.custom_template_dir,
            extra_template_data=self.extra_template_data,
        )
        self.generation_store.register_model(enum)

    def parse_field(
        self,
        field_name: str,
        alias: str | list[str] | None,
        field: graphql.GraphQLField | graphql.GraphQLInputField,
        original_field_name: str,
        class_name: str | None = None,
    ) -> DataModelFieldBase:
        """Parse a GraphQL field and return a data model field."""
        final_data_type = DataType(
            is_optional=True,
            use_union_operator=self.use_union_operator,
            use_standard_collections=self.use_standard_collections,
        )
        data_type = final_data_type
        obj = field.type

        while graphql.is_list_type(obj) or graphql.is_non_null_type(obj):
            if graphql.is_list_type(obj):
                data_type.is_list = True

                new_data_type = DataType(
                    is_optional=True,
                    use_union_operator=self.use_union_operator,
                    use_standard_collections=self.use_standard_collections,
                )
                data_type.data_types = [new_data_type]

                data_type = new_data_type
            elif graphql.is_non_null_type(obj):  # pragma: no cover
                data_type.is_optional = False

            obj = graphql.assert_wrapping_type(obj)
            obj = obj.of_type

        obj = graphql.assert_named_type(obj)
        if obj.name in self.references:
            self.generation_store.replace_data_type_ref(data_type, self.references[obj.name])
        else:
            # Operation roots are intentionally not emitted as models.
            any_data_type = self.data_type_manager.get_data_type(Types.any)
            data_type.type = any_data_type.type
            data_type.import_ = any_data_type.import_

        has_schema_default = self._has_schema_default(field)
        required = (
            (not self.force_optional_for_required_fields)
            and (not final_data_type.is_optional)
            and not has_schema_default
        )
        nullable = False if has_schema_default and not final_data_type.is_optional else None

        default = self._get_default(field, final_data_type, required=required)
        effective_default, effective_has_default, use_default_with_required = self._effective_default_state(
            original_field_name,
            default,
            has_default=has_schema_default,
            required=required,
            class_name=class_name,
        )

        extras = {} if self.default_field_extras is None else self.default_field_extras.copy()

        if field.description is not None:  # pragma: no cover
            extras["description"] = field.description

        extras.update(self._field_directive_extras(field))

        single_alias, validation_aliases = self._split_field_alias(alias)
        return self.data_model_field_type(
            name=field_name,
            default=effective_default,
            data_type=final_data_type,
            required=required,
            nullable=nullable,
            extras=extras,
            alias=single_alias,
            validation_aliases=validation_aliases,
            serialization_alias=self.get_serialization_alias(original_field_name, field_name, class_name),
            strip_default_none=self.strip_default_none,
            use_annotated=self.use_annotated,
            use_serialize_as_any=self.use_serialize_as_any,
            use_field_description=self.use_field_description,
            use_field_description_example=self.use_field_description_example,
            use_inline_field_description=self.use_inline_field_description,
            use_default_kwarg=self.use_default_kwarg,
            original_name=field_name,
            has_default=effective_has_default,
            use_serialization_alias=self.use_serialization_alias,
            use_default_with_required=use_default_with_required,
            **self._data_model_field_common_kwargs(),
        )

    def parse_object_like(
        self,
        obj: graphql.GraphQLInterfaceType | graphql.GraphQLObjectType | graphql.GraphQLInputObjectType,
    ) -> None:
        """Parse a GraphQL object-like type and add it to results."""
        fields = []
        exclude_field_names: set[str] = set()

        for original_field_name, field in obj.fields.items():
            field_name_, alias = self.model_resolver.get_valid_field_name_and_alias(
                original_field_name,
                excludes=exclude_field_names,
                model_type=self.field_name_model_type,
                class_name=obj.name,
            )
            exclude_field_names.add(field_name_)

            data_model_field_type = self.parse_field(
                field_name_, alias, field, original_field_name, class_name=obj.name
            )
            fields.append(data_model_field_type)

        if not self.config.graphql_no_typename:
            fields.append(self._typename_field(obj.name, exclude_field_names))

        base_classes = []
        if hasattr(obj, "interfaces"):
            base_classes = [self.references[i.name] for i in obj.interfaces]  # ty: ignore[not-iterable]

        if (
            not self.config.graphql_no_typename
            and fields[-1].name != "typename__"
            and isinstance(obj, graphql.GraphQLObjectType | graphql.GraphQLInterfaceType)
        ):
            if self._typename_collisions is None:
                self._typename_collisions = []
            self._typename_collisions.append(obj)

        if self.graphql_keep_directives:
            if isinstance(obj, graphql.GraphQLObjectType):
                type_location = "OBJECT"
            elif isinstance(obj, graphql.GraphQLInterfaceType):
                type_location = "INTERFACE"
            else:
                type_location = "INPUT_OBJECT"
            type_directives = _build_directive_entries(
                type_location, obj.ast_node, *obj.extension_ast_nodes
            )
            if type_directives:
                template_data = self.extra_template_data[self.references[obj.name].path]
                existing_model_extras = cast("dict[str, Any]", template_data.get("model_extras") or {})
                template_data["model_extras"] = {**existing_model_extras, "directives": type_directives}

        data_model_type = self._create_data_model(
            reference=self.references[obj.name],
            fields=fields,
            base_classes=base_classes,
            custom_base_class=self._resolve_base_class(obj.name),
            custom_template_dir=self.custom_template_dir,
            extra_template_data=self.extra_template_data,
            path=self.current_source_path,
            description=obj.description,
            keyword_only=self.keyword_only,
            treat_dot_as_module=self.treat_dot_as_module,
            dataclass_arguments=self.dataclass_arguments,
        )
        self.generation_store.register_model(data_model_type)

    def parse_interface(self, interface_graphql_object: graphql.GraphQLInterfaceType) -> None:
        """Parse a GraphQL interface type and add it to results."""
        self.parse_object_like(interface_graphql_object)

    def parse_object(self, graphql_object: graphql.GraphQLObjectType) -> None:
        """Parse a GraphQL object type and add it to results."""
        self.parse_object_like(graphql_object)

    def parse_input_object(self, input_graphql_object: graphql.GraphQLInputObjectType) -> None:
        """Parse a GraphQL input object type and add it to results."""
        self.parse_object_like(input_graphql_object)

    def parse_union(self, union_object: graphql.GraphQLUnionType) -> None:
        """Parse a GraphQL union type and add it to results."""
        self._store_type_directives(
            union_object.name,
            self._type_directive_entries("UNION", union_object),
        )
        fields = [
            self.data_model_field_type(
                name=self.references[type_.name].name,
                data_type=DataType(),
                **self._data_model_field_common_kwargs(),
            )
            for type_ in union_object.types
        ]
        data_model_type = self.data_model_union_type(
            reference=self.references[union_object.name],
            fields=fields,
            custom_base_class=self._resolve_base_class(union_object.name),
            custom_template_dir=self.custom_template_dir,
            extra_template_data=self.extra_template_data,
            path=self.current_source_path,
            description=union_object.description,
        )
        self.generation_store.register_model(data_model_type)

    def parse_raw(self) -> None:
        """Parse the raw GraphQL schema and generate all data models."""
        self.all_graphql_objects = {}
        self.references: dict[str, Reference] = {}
        self._typename_collisions = None

        self.support_graphql_types = {
            graphql.type.introspection.TypeKind.SCALAR: [],
            graphql.type.introspection.TypeKind.ENUM: [],
            graphql.type.introspection.TypeKind.UNION: [],
            graphql.type.introspection.TypeKind.INTERFACE: [],
            graphql.type.introspection.TypeKind.OBJECT: [],
            graphql.type.introspection.TypeKind.INPUT_OBJECT: [],
        }

        # may be as a parameter in the future (??)
        mapper_from_graphql_type_to_parser_method = {
            graphql.type.introspection.TypeKind.SCALAR: self.parse_scalar,
            graphql.type.introspection.TypeKind.ENUM: self.parse_enum,
            graphql.type.introspection.TypeKind.INTERFACE: self.parse_interface,
            graphql.type.introspection.TypeKind.OBJECT: self.parse_object,
            graphql.type.introspection.TypeKind.INPUT_OBJECT: self.parse_input_object,
            graphql.type.introspection.TypeKind.UNION: self.parse_union,
        }

        source_paths: list[str] = []
        source_texts: list[str] = []
        source_segments: list[_GraphQLSourceSegment] = []
        next_start_line = 1
        for source in self.iter_source:
            display_path = self._source_path_for_diagnostics(source.path)
            if display_path != "<input>":
                source_paths.append(display_path)
            source_texts.append(source.text)
            source_segments.append(
                _GraphQLSourceSegment(
                    name=display_path,
                    start_line=next_start_line,
                    line_count=source.text.count("\n") + 1,
                )
            )
            next_start_line += source.text.count("\n") + 1

        schema: graphql.GraphQLSchema = build_graphql_schema(
            "\n".join(source_texts),
            source=", ".join(source_paths) or "<input>",
            keep_directives=self.graphql_keep_directives,
            segments=tuple(source_segments),
        )
        self.raw_obj = schema

        self._resolve_types([], schema)

        for next_type in self.parse_order:
            for obj in self.support_graphql_types[next_type]:
                parser_ = mapper_from_graphql_type_to_parser_method[next_type]
                parser_(obj)  # ty: ignore[invalid-argument-type]

        if not (collisions := self._typename_collisions):
            return
        self._typename_collisions = None
        self._resolve_typename_collisions(schema, collisions)

    def _resolve_typename_collisions(
        self,
        schema: graphql.GraphQLSchema,
        collisions: list[graphql.GraphQLObjectType | graphql.GraphQLInterfaceType],
    ) -> None:
        """Keep one synthetic slot throughout each affected inheritance family."""
        visited: set[graphql.GraphQLObjectType | graphql.GraphQLInterfaceType] = set()
        for root in collisions:
            if root in visited:
                continue
            pending = [root]
            excludes: set[str] = set()
            typename_fields: list[tuple[str, DataModelFieldBase]] = []
            while pending:
                obj = pending.pop()
                if obj in visited or obj.name not in self.references:
                    continue
                visited.add(obj)
                pending.extend(obj.interfaces)
                if isinstance(obj, graphql.GraphQLInterfaceType):
                    implementations = schema.get_implementations(obj)
                    pending.extend(implementations.objects)
                    pending.extend(implementations.interfaces)
                source = cast("DataModel", self.references[obj.name].source)
                for field in source.fields:
                    if field.alias == "__typename":
                        typename_fields.append((obj.name, field))
                    else:
                        excludes.add(cast("str", field.name))
            field_name = self.model_resolver.get_valid_field_name(
                "typename__", excludes=excludes, model_type=self.field_name_model_type
            )
            for name, field in typename_fields:
                field.name = field_name
                field.serialization_alias = self.get_serialization_alias("__typename", field_name, name)
