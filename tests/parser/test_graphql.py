"""Tests for GraphQL schema parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from datamodel_code_generator import InputFileType, InvalidFileFormatError, generate
from datamodel_code_generator.config import GenerateConfig, GraphQLParserConfig
from datamodel_code_generator.model.dataclass import DataClass
from datamodel_code_generator.parser.graphql import GraphQLParser
from datamodel_code_generator.reference import Reference
from tests.conftest import create_assert_file_content
from tests.main.conftest import GRAPHQL_DATA_PATH, run_main_and_assert
from tests.main.test_main_general import DATA_PATH

if TYPE_CHECKING:
    from pathlib import Path

EXPECTED_GRAPHQL_PATH: Path = DATA_PATH / "expected" / "parser" / "graphql"

assert_file_content = create_assert_file_content(EXPECTED_GRAPHQL_PATH)

GRAPHQL_COLLECTION_OPTIONS_PATH = GRAPHQL_DATA_PATH / "collection-options.graphql"
GRAPHQL_COLLECTION_OPTIONS_EXPECTED = "collection_options.py"
GRAPHQL_COLLECTION_OPTIONS_GENERATE_EXPECTED = "collection_options_generate.py"


def _assert_collection_options_output(output_file: Path, output: object) -> None:
    assert isinstance(output, str)
    output_file.write_text(output, encoding="utf-8")
    assert_file_content(output_file, GRAPHQL_COLLECTION_OPTIONS_EXPECTED)


def test_graphql_field_enum(output_file: Path) -> None:
    """Test parsing GraphQL field with enum default value."""
    run_main_and_assert(
        input_path=GRAPHQL_DATA_PATH / "field-default-enum.graphql",
        output_path=output_file,
        input_file_type="graphql",
        assert_func=assert_file_content,
        expected_file="field-default-enum.py",
        extra_args=["--set-default-enum-member"],
    )


def test_graphql_union_aliased_bug(output_file: Path) -> None:
    """Test parsing GraphQL union with aliased types."""
    run_main_and_assert(
        input_path=GRAPHQL_DATA_PATH / "union-aliased-bug.graphql",
        output_path=output_file,
        input_file_type="graphql",
        assert_func=assert_file_content,
        expected_file="union-aliased-bug.py",
    )


def test_graphql_union_commented(output_file: Path) -> None:
    """Test parsing GraphQL union with comments."""
    run_main_and_assert(
        input_path=GRAPHQL_DATA_PATH / "union-commented.graphql",
        output_path=output_file,
        input_file_type="graphql",
        assert_func=assert_file_content,
        expected_file="union-commented.py",
    )


def test_graphql_union_with_prefix(output_file: Path) -> None:
    """Test parsing GraphQL union with class name prefix (Unions should reference prefixed class names)."""
    run_main_and_assert(
        input_path=GRAPHQL_DATA_PATH / "union.graphql",
        output_path=output_file,
        input_file_type="graphql",
        assert_func=assert_file_content,
        expected_file="union_with_prefix.py",
        extra_args=["--class-name-prefix", "Foo"],
    )


def test_graphql_parser_options_path_collection_options_parse(output_file: Path) -> None:
    """Exercise direct GraphQLParser options that are consumed while parsing."""
    parser = GraphQLParser(
        source=GRAPHQL_COLLECTION_OPTIONS_PATH,
        use_standard_collections=True,
        use_union_operator=True,
    )

    output = parser.parse()

    assert parser.use_standard_collections is True
    assert parser.use_union_operator is True
    _assert_collection_options_output(output_file, output)


def test_graphql_parser_text_collection_options_parse(output_file: Path) -> None:
    """Exercise in-memory GraphQL input without changing generated output."""
    parser = GraphQLParser(
        source=GRAPHQL_COLLECTION_OPTIONS_PATH.read_text(encoding="utf-8"),
        use_standard_collections=True,
        use_union_operator=True,
    )

    output = parser.parse()

    _assert_collection_options_output(output_file, output)


def test_graphql_parser_config_path_collection_options_parse(output_file: Path) -> None:
    """Exercise GraphQLParser config object options that are consumed while parsing."""
    parser = GraphQLParser(
        source=GRAPHQL_COLLECTION_OPTIONS_PATH,
        config=GraphQLParserConfig(
            use_standard_collections=True,
            use_union_operator=True,
        ),
    )

    output = parser.parse()

    assert parser.config.use_standard_collections is True
    assert parser.config.use_union_operator is True
    assert parser.use_standard_collections is True
    assert parser.use_union_operator is True
    _assert_collection_options_output(output_file, output)


def test_graphql_generate_config_path_collection_options_parse(output_file: Path) -> None:
    """Exercise generate(config=...) GraphQL options that are consumed while parsing."""
    config = GenerateConfig(
        input_file_type=InputFileType.GraphQL,
        output=output_file,
        use_standard_collections=True,
        use_union_operator=True,
    )

    generate(GRAPHQL_COLLECTION_OPTIONS_PATH, config=config)
    # generate() emits the file-output variant with the header and TypeAliasType aliases.
    assert_file_content(output_file, GRAPHQL_COLLECTION_OPTIONS_GENERATE_EXPECTED)


@pytest.mark.parametrize(
    ("frozen_dataclasses", "keyword_only", "parser_dataclass_args", "kwargs_dataclass_args", "expected"),
    [
        (False, False, None, None, {}),
        (True, False, None, None, {"frozen": True}),
        (False, True, None, None, {"kw_only": True}),
        (True, True, None, None, {"frozen": True, "kw_only": True}),
        (False, False, {"slots": True}, None, {"slots": True}),
        (True, True, {"slots": True}, None, {"slots": True}),
        (True, True, {"slots": True}, {"order": True}, {"order": True}),
    ],
)
def test_create_data_model_dataclass_arguments(
    frozen_dataclasses: bool,
    keyword_only: bool,
    parser_dataclass_args: dict | None,
    kwargs_dataclass_args: dict | None,
    expected: dict,
) -> None:
    """Test _create_data_model handles dataclass_arguments correctly."""
    parser = GraphQLParser(
        source="type Query { id: ID }",
        data_model_type=DataClass,
        frozen_dataclasses=frozen_dataclasses,
        keyword_only=keyword_only,
    )
    parser.dataclass_arguments = parser_dataclass_args

    reference = Reference(path="test", original_name="Test", name="Test")
    kwargs: dict[str, Any] = {"reference": reference, "fields": []}
    if kwargs_dataclass_args is not None:
        kwargs["dataclass_arguments"] = kwargs_dataclass_args
    result = parser._create_data_model(**kwargs)
    assert isinstance(result, DataClass)
    assert result.dataclass_arguments == expected


def test_create_data_model_class_decorators() -> None:
    """Test _create_data_model applies class_decorators correctly."""
    parser = GraphQLParser(
        source="type Query { id: ID }",
        data_model_type=DataClass,
        class_decorators=["@dataclass_json"],
    )

    reference = Reference(path="test", original_name="Test", name="Test")
    result = parser._create_data_model(reference=reference, fields=[])
    assert isinstance(result, DataClass)
    assert result.decorators == ["@dataclass_json"]


def test_graphql_no_typename(output_file: Path) -> None:
    """Test that --graphql-no-typename excludes typename__ field from all types."""
    run_main_and_assert(
        input_path=GRAPHQL_DATA_PATH / "no-typename.graphql",
        output_path=output_file,
        input_file_type="graphql",
        assert_func=assert_file_content,
        expected_file="no_typename.py",
        extra_args=["--graphql-no-typename"],
    )


def test_graphql_typename_included_by_default(output_file: Path) -> None:
    """Regression test: typename__ field is included by default."""

    def assert_typename_present(output_path: Path, _: str | None, **_kwargs: object) -> None:
        content = output_path.read_text(encoding="utf-8")
        assert "typename__" in content, "typename__ field should be present by default"
        assert "__typename" in content, "__typename alias should be present by default"

    run_main_and_assert(
        input_path=GRAPHQL_DATA_PATH / "no-typename.graphql",
        output_path=output_file,
        input_file_type="graphql",
        assert_func=assert_typename_present,
        expected_file=None,
    )


def test_graphql_schema_features() -> None:
    """Test that GraphQLParser has schema_features property returning JsonSchemaFeatures."""
    from inline_snapshot import snapshot

    from datamodel_code_generator.parser.schema_version import JsonSchemaFeatures

    parser = GraphQLParser(
        source="type Query { id: ID }",
        data_model_type=DataClass,
    )

    features = parser.schema_features
    assert isinstance(features, JsonSchemaFeatures)
    assert features == snapshot(
        JsonSchemaFeatures(
            null_in_type_array=True,
            defs_not_definitions=True,
            prefix_items=True,
            boolean_schemas=True,
            id_field="$id",
            definitions_key="$defs",
            exclusive_as_number=True,
            read_only_write_only=True,
            anchor=True,
            recursive_ref=True,
            dynamic_ref=True,
        )
    )


DIRECTIVE_SCHEMA = """
directive @sensitive(role: String!)
  on FIELD_DEFINITION | OBJECT | INTERFACE | INPUT_OBJECT | ENUM | ENUM_VALUE | INPUT_FIELD_DEFINITION
directive @paginated(limit: Int = 10, flags: [Boolean!], mode: Color) on FIELD_DEFINITION
enum Color { RED GREEN }
interface Node @sensitive(role: "node") {
  id: ID! @sensitive(role: "id")
}
type User implements Node @sensitive(role: "user") {
  name: String @sensitive(role: "name")
  friends: [User!] @paginated(limit: 5, flags: [true, false], mode: RED)
}
input UserInput @sensitive(role: "input") {
  name: String! @sensitive(role: "iname")
}
enum Status {
  ACTIVE @sensitive(role: "active")
  INACTIVE
}
type Query {
  me: User
  palette: Status
}
"""


def _generated_model(parser: GraphQLParser, name: str) -> Any:
    """Look up a generated data model by type name."""
    return parser.references[name].source


def _generated_field(model: Any, name: str) -> Any:
    """Look up a generated field by name."""
    return next(field for field in model.fields if field.name == name)


def test_graphql_keep_directives_parser_option_attaches_metadata() -> None:
    """Directive name, arguments and location are attached to models and fields."""
    parser = GraphQLParser(source=DIRECTIVE_SCHEMA, graphql_keep_directives=True)
    parser.parse()

    node = _generated_model(parser, "Node")
    assert node.extra_template_data["model_extras"]["directives"] == [
        {"name": "sensitive", "arguments": {"role": "node"}, "location": "INTERFACE"}
    ]
    assert _generated_field(node, "id").extras["directives"] == [
        {"name": "sensitive", "arguments": {"role": "id"}, "location": "FIELD_DEFINITION"}
    ]

    user = _generated_model(parser, "User")
    assert user.extra_template_data["model_extras"]["directives"] == [
        {"name": "sensitive", "arguments": {"role": "user"}, "location": "OBJECT"}
    ]
    assert _generated_field(user, "friends").extras["directives"] == [
        {
            "name": "paginated",
            "arguments": {"limit": 5, "flags": [True, False], "mode": "RED"},
            "location": "FIELD_DEFINITION",
        }
    ]

    user_input = _generated_model(parser, "UserInput")
    assert user_input.extra_template_data["model_extras"]["directives"] == [
        {"name": "sensitive", "arguments": {"role": "input"}, "location": "INPUT_OBJECT"}
    ]
    assert _generated_field(user_input, "name").extras["directives"] == [
        {"name": "sensitive", "arguments": {"role": "iname"}, "location": "INPUT_FIELD_DEFINITION"}
    ]

    status = _generated_model(parser, "Status")
    assert not status.extra_template_data.get("directives")
    active = _generated_field(status, "ACTIVE")
    assert active.extras["directives"] == [
        {"name": "sensitive", "arguments": {"role": "active"}, "location": "ENUM_VALUE"}
    ]


def test_graphql_keep_directives_off_keeps_metadata_absent() -> None:
    """The default configuration neither validates nor stores directive metadata."""
    parser = GraphQLParser(source=DIRECTIVE_SCHEMA)
    parser.parse()

    for name in ("Node", "User", "UserInput"):
        model = _generated_model(parser, name)
        assert "model_extras" not in model.extra_template_data
        assert all("directives" not in field.extras for field in model.fields)
    status = _generated_model(parser, "Status")
    assert "directives" not in status.extra_template_data
    assert all("directives" not in field.extras for field in status.fields)


def test_graphql_keep_directives_config_object() -> None:
    """GraphQLParserConfig carries the option like other GraphQL-only flags."""
    parser = GraphQLParser(source=DIRECTIVE_SCHEMA, config=GraphQLParserConfig(graphql_keep_directives=True))
    assert parser.config.graphql_keep_directives is True
    assert parser.graphql_keep_directives is True


@pytest.mark.parametrize(
    ("schema", "expected_detail"),
    [
        ("type User { name: String @unknown }", "<input>:1:26: Unknown directive '@unknown'."),
        (
            'directive @d(a: String) on FIELD_DEFINITION\ntype User { name: String @d(a: "1", a: "2") }',
            "There can be only one argument named 'a'.",
        ),
        (
            "directive @d on OBJECT\ntype User { name: String @d }",
            "Directive '@d' may not be used on field definition.",
        ),
    ],
)
def test_graphql_keep_directives_rejects_invalid_directives(schema: str, expected_detail: str) -> None:
    """Invalid directive usage fails with a source location instead of partial output."""
    parser = GraphQLParser(source=schema, graphql_keep_directives=True)
    with pytest.raises(InvalidFileFormatError) as exc_info:
        parser.parse()
    assert expected_detail in str(exc_info.value)


def test_graphql_keep_directives_stitched_sources_map_locations(tmp_path: Path) -> None:
    """Joined-document errors map back to the originating external schema file."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    (schema_dir / "a.graphql").write_text("type A {\n  x: Int\n}\n", encoding="utf-8")
    (schema_dir / "b.graphql").write_text("type B {\n  y: String @nope\n}\n", encoding="utf-8")

    parser = GraphQLParser(source=schema_dir, graphql_keep_directives=True)
    with pytest.raises(InvalidFileFormatError) as exc_info:
        parser.parse()
    message = str(exc_info.value)
    assert "ValueError: b.graphql:2:13: Unknown directive '@nope'." in message
    # The error detail must not attribute the failure to the other stitched file.
    assert "a.graphql:" not in message


def test_graphql_keep_directives_syntax_error_keeps_location() -> None:
    """Syntax errors keep their line/column with the directive-aware build path."""
    parser = GraphQLParser(source="type User { name: String ", graphql_keep_directives=True)
    with pytest.raises(InvalidFileFormatError) as exc_info:
        parser.parse()
    assert "<input>:1:26" in str(exc_info.value)


@pytest.mark.parametrize("graphql_keep_directives", [False, True])
def test_graphql_keep_directives_accepts_declared_directives(graphql_keep_directives: bool) -> None:
    """Declared directives build successfully regardless of the preservation option."""
    parser = GraphQLParser(
        source="directive @d on OBJECT\ntype User @d { name: String }",
        graphql_keep_directives=graphql_keep_directives,
    )
    parser.parse()
    user = _generated_model(parser, "User")
    stored = user.extra_template_data.get("model_extras")
    if graphql_keep_directives:
        assert stored["directives"] == [{"name": "d", "arguments": {}, "location": "OBJECT"}]
    else:
        assert not stored
