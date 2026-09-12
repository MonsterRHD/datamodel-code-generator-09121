"""Tests for GraphQL schema parser."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from datamodel_code_generator import InputFileType, SchemaParseError, generate
from datamodel_code_generator.config import GenerateConfig, GraphQLParserConfig
from datamodel_code_generator.enums import GraphQLScope
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


SUBSCRIPTION_SCOPE_SCHEMA = """
interface Node { id: ID! }
enum EventKind { CREATED UPDATED }
union Event = Story | Alert
type Story implements Node { id: ID! title: String! }
type Alert implements Node { id: ID! level: Int }
input EventFilter { kinds: [EventKind!] keyword: String }
type Query { ping: String }
type Mutation { noop: Boolean }
type Subscription implements Node {
  id: ID!
  events(kind: EventKind!, filter: EventFilter, limit: Int = 20): [Event!]!
  alerts: [Alert!]
}
"""


def test_graphql_subscription_scope_emits_root_and_arguments() -> None:
    """Subscription scope keeps the root, arguments models, and wrapper layers."""
    parser = GraphQLParser(
        source=SUBSCRIPTION_SCOPE_SCHEMA,
        graphql_scopes=[GraphQLScope.Subscription],
    )
    output = parser.parse()

    assert "class Subscription(Node):" in output
    assert "events: List[Event]" in output
    assert "alerts: Optional[List[Alert]]" in output
    assert "id: ID" in output
    assert "class SubscriptionEventsArguments(BaseModel):" in output
    assert "kind: EventKind" in output
    assert "filter: Optional[EventFilter]" in output
    assert "limit: Optional[Int] = 20" in output
    assert "class Query" not in output
    assert "class Mutation" not in output


def test_graphql_subscription_scope_config_object() -> None:
    """Subscription scope can be enabled through GraphQLParserConfig."""
    parser = GraphQLParser(
        source=SUBSCRIPTION_SCOPE_SCHEMA,
        config=GraphQLParserConfig(graphql_scopes=[GraphQLScope.Subscription]),
    )
    assert parser.graphql_scopes == [GraphQLScope.Subscription]
    assert "class Subscription(Node):" in parser.parse()


def test_graphql_subscription_root_skipped_without_scope() -> None:
    """Default and explicit schema scope keep omitting the Subscription root."""
    for scopes in (None, [GraphQLScope.Schema]):
        parser = GraphQLParser(source=SUBSCRIPTION_SCOPE_SCHEMA, graphql_scopes=scopes)
        output = parser.parse()
        assert "class Subscription(" not in output
        assert "SubscriptionEventsArguments" not in output
        assert "class Story(Node):" in output


@pytest.mark.parametrize(
    ("schema", "expected_path"),
    [
        pytest.param(
            "type Query { x: String }\ntype Subscription { self: Subscription }\n",
            "Subscription/self/Subscription",
            id="direct-cycle",
        ),
        pytest.param(
            "type Query { x: String }\ntype Event { sub: Subscription }\n"
            "type Subscription { events: Event }\n",
            "Subscription/events/Event/sub/Subscription",
            id="nested-cycle",
        ),
        pytest.param(
            "type Query { x: String }\ntype Other { x: String }\n"
            "union Choice = Subscription | Other\ntype Subscription { choice: Choice }\n",
            "Subscription/choice/Choice/Subscription",
            id="union-cycle",
        ),
        pytest.param(
            "type Query { x: String }\ntype Subscription { q: Query }\n",
            "Subscription/q",
            id="unknown-return-type",
        ),
        pytest.param(
            "schema { query: Root subscription: Root }\ntype Root { a: String }\n",
            "schema/subscription/Root",
            id="multiple-roots",
        ),
    ],
)
def test_graphql_subscription_scope_errors(schema: str, expected_path: str) -> None:
    """Subscription scope violations raise field-path errors before models are written."""
    parser = GraphQLParser(source=schema, graphql_scopes=[GraphQLScope.Subscription])
    with pytest.raises(SchemaParseError) as exc_info:
        parser.parse()
    assert expected_path in str(exc_info.value)


def test_graphql_subscription_scope_allows_ordinary_recursive_types() -> None:
    """Cycles that do not re-enter the subscription root remain valid."""
    parser = GraphQLParser(
        source=(
            "type Query { x: String }\n"
            "type Comment { reply: Comment text: String }\n"
            "type Subscription { comment: Comment }\n"
        ),
        graphql_scopes=[GraphQLScope.Subscription],
    )
    output = parser.parse()
    assert "class Comment(BaseModel):" in output
    assert "class Subscription(BaseModel):" in output


def test_graphql_subscription_arguments_model_name_clash_disambiguated() -> None:
    """Arguments models avoid overwriting an ordinary type with the same name."""
    parser = GraphQLParser(
        source=(
            "type Query { x: String }\n"
            "type Event { id: ID! }\n"
            "type SubscriptionEventsArguments { note: String }\n"
            "type Subscription { events(topic: String!): [Event!]! }\n"
        ),
        graphql_scopes=[GraphQLScope.Subscription],
    )
    output = parser.parse()
    assert "class SubscriptionEventsArguments(BaseModel):" in output
    assert "class SubscriptionEventsArguments1(BaseModel):" in output
    assert parser.subscription_argument_models == {"SubscriptionEventsArguments1": "events"}


def test_graphql_subscription_scope_generate_api(output_file: Path) -> None:
    """generate() forwards graphql_scopes to the GraphQL parser."""
    config = GenerateConfig(
        input_file_type=InputFileType.GraphQL,
        output=output_file,
        graphql_scopes=[GraphQLScope.Subscription],
        disable_timestamp=True,
    )
    generate(SUBSCRIPTION_SCOPE_SCHEMA, config=config)
    content = output_file.read_text(encoding="utf-8")
    assert "class Subscription(Node):" in content
    assert "class SubscriptionEventsArguments(" in content
