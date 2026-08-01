"""OpenResponses public output-format model tests."""

from molto.types.openresponses import OpenResponsesJsonSchemaFormat


def test_json_schema_format_round_trips_through_model_dump() -> None:
    fmt = OpenResponsesJsonSchemaFormat(name="answer", schema={"type": "object"})

    dumped = fmt.model_dump()

    assert dumped["schema"] == {"type": "object"}
    assert "schema_" not in dumped
    assert OpenResponsesJsonSchemaFormat.model_validate(dumped) == fmt
