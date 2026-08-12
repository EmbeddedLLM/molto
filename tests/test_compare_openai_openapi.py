from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from scripts.compare_openai_openapi import compare_documents, main


def _schema_reference(name: str) -> dict:
    return {"$ref": f"#/components/schemas/{name}"}


def _operation(request_schema: str, response_schema: str) -> dict:
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": _schema_reference(request_schema)}},
        },
        "responses": {
            "200": {
                "description": "Successful response.",
                "content": {
                    "application/json": {"schema": _schema_reference(response_schema)},
                    "text/event-stream": {"schema": _schema_reference("StreamEvent")},
                },
            }
        },
    }


def _document() -> dict:
    chat_operation = _operation("ChatRequest", "ChatResponse")
    chat_operation["operationId"] = "createChatCompletion"
    response_operation = _operation("ResponseRequest", "Response")
    response_operation["operationId"] = "createResponse"
    return {
        "openapi": "3.1.0",
        "info": {"title": "OpenAI", "version": "1"},
        "paths": {
            "/chat/completions": {"post": chat_operation},
            "/responses": {"post": response_operation},
            "/unrelated": {"get": {"operationId": "unrelated", "responses": {}}},
        },
        "components": {
            "schemas": {
                "ChatRequest": {
                    "type": "object",
                    "properties": {
                        "messages": {
                            "type": "array",
                            "items": _schema_reference("Message"),
                        },
                        "model": {"type": "string"},
                    },
                    "required": ["messages", "model"],
                },
                "ResponseRequest": {
                    "type": "object",
                    "properties": {
                        "input": {"oneOf": [{"type": "string"}, _schema_reference("Message")]},
                        "model": {"type": "string"},
                    },
                    "required": ["input"],
                },
                "Message": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "role": {"type": "string", "enum": ["assistant", "user"]},
                    },
                    "required": ["content", "role"],
                },
                "ChatResponse": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                    "required": ["id"],
                },
                "Response": {
                    "type": "object",
                    "properties": {"output": {"type": "string"}},
                    "required": ["output"],
                },
                "StreamEvent": {
                    "type": "object",
                    "properties": {"type": {"type": "string"}},
                    "required": ["type"],
                },
                "Unused": {"type": "object", "properties": {"old": {"type": "string"}}},
            }
        },
    }


def test_ignores_documentation_order_and_unreachable_changes() -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    latest["info"]["description"] = "New documentation."
    latest["paths"]["/responses"]["post"]["summary"] = "Create a response."
    latest["components"]["schemas"]["Message"]["description"] = "A message."
    latest["components"]["schemas"]["Message"]["additionalProperties"] = True
    latest["components"]["schemas"]["Message"]["readOnly"] = False
    latest["components"]["schemas"]["Message"]["properties"]["role"]["enum"].reverse()
    latest["components"]["schemas"]["ResponseRequest"]["properties"]["input"]["oneOf"].reverse()
    latest["components"]["schemas"]["Unused"]["properties"] = {"new": {"type": "integer"}}

    assert compare_documents(baseline, latest) == []


def test_detects_added_dropped_and_renamed_request_fields() -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    properties = latest["components"]["schemas"]["ChatRequest"]["properties"]
    properties["prompt"] = properties.pop("messages")
    properties["metadata"] = {"type": "object"}
    latest["components"]["schemas"]["ChatRequest"]["required"] = ["prompt", "model"]

    differences = compare_documents(baseline, latest)

    assert any(
        "request application/json $/messages" in item and "<absent>" in item
        for item in differences
    )
    assert any(
        "request application/json $/prompt" in item and "<absent>" in item for item in differences
    )
    assert any(
        "request application/json $/metadata" in item and "<absent>" in item
        for item in differences
    )


def test_detects_nested_input_and_output_type_changes() -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    latest["components"]["schemas"]["Message"]["properties"]["content"]["type"] = "array"
    latest["components"]["schemas"]["Message"]["properties"]["content"]["items"] = {
        "type": "integer"
    }
    latest["components"]["schemas"]["StreamEvent"]["properties"]["type"]["type"] = "integer"

    differences = compare_documents(baseline, latest)

    assert any("$/messages[]/content" in item and "type changed" in item for item in differences)
    assert any("response 200 text/event-stream $/type" in item for item in differences)
    assert any("$/messages[]/content[]" in item and '"integer"' in item for item in differences)


def test_detects_composition_changes_without_treating_branch_order_as_significant() -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    variants = [
        {
            "type": "object",
            "properties": {
                "kind": {"const": "text"},
                "value": {"type": "string"},
            },
        },
        {
            "type": "object",
            "properties": {
                "kind": {"const": "tokens"},
                "value": {"type": "integer"},
            },
        },
    ]
    input_schema = baseline["components"]["schemas"]["ResponseRequest"]["properties"]["input"]
    input_schema["oneOf"] = variants
    latest = deepcopy(baseline)
    latest_variants = latest["components"]["schemas"]["ResponseRequest"]["properties"]["input"][
        "oneOf"
    ]
    latest_variants[0]["properties"]["value"]["type"] = "integer"
    latest_variants[1]["properties"]["value"]["type"] = "string"

    differences = compare_documents(baseline, latest)

    assert any("$/input" in item and "oneOf branch changed" in item for item in differences)


def test_normalizes_equivalent_type_nullable_and_composition_spellings() -> None:
    baseline = _document()
    content = baseline["components"]["schemas"]["Message"]["properties"]["content"]
    content["nullable"] = True
    latest = deepcopy(baseline)
    latest_content = latest["components"]["schemas"]["Message"]["properties"]["content"]
    latest_content.pop("nullable")
    latest_content["type"] = ["null", "string"]
    latest_content["allOf"] = [{"type": ["string", "null"]}]

    assert compare_documents(baseline, latest) == []


def test_detects_recursive_reference_target_and_pattern_property_changes() -> None:
    baseline = _document()
    response = baseline["components"]["schemas"]["Response"]
    response["properties"]["recursive"] = {"$recursiveRef": "#"}
    response["patternProperties"] = {"^x-": {"type": "string"}}
    latest = deepcopy(baseline)
    latest_response = latest["components"]["schemas"]["Response"]
    latest_response["properties"]["recursive"] = {"$recursiveRef": "#/components/schemas/Message"}
    latest_response["patternProperties"]["^x-"]["type"] = "integer"

    differences = compare_documents(baseline, latest)

    assert any("$/recursive" in item and "$recursiveRef changed" in item for item in differences)
    assert any("$/{pattern:^x-}" in item and "type changed" in item for item in differences)


def test_detects_operation_path_media_type_status_and_header_changes() -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    operation = latest["paths"].pop("/responses")["post"]
    latest["paths"]["/v2/responses"] = {"post": operation}
    response = operation["responses"].pop("200")
    response["headers"] = {"Request-ID": {"schema": {"type": "string"}}}
    del response["content"]["application/json"]
    operation["responses"]["201"] = response

    differences = compare_documents(baseline, latest)

    assert any("createResponse operation $: path changed" in item for item in differences)
    assert any("response statuses changed" in item for item in differences)
    assert any("media types changed" in item for item in differences)
    assert any("headers changed" in item for item in differences)


def test_component_renames_and_inline_schemas_do_not_change_the_contract() -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    message = latest["components"]["schemas"].pop("Message")
    latest["components"]["schemas"]["RenamedMessage"] = message
    latest["components"]["schemas"]["ChatRequest"]["properties"]["messages"]["items"] = (
        _schema_reference("RenamedMessage")
    )
    latest["components"]["schemas"]["ResponseRequest"]["properties"]["input"]["oneOf"][1] = message

    assert compare_documents(baseline, latest) == []


def test_handles_recursive_schemas() -> None:
    baseline = _document()
    compound_filter = {
        "$recursiveAnchor": True,
        "type": "object",
        "properties": {
            "filters": {
                "type": "array",
                "items": {"oneOf": [{"type": "string"}, {"$recursiveRef": "#"}]},
            }
        },
    }
    baseline["components"]["schemas"]["CompoundFilter"] = compound_filter
    baseline["components"]["schemas"]["ResponseRequest"]["properties"]["filter"] = (
        _schema_reference("CompoundFilter")
    )
    latest = deepcopy(baseline)
    latest["components"]["schemas"]["CompoundFilter"]["properties"]["filters"]["items"]["oneOf"][
        0
    ]["type"] = "integer"

    differences = compare_documents(baseline, latest)

    assert any("$/filter/filters[]" in item and "type changed" in item for item in differences)


def test_cli_exit_status_reports_contract_changes(tmp_path: Path, capsys) -> None:
    baseline = _document()
    latest = deepcopy(baseline)
    latest["components"]["schemas"]["Response"]["properties"]["usage"] = {
        "type": ["object", "null"]
    }
    baseline_path = tmp_path / "baseline.json"
    latest_path = tmp_path / "latest.json"
    baseline_path.write_text(json.dumps(baseline))
    latest_path.write_text(json.dumps(latest))

    assert main([str(baseline_path), str(baseline_path)]) == 0
    assert main([str(baseline_path), str(latest_path)]) == 1
    output = capsys.readouterr().out
    assert "programmatic contract changes detected" in output
    assert "response 200 application/json $/usage" in output
