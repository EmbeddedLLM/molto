#!/usr/bin/env python3
"""Compare the programmatic OpenAPI surface of OpenAI's chat APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

TARGET_OPERATION_IDS = ("createChatCompletion", "createResponse")
HTTP_METHODS = frozenset({"delete", "get", "head", "options", "patch", "post", "put", "trace"})
SCHEMA_ATTRIBUTES = (
    "contentEncoding",
    "contentMediaType",
    "format",
)
SCHEMA_FLAGS = ("readOnly", "writeOnly")

JsonObject = dict[str, Any]
FactKey = tuple[str, tuple[str, ...], str]
Contract = dict[FactKey, frozenset[str]]


class OpenAPIError(ValueError):
    """Raised when a target OpenAPI operation cannot be compared."""


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _resolve_pointer(document: JsonObject, reference: str) -> Any:
    if not reference.startswith("#/"):
        raise OpenAPIError(f"Only local OpenAPI references are supported: {reference!r}")

    value: Any = document
    for token in reference[2:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        try:
            value = value[token]
        except (KeyError, TypeError) as exc:
            raise OpenAPIError(f"OpenAPI reference does not exist: {reference!r}") from exc
    return value


def _resolve_object(document: JsonObject, value: Any) -> JsonObject:
    if not isinstance(value, dict):
        raise OpenAPIError("Expected an OpenAPI object.")

    resolved: JsonObject = {}
    seen: set[str] = set()
    current = value
    while isinstance(reference := current.get("$ref"), str):
        if reference in seen:
            raise OpenAPIError(f"Circular non-schema reference: {reference!r}")
        seen.add(reference)
        target = _resolve_pointer(document, reference)
        if not isinstance(target, dict):
            raise OpenAPIError(f"OpenAPI reference is not an object: {reference!r}")
        resolved = {**target, **{key: item for key, item in current.items() if key != "$ref"}}
        current = resolved
    return current


def _add_fact(
    facts: defaultdict[FactKey, set[str]],
    surface: str,
    path: tuple[str, ...],
    attribute: str,
    value: Any,
) -> None:
    facts[(surface, path, attribute)].add(_json_value(value))


def _collect_schema_facts(
    document: JsonObject,
    schema: Any,
    facts: defaultdict[FactKey, set[str]],
    surface: str,
    path: tuple[str, ...] = (),
    active_references: tuple[str, ...] = (),
    collect_branch_signatures: bool = True,
) -> None:
    if isinstance(schema, bool):
        _add_fact(facts, surface, path, "schema", schema)
        return
    if not isinstance(schema, dict):
        raise OpenAPIError(f"Schema at {surface} must be an object or boolean.")

    _add_fact(facts, surface, path, "field", True)

    reference = schema.get("$ref")
    if isinstance(reference, str):
        if reference in active_references:
            _add_fact(facts, surface, path, "recursive", True)
        else:
            _collect_schema_facts(
                document,
                _resolve_pointer(document, reference),
                facts,
                surface,
                path,
                (*active_references, reference),
                collect_branch_signatures,
            )
        schema = {key: value for key, value in schema.items() if key != "$ref"}

    if "$recursiveRef" in schema or "$dynamicRef" in schema:
        for attribute in ("$recursiveRef", "$dynamicRef"):
            if attribute in schema:
                _add_fact(facts, surface, path, attribute, schema[attribute])

    for attribute in SCHEMA_ATTRIBUTES:
        if attribute not in schema:
            continue
        _add_fact(facts, surface, path, attribute, schema[attribute])
    if "type" in schema or schema.get("nullable") is True:
        declared_types = schema.get("type", [])
        if isinstance(declared_types, str):
            declared_types = [declared_types]
        if not isinstance(declared_types, list):
            raise OpenAPIError(f"Schema type at {surface} must be a string or array.")
        if schema.get("nullable") is True:
            declared_types = [*declared_types, "null"]
        _add_fact(facts, surface, path, "type", sorted(set(declared_types)))
    if "enum" in schema or "const" in schema:
        allowed_values = schema.get("enum", [])
        if not isinstance(allowed_values, list):
            raise OpenAPIError(f"Schema enum at {surface} must be an array.")
        if "const" in schema:
            allowed_values = [*allowed_values, schema["const"]]
        _add_fact(
            facts,
            surface,
            path,
            "allowed values",
            sorted({_json_value(value) for value in allowed_values}),
        )
    for attribute in SCHEMA_FLAGS:
        if schema.get(attribute) is True:
            _add_fact(facts, surface, path, attribute, True)

    discriminator = schema.get("discriminator")
    if isinstance(discriminator, dict) and "propertyName" in discriminator:
        _add_fact(
            facts,
            surface,
            path,
            "discriminator",
            discriminator["propertyName"],
        )

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        raise OpenAPIError(f"Schema properties at {surface} must be an object.")
    for name, property_schema in properties.items():
        _collect_schema_facts(
            document,
            property_schema,
            facts,
            surface,
            (*path, name),
            active_references,
            collect_branch_signatures,
        )

    pattern_properties = schema.get("patternProperties", {})
    if not isinstance(pattern_properties, dict):
        raise OpenAPIError(f"Schema patternProperties at {surface} must be an object.")
    for pattern, property_schema in pattern_properties.items():
        _collect_schema_facts(
            document,
            property_schema,
            facts,
            surface,
            (*path, f"{{pattern:{pattern}}}"),
            active_references,
            collect_branch_signatures,
        )

    required = schema.get("required", [])
    if not isinstance(required, list):
        raise OpenAPIError(f"Schema required fields at {surface} must be an array.")
    for name in required:
        _add_fact(facts, surface, (*path, str(name)), "required", True)

    if "items" in schema:
        _collect_schema_facts(
            document,
            schema["items"],
            facts,
            surface,
            (*path, "[]"),
            active_references,
            collect_branch_signatures,
        )
    for index, item in enumerate(schema.get("prefixItems", [])):
        _collect_schema_facts(
            document,
            item,
            facts,
            surface,
            (*path, f"[{index}]"),
            active_references,
            collect_branch_signatures,
        )

    for keyword in ("additionalProperties", "unevaluatedProperties"):
        additional_properties = schema.get(keyword, True)
        if isinstance(additional_properties, (dict, bool)):
            if additional_properties is False:
                _add_fact(facts, surface, path, keyword, False)
            elif isinstance(additional_properties, dict) and additional_properties:
                marker = "{*}" if keyword == "additionalProperties" else "{unevaluated:*}"
                _collect_schema_facts(
                    document,
                    additional_properties,
                    facts,
                    surface,
                    (*path, marker),
                    active_references,
                    collect_branch_signatures,
                )

    for keyword in ("allOf", "anyOf", "oneOf"):
        branches = schema.get(keyword, [])
        if not isinstance(branches, list):
            raise OpenAPIError(f"Schema {keyword} at {surface} must be an array.")
        branch_signatures = (
            {_schema_signature(document, branch, active_references) for branch in branches}
            if collect_branch_signatures and keyword in {"anyOf", "oneOf"}
            else set()
        )
        if len(branch_signatures) > 1:
            for signature in branch_signatures:
                _add_fact(facts, surface, path, f"{keyword} branch", signature)
        for branch in branches:
            _collect_schema_facts(
                document,
                branch,
                facts,
                surface,
                path,
                active_references,
                collect_branch_signatures,
            )
    if "not" in schema:
        _collect_schema_facts(
            document,
            schema["not"],
            facts,
            f"{surface} not",
            path,
            active_references,
            collect_branch_signatures,
        )


def _schema_signature(
    document: JsonObject,
    schema: Any,
    active_references: tuple[str, ...],
) -> str:
    branch_facts: defaultdict[FactKey, set[str]] = defaultdict(set)
    _collect_schema_facts(
        document,
        schema,
        branch_facts,
        "branch",
        active_references=active_references,
        collect_branch_signatures=False,
    )
    entries = [
        (surface, path, attribute, sorted(values))
        for (surface, path, attribute), values in sorted(branch_facts.items())
    ]
    return hashlib.sha256(_json_value(entries).encode()).hexdigest()


def _find_operation(
    document: JsonObject, operation_id: str
) -> tuple[str, str, JsonObject, JsonObject]:
    matches: list[tuple[str, str, JsonObject, JsonObject]] = []
    paths = document.get("paths")
    if not isinstance(paths, dict):
        raise OpenAPIError("OpenAPI document must contain a paths object.")

    for path, path_item_value in paths.items():
        if not isinstance(path_item_value, dict):
            continue
        path_item = _resolve_object(document, path_item_value)
        for method, operation in path_item.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            operation = _resolve_object(document, operation)
            if operation.get("operationId") == operation_id:
                matches.append((path, method.lower(), path_item, operation))

    if len(matches) != 1:
        raise OpenAPIError(
            f"Expected exactly one {operation_id!r} operation, found {len(matches)}."
        )
    return matches[0]


def _collect_media_types(
    document: JsonObject,
    content: Any,
    facts: defaultdict[FactKey, set[str]],
    surface: str,
) -> None:
    if not isinstance(content, dict):
        raise OpenAPIError(f"Content at {surface} must be an object.")
    _add_fact(facts, surface, (), "media types", sorted(content))
    for media_type, media in content.items():
        if not isinstance(media, dict):
            raise OpenAPIError(f"Media type {media_type!r} at {surface} must be an object.")
        if "schema" in media:
            _collect_schema_facts(document, media["schema"], facts, f"{surface} {media_type}")


def _parameter_identity(parameter: JsonObject) -> tuple[str, str]:
    name = parameter.get("name")
    location = parameter.get("in")
    if not isinstance(name, str) or not isinstance(location, str):
        raise OpenAPIError("Each parameter must have string name and in fields.")
    if location == "header":
        name = name.lower()
    return location, name


def _collect_parameters(
    document: JsonObject,
    path_parameters: Any,
    operation_parameters: Any,
    facts: defaultdict[FactKey, set[str]],
) -> None:
    parameters: dict[tuple[str, str], JsonObject] = {}
    for parameter_list in (path_parameters or [], operation_parameters or []):
        if not isinstance(parameter_list, list):
            raise OpenAPIError("Operation parameters must be an array.")
        for parameter_value in parameter_list:
            parameter = _resolve_object(document, parameter_value)
            parameters[_parameter_identity(parameter)] = parameter

    _add_fact(facts, "operation", (), "parameters", sorted(parameters))
    for (location, name), parameter in parameters.items():
        surface = f"parameter {location} {name}"
        default_style = "form" if location in {"cookie", "query"} else "simple"
        style = parameter.get("style", default_style)
        defaults = {
            "allowEmptyValue": False,
            "allowReserved": False,
            "explode": style == "form",
            "required": False,
            "style": default_style,
        }
        for attribute, default in defaults.items():
            _add_fact(facts, surface, (), attribute, parameter.get(attribute, default))
        if "schema" in parameter:
            _collect_schema_facts(document, parameter["schema"], facts, surface)
        if "content" in parameter:
            _collect_media_types(document, parameter["content"], facts, surface)


def _collect_headers(
    document: JsonObject,
    headers: Any,
    facts: defaultdict[FactKey, set[str]],
    surface: str,
) -> None:
    if not isinstance(headers, dict):
        raise OpenAPIError(f"Headers at {surface} must be an object.")
    normalized_headers = {name.lower(): value for name, value in headers.items()}
    _add_fact(facts, surface, (), "headers", sorted(normalized_headers))
    for name, header_value in normalized_headers.items():
        header = _resolve_object(document, header_value)
        header_surface = f"{surface} header {name}"
        for attribute, default in {"explode": False, "required": False, "style": "simple"}.items():
            _add_fact(facts, header_surface, (), attribute, header.get(attribute, default))
        if "schema" in header:
            _collect_schema_facts(document, header["schema"], facts, header_surface)
        if "content" in header:
            _collect_media_types(document, header["content"], facts, header_surface)


def _operation_contract(document: JsonObject, operation_id: str) -> Contract:
    path, method, path_item, operation = _find_operation(document, operation_id)
    facts: defaultdict[FactKey, set[str]] = defaultdict(set)
    _add_fact(facts, "operation", (), "path", path)
    _add_fact(facts, "operation", (), "method", method)
    _collect_parameters(
        document,
        path_item.get("parameters"),
        operation.get("parameters"),
        facts,
    )

    if "requestBody" in operation:
        request_body = _resolve_object(document, operation["requestBody"])
        _add_fact(facts, "request", (), "required", request_body.get("required", False))
        _collect_media_types(document, request_body.get("content", {}), facts, "request")
    else:
        _add_fact(facts, "operation", (), "request body", False)

    responses = operation.get("responses")
    if not isinstance(responses, dict):
        raise OpenAPIError(f"Operation {operation_id!r} must contain responses.")
    _add_fact(facts, "operation", (), "response statuses", sorted(responses))
    for status, response_value in responses.items():
        response = _resolve_object(document, response_value)
        surface = f"response {status}"
        _collect_media_types(document, response.get("content", {}), facts, surface)
        _collect_headers(document, response.get("headers", {}), facts, surface)

    return {key: frozenset(values) for key, values in facts.items()}


def compare_documents(baseline: JsonObject, latest: JsonObject) -> list[str]:
    """Return stable diagnostics for programmatic contract changes."""
    differences: list[str] = []
    for operation_id in TARGET_OPERATION_IDS:
        baseline_contract = _operation_contract(baseline, operation_id)
        latest_contract = _operation_contract(latest, operation_id)
        for key in sorted(baseline_contract.keys() | latest_contract.keys()):
            baseline_values = baseline_contract.get(key, frozenset())
            latest_values = latest_contract.get(key, frozenset())
            if baseline_values == latest_values:
                continue
            surface, path, attribute = key
            location = _format_location(operation_id, surface, path)
            differences.append(
                f"{location}: {attribute} changed from "
                f"{_format_values(baseline_values)} to {_format_values(latest_values)}"
            )
    return differences


def _format_location(operation_id: str, surface: str, path: tuple[str, ...]) -> str:
    pointer = "".join(
        token if token in {"[]", "{*}"} or token.startswith("[") else f"/{token}" for token in path
    )
    return f"{operation_id} {surface} ${pointer}"


def _format_values(values: frozenset[str]) -> str:
    if not values:
        return "<absent>"
    return ", ".join(sorted(values))


def _load_document(path: Path) -> JsonObject:
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenAPIError(f"Could not read OpenAPI document {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise OpenAPIError(f"OpenAPI document {path} must contain a JSON object.")
    return document


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the request and response fields, paths, and types for OpenAI Chat "
            "Completions and Responses. Documentation-only changes are ignored."
        )
    )
    parser.add_argument("baseline", type=Path)
    parser.add_argument("latest", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        differences = compare_documents(
            _load_document(args.baseline),
            _load_document(args.latest),
        )
    except OpenAPIError as exc:
        print(f"OpenAPI comparison failed: {exc}", file=sys.stderr)
        return 2

    if differences:
        print("OpenAI API programmatic contract changes detected:")
        for difference in differences:
            print(f"- {difference}")
        return 1

    print("Chat Completions and Responses programmatic contracts are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
