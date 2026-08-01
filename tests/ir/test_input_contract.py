import pytest
from pydantic import ValidationError

from molto.types.ir.portable import (
    AllowedFunctionTools,
    FunctionCallRequestItem,
    FunctionCallResultItem,
    FunctionToolChoice,
    FunctionToolDefinition,
    InputIR,
    InputTextPart,
    RequestMessageItem,
)


def test_stream_selection_is_mandatory_and_boolean() -> None:
    assert InputIR(model="test/model", stream=True, items=()).stream is True
    with pytest.raises(ValidationError, match="stream"):
        InputIR.model_validate({"model": "test/model", "items": []})


def test_function_tool_choices_require_unique_declared_names() -> None:
    with pytest.raises(ValidationError, match="undeclared tools: missing"):
        InputIR(
            model="test/model",
            stream=False,
            items=(),
            tools=(FunctionToolDefinition(name="lookup", parameters={}),),
            tool_choice=FunctionToolChoice(name="missing"),
        )

    with pytest.raises(ValidationError, match="must be unique"):
        AllowedFunctionTools(
            mode="required",
            tools=(FunctionToolChoice(name="lookup"), FunctionToolChoice(name="lookup")),
        )


def test_function_call_history_rejects_more_than_one_result() -> None:
    with pytest.raises(ValidationError, match="more than one result"):
        InputIR(
            model="test/model",
            stream=False,
            items=(
                FunctionCallRequestItem(call_id="call_1", name="lookup", arguments="{}"),
                FunctionCallResultItem(call_id="call_1", output="first"),
                FunctionCallResultItem(call_id="call_1", output="second"),
            ),
        )


def test_request_item_ids_and_roles_round_trip() -> None:
    request = InputIR(
        model="test/model",
        stream=False,
        items=(
            RequestMessageItem(
                id="message_1",
                role="user",
                content=(InputTextPart(text="hello"),),
            ),
        ),
    )

    assert InputIR.model_validate_json(request.model_dump_json()) == request
