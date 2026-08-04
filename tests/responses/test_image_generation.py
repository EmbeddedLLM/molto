import pytest
from openai.types.responses.response_image_gen_call_partial_image_event import (
    ResponseImageGenCallPartialImageEvent,
)
from openai.types.responses.response_output_item import ImageGenerationCall

from molto.types.openai.responses import (
    OpenAIImageGenerationCall,
    OpenAIResponsesStreamImageGeneration,
)
from tests.support import (
    STREAM_PARAMS,
    prepend_model_prefix,
    run_responses_parity,
)

REQUEST_BODY = {
    "model": "gpt-5.1",
    "input": "Generate a simple flat icon of a red circle on a transparent background.",
    "tools": [
        {
            "type": "image_generation",
            "background": "auto",
            "output_format": "png",
            "quality": "low",
            "size": "1024x1024",
        }
    ],
    "tool_choice": {"type": "image_generation"},
}
EDIT_TOOL = {**REQUEST_BODY["tools"][0], "action": "edit"}


async def _generate_image_for_continuation(
    cassette_manager,
    *,
    store: bool | None,
) -> ImageGenerationCall:
    setup_body = dict(REQUEST_BODY)
    if store is not None:
        setup_body["store"] = store

    parity = await run_responses_parity(
        cassette_manager,
        setup_body,
        cassette_suffix="setup",
    )
    sdk_image = next(
        item for item in parity.sdk_response.output if isinstance(item, ImageGenerationCall)
    )
    assert sdk_image.result
    return sdk_image


@pytest.mark.parametrize("stream", **STREAM_PARAMS)
async def test_image_generation_round_trip_through_sdk(
    stream: bool,
    cassette_manager,
) -> None:
    """Preserve complete image-generation output and events through the public IR route."""
    req_body = {**REQUEST_BODY, "stream": stream}
    if stream:
        req_body["tools"] = [{**REQUEST_BODY["tools"][0], "partial_images": 1}]

    parity = await run_responses_parity(cassette_manager, req_body)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    if stream:
        sdk_partial_events = [
            event
            for event in parity.sdk_events
            if isinstance(event, ResponseImageGenCallPartialImageEvent)
        ]
        assert sdk_partial_events
        assert all(event.partial_image_b64 for event in sdk_partial_events)

    sdk_image = next(item for item in sdk_response.output if isinstance(item, ImageGenerationCall))
    assert sdk_image.status == ("generating" if stream else "completed")
    assert sdk_image.result

    if stream:
        partial_events = [
            event
            for event in parity.molto_events
            if isinstance(event, OpenAIResponsesStreamImageGeneration)
            and event.type == "response.image_generation_call.partial_image"
        ]
        assert [
            (event.partial_image_b64, event.partial_image_index) for event in partial_events
        ] == [(event.partial_image_b64, event.partial_image_index) for event in sdk_partial_events]

    image = next(item for item in response.output if isinstance(item, OpenAIImageGenerationCall))
    assert image.model_dump(mode="python", exclude_unset=True) == sdk_image.model_dump(
        mode="python",
        exclude_unset=True,
    )


async def test_image_generation_output_history_round_trip_through_sdk(cassette_manager) -> None:
    """Continue image generation with the complete prior output item."""
    sdk_image = await _generate_image_for_continuation(cassette_manager, store=True)
    complete_image_history = sdk_image.model_dump(mode="python", exclude_unset=True)
    # The legal projection keeps the `id`: the API requires it on
    # image-generation input items and resolves it against the persisted item,
    # so the setup request must leave `store` enabled.
    image_input = {
        "type": sdk_image.type,
        "id": sdk_image.id,
        "result": sdk_image.result,
        "status": sdk_image.status,
    }
    user_input = {"role": "user", "content": "Change the circle from red to blue."}
    request_body = {
        "model": REQUEST_BODY["model"],
        "input": [user_input, image_input],
        "tools": [EDIT_TOOL],
        "tool_choice": {"type": "image_generation"},
    }
    # The SDK oracle receives the legal input projection. Molto receives the
    # complete prior output and must reconstruct that same canonical request.
    molto_request_body = prepend_model_prefix(request_body)
    molto_request_body["input"] = [user_input, complete_image_history]

    parity = await run_responses_parity(
        cassette_manager,
        request_body,
        cassette_suffix="continuation",
        molto_request_body=molto_request_body,
    )

    continued_image = next(
        item for item in parity.sdk_response.output if isinstance(item, ImageGenerationCall)
    )
    assert continued_image.result


async def test_image_generation_call_id_continuation_round_trip_through_sdk(
    cassette_manager,
) -> None:
    """Continue image generation with OpenAI's documented image-call ID item."""
    sdk_image = await _generate_image_for_continuation(cassette_manager, store=None)
    request_body = {
        "model": REQUEST_BODY["model"],
        "input": [
            {"role": "user", "content": "Change the circle from red to blue."},
            {"type": "image_generation_call", "id": sdk_image.id},
        ],
        "tools": [EDIT_TOOL],
        "tool_choice": {"type": "image_generation"},
    }

    parity = await run_responses_parity(
        cassette_manager,
        request_body,
        cassette_suffix="continuation",
    )

    continued_image = next(
        item for item in parity.sdk_response.output if isinstance(item, ImageGenerationCall)
    )
    assert continued_image.result
