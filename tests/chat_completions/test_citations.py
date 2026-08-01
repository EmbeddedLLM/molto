from tests.support import run_chat_parity

REQUEST_BODY = {
    "model": "gpt-5-search-api",
    "messages": [
        {
            "role": "user",
            "content": (
                "When was the OpenAI Responses API introduced? "
                "Answer in one sentence with an inline citation."
            ),
        }
    ],
    "extra_body": {"web_search_options": {}},
}


async def test_chat_url_citations_round_trip_through_sdk(cassette_manager) -> None:
    """Preserve the pinned SDK's nested Chat URL-citation contract."""

    parity = await run_chat_parity(cassette_manager, REQUEST_BODY)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    sdk_annotations = sdk_response.choices[0].message.annotations
    assert sdk_annotations
    sdk_payloads = [
        annotation.model_dump(mode="python", exclude_unset=True) for annotation in sdk_annotations
    ]
    assert all(
        annotation["type"] == "url_citation"
        and set(annotation["url_citation"]) == {"url", "title", "start_index", "end_index"}
        for annotation in sdk_payloads
    )

    assert response.choices[0].message.annotations == sdk_payloads
