import base64
import io
import wave

from molto.types.openai.chat_completions import OpenAIChatCompletionRequest
from tests.support import (
    make_molto_client,
    make_openai_client,
    prepend_model_prefix,
    run_chat_parity,
)

REQUEST_BODY = {
    "model": "gpt-audio",
    "messages": [{"role": "user", "content": "Say exactly: hello from Molto"}],
    "modalities": ["text", "audio"],
    "audio": {"format": "mp3", "voice": "alloy"},
}


def _silent_wav_base64() -> str:
    """Return a deterministic 100 ms mono WAV payload for request parity."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\x00\x00" * 800)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def test_generated_audio_round_trip_through_sdk(cassette_manager) -> None:
    """Preserve complete parsed SDK audio through both mandatory IR boundaries."""
    parity = await run_chat_parity(cassette_manager, REQUEST_BODY)
    sdk_response = parity.sdk_response
    response = parity.molto_response

    sdk_audio = sdk_response.choices[0].message.audio
    assert sdk_audio is not None
    assert sdk_audio.id
    assert sdk_audio.data
    assert sdk_audio.expires_at > 0
    assert sdk_audio.transcript

    assert response.choices[0].message.audio is not None
    assert response.choices[0].message.audio.model_dump(mode="python") == sdk_audio.model_dump(
        mode="python"
    )


async def test_input_audio_round_trip_through_sdk(cassette_manager) -> None:
    """Preserve inline Chat audio input through the public request and both IR boundaries."""
    request_body = {
        "model": "gpt-audio",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _silent_wav_base64(),
                            "format": "wav",
                        },
                    },
                    {"type": "text", "text": "Reply with a short description of this audio."},
                ],
            }
        ],
    }
    await run_chat_parity(cassette_manager, request_body)


async def test_previous_audio_reference_round_trip_through_sdk(cassette_manager) -> None:
    """Preserve an assistant audio reference returned by an approved setup request."""
    setup_client = make_openai_client(
        cassette_manager,
        cassette_suffix="setup",
        cassette_endpoint="chat_completions",
    )
    setup_response = await setup_client.chat.completions.create(**REQUEST_BODY)
    setup_audio = setup_response.choices[0].message.audio
    assert setup_audio is not None
    assert setup_audio.id

    molto_setup = make_molto_client(
        cassette_manager,
        cassette_suffix="setup",
    )
    molto_setup_response = await molto_setup.chat_completions(
        OpenAIChatCompletionRequest(**prepend_model_prefix(REQUEST_BODY))
    )
    molto_setup_audio = molto_setup_response.choices[0].message.audio
    assert molto_setup_audio is not None
    assert molto_setup_audio.id == setup_audio.id

    request_body = {
        "model": "gpt-audio",
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "audio": {"id": molto_setup_audio.id},
            },
            {
                "role": "user",
                "content": "Reply with the transcript of the previous audio.",
            },
        ],
    }
    await run_chat_parity(cassette_manager, request_body, cassette_suffix="continuation")
