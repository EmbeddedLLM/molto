"""Mocked native-SDK recording scenarios; no live provider calls or committed fixture writes."""

import argparse
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import httpx
import pytest
from openai.types.chat import ChatCompletionMessage

from scripts import record_provider_conversation as driver
from scripts import recorder_proxy


def tool_call(identity, a, b, **extra):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": "add", "arguments": json.dumps({"a": a, "b": b})},
        **extra,
    }


def completion(message, finish=None):
    return {
        "id": "chatcmpl-recording",
        "object": "chat.completion",
        "created": 1,
        "model": "returned-model",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish or ("tool_calls" if message.get("tool_calls") else "stop"),
            }
        ],
    }


def streaming_response(
    message, finish=None, *, missing_index=False, repeat_extension=False, repeated_role=False
):
    delta = deepcopy(message)
    for index, call in enumerate(delta.get("tool_calls") or []):
        call["index"] = None if missing_index else index
    deltas = [delta]
    if repeated_role:
        deltas = [
            {"role": "assistant", "reasoning": " First"},
            {"role": "assistant", "reasoning": " thought\n"},
            delta,
        ]
    if repeat_extension:
        deltas.append({"extra_content": deepcopy(message["extra_content"])})
    chunks = []
    for item in [*deltas, {}]:
        chunk = {
            "id": "chatcmpl-recording",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "returned-model",
            "choices": [{"index": 0, "delta": item, "finish_reason": None}],
        }
        if not item:
            chunk["choices"][0]["finish_reason"] = finish or (
                "tool_calls" if message.get("tool_calls") else "stop"
            )
        chunks.append("data: " + json.dumps(chunk) + "\n\n")
    return httpx.Response(
        200,
        text="".join(chunks) + "data: [DONE]\n\n",
        headers={"content-type": "text/event-stream"},
    )


@pytest.fixture
def recording_args(monkeypatch, tmp_path):
    """Isolate provider credentials and staging from the developer environment."""
    monkeypatch.setattr(driver, "load_dotenv", lambda **_: None)
    monkeypatch.setenv("TEST_RECORDING_KEY", "test-provider-credential")
    return argparse.Namespace(
        key_env="TEST_RECORDING_KEY",
        provider="synthetic",
        model="requested-model",
        base_url="https://provider.test/prefix/v1",
        output_dir=tmp_path / "run",
        scenario="text",
        stream=False,
        max_tokens=512,
        extra_body=None,
    )


async def run_recording(monkeypatch, args, messages, **stream_options):
    requests = []

    def handler(request):
        assert str(request.url) == "https://provider.test/prefix/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-provider-credential"
        requests.append(json.loads(request.content))
        message = messages[len(requests) - 1]
        if args.stream:
            return streaming_response(message, **stream_options)
        return httpx.Response(200, json=completion(message))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        monkeypatch.setattr(recorder_proxy, "HTTP_CLIENT", upstream)
        code = await driver.record(args)
        assert upstream.is_closed
    return code, json.loads((args.output_dir / "manifest.json").read_text()), requests


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("scenario", ["text", "tools", "parallel"])
async def test_complete_native_conversations_carry_actual_history(
    monkeypatch, recording_args, stream, scenario
):
    """Ensures SDK output, reasoning, opaque state, calls, and actual linked results reach later turns."""
    args = recording_args
    args.stream, args.scenario = stream, scenario
    opaque = {"google": {"thought_signature": "opaque-message"}}
    first = {
        "role": "assistant",
        "content": "Keep  exact\nwording",
        "reasoning": " Think\n ",
        "extra_content": opaque,
    }
    messages = [first, {"role": "assistant", "content": "Final  text."}]
    if scenario != "text":
        first["tool_calls"] = [
            tool_call(
                "call-one",
                2,
                3,
                extra_content={"google": {"thought_signature": "opaque-call-one"}},
            )
        ]
        if scenario == "parallel":
            first["tool_calls"].append(
                tool_call(
                    "call-two",
                    7,
                    4,
                    extra_content={"google": {"thought_signature": "opaque-call-two"}},
                )
            )
        else:
            messages.insert(
                1,
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call("call-two", 5, 4)],
                },
            )
    code, manifest, requests = await run_recording(monkeypatch, args, messages)

    assert code == 0
    assert manifest["status"] == "complete"
    assert len(requests) == (3 if scenario == "tools" else 2)
    for turn, info in enumerate(manifest["turns"]):
        assert info["status"] == "complete"
        assert info["returned_model"] == "returned-model"
        assert len(info["files"]) == 1
        assert (args.output_dir / info["files"][0]).exists()
        assert requests[turn]["stream"] is stream
    expected_first = deepcopy(first)
    if stream:
        # The pinned SDK's snapshot explicitly materializes these null placeholders.
        for field in ("refusal", "tool_calls"):
            expected_first.setdefault(field, None)
    assert requests[1]["messages"][1] == expected_first
    if scenario == "text":
        assert len(requests[0]["messages"][0]["content"]) == 2
        assert len(requests[1]["messages"][-1]["content"]) == 2
    else:
        assert requests[1]["messages"][2] == {
            "role": "tool",
            "tool_call_id": "call-one",
            "content": "5",
        }
        if scenario == "parallel":
            assert requests[0]["parallel_tool_calls"] is True
            assert requests[0]["tool_choice"] == "required"
            assert requests[1]["tool_choice"] == "none"
            assert requests[1]["messages"][3] == {
                "role": "tool",
                "tool_call_id": "call-two",
                "content": "11",
            }
        else:
            expected_second = deepcopy(messages[1])
            if stream:
                for field in ("refusal",):
                    expected_second.setdefault(field, None)
            assert requests[2]["messages"][3] == expected_second
            assert requests[2]["messages"][4] == {
                "role": "tool",
                "tool_call_id": "call-two",
                "content": "9",
            }
    assert manifest["source_hashes"] == {
        "driver_sha256": sha256(Path(driver.__file__).read_bytes()).hexdigest(),
        "recorder_sha256": sha256(Path(recorder_proxy.__file__).read_bytes()).hexdigest(),
    }
    assert set(manifest["versions"]) == {"openai", "anthropic", "httpx"}
    assert manifest["upstream_url"] == args.base_url + "/chat/completions"
    assert manifest["sdk_retries"] == manifest["proxy_retries"] == 0


@pytest.mark.parametrize(
    ("scenario", "messages", "expected_requests"),
    [
        ("tools", [{"role": "assistant", "content": "No tool."}], 1),
        (
            "tools",
            [
                {"role": "assistant", "tool_calls": [tool_call("first", 2, 3)]},
                {"role": "assistant", "content": "Skipped second call."},
            ],
            2,
        ),
        ("parallel", [{"role": "assistant", "tool_calls": [tool_call("first", 2, 3)]}], 1),
        (
            "parallel",
            [
                {
                    "role": "assistant",
                    "tool_calls": [tool_call("same", 2, 3), tool_call("same", 7, 4)],
                }
            ],
            1,
        ),
        ("tools", [{"role": "assistant", "tool_calls": [tool_call("first", 2, 9)]}], 1),
        ("text", [{"role": "assistant", "content": ""}], 1),
    ],
)
async def test_incomplete_scenarios_remain_failed_evidence(
    monkeypatch, recording_args, scenario, messages, expected_requests
):
    """Ensures missing calls, wrong arguments, duplicate identities, or empty answers cannot certify a scenario."""
    recording_args.scenario = scenario
    code, manifest, requests = await run_recording(monkeypatch, recording_args, messages)
    assert code == 1
    assert manifest["status"] == "incomplete"
    assert manifest["failure"]["type"] == "ValueError"
    assert manifest["failure"]["traceback"]
    assert len(requests) == expected_requests
    assert manifest["turns"][-1]["status"] == "incomplete"
    assert len(manifest["turns"][-1]["files"]) == 1


@pytest.mark.parametrize("missing_index", [False, True])
async def test_sdk_accumulation_failures_retain_original_capture(
    monkeypatch, recording_args, missing_index
):
    """Ensures missing indices or corrupted opaque snapshots fail without repairing provider output."""
    args = recording_args
    args.stream = True
    args.scenario = "tools" if missing_index else "text"
    message = {
        "role": "assistant",
        "content": "Answer",
        "extra_content": {"google": {"thought_signature": "opaque"}},
    }
    if missing_index:
        message["tool_calls"] = [tool_call("first", 2, 3)]
    code, manifest, requests = await run_recording(
        monkeypatch,
        args,
        [message],
        missing_index=missing_index,
        repeat_extension=not missing_index,
    )
    assert code == 1
    assert manifest["status"] == "incomplete"
    assert len(requests) == 1
    assert len(manifest["turns"][0]["files"]) == 1
    assert manifest["failure"]["type"] in {"TypeError", "ValueError"}


def test_history_cleanup_preserves_populated_metadata_and_source():
    """Ensures bookkeeping removal does not erase populated annotations, opaque state, or mutate SDK output."""
    source = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": "Answer",
            "parsed": {"internal": True},
            "annotations": [
                {
                    "type": "url_citation",
                    "url_citation": {
                        "url": "https://example.test",
                        "title": "Example",
                        "start_index": 0,
                        "end_index": 6,
                    },
                }
            ],
            "tool_calls": [
                {
                    "index": 0,
                    **tool_call(
                        "call", 2, 3, extra_content={"google": {"thought_signature": "signature"}}
                    ),
                }
            ],
        }
    )
    before = source.model_dump(exclude_unset=True)
    history = driver.assistant_history(source)
    assert "parsed" not in history
    assert "index" not in history["tool_calls"][0]
    assert history["annotations"] == before["annotations"]
    assert history["tool_calls"][0]["extra_content"] == before["tool_calls"][0]["extra_content"]
    assert source.model_dump(exclude_unset=True) == before


async def test_manifest_masks_configured_key_and_keeps_settings(monkeypatch, recording_args):
    """Ensures persisted provider settings cannot echo the configured authentication credential."""
    recording_args.extra_body = {
        "reasoning": {"enabled": False},
        "test_echo": "test-provider-credential",
    }
    messages = [
        {"role": "assistant", "content": "First."},
        {"role": "assistant", "content": "Second."},
    ]
    code, manifest, _ = await run_recording(monkeypatch, recording_args, messages)
    assert code == 0
    assert manifest["extra_body"] == {"reasoning": {"enabled": False}, "test_echo": "***"}
    assert (
        "test-provider-credential" not in (recording_args.output_dir / "manifest.json").read_text()
    )


@pytest.mark.parametrize("finish_reason", ["length", "tool_calls"])
def test_text_turn_requires_successful_text_termination(finish_reason):
    """Ensures incomplete or tool-terminated text cannot be labeled a completed text turn."""
    with pytest.raises(ValueError, match="complete"):
        driver.validate_turn(
            {"role": "assistant", "content": "Partial text"}, finish_reason, "text", 0
        )


def test_final_tool_turn_rejects_unresolved_calls():
    """Ensures a third sequential turn with another tool call is retained as incomplete evidence."""
    with pytest.raises(ValueError, match="number"):
        driver.validate_turn(
            {"role": "assistant", "tool_calls": [tool_call("third", 9, 1)]},
            "tool_calls",
            "tools",
            2,
        )


async def test_repeated_role_snapshots_preserve_original_recording_and_history(
    monkeypatch, recording_args
):
    """Ensures demonstrated Together role repetition reconstructs one assistant without editing evidence."""
    args = recording_args
    args.stream = True
    messages = [
        {"role": "assistant", "content": "Actual first answer."},
        {"role": "assistant", "content": "Actual second answer."},
    ]
    code, manifest, requests = await run_recording(monkeypatch, args, messages, repeated_role=True)
    assert code == 0
    assert requests[1]["messages"][1]["role"] == "assistant"
    assert requests[1]["messages"][1]["content"] == messages[0]["content"]
    assert requests[1]["messages"][1]["reasoning"] == " First thought\n"
    assert "role snapshots omitted on copies" in manifest["accumulation_policy"]
    from yaml import safe_load

    capture = safe_load((args.output_dir / manifest["turns"][0]["files"][0]).read_text())
    assert "".join(capture["response"]["sse"]).count('"role": "assistant"') == 3


def test_accumulation_rejects_changing_roles_without_mutating_chunks():
    """Ensures reconstruction only removes identical role snapshots and rejects identity changes."""
    from openai.types.chat import ChatCompletionChunk

    original = ChatCompletionChunk.model_validate(
        {
            "id": "chat",
            "created": 1,
            "model": "m",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "text"},
                    "finish_reason": None,
                }
            ],
        }
    )
    before = original.model_dump()
    copy = driver.accumulation_chunk(original, {0: "assistant"})
    assert "role" not in copy.choices[0].delta.model_dump(exclude_unset=True)
    assert original.model_dump() == before
    with pytest.raises(ValueError, match="changed the role"):
        driver.accumulation_chunk(original, {0: "user"})


@pytest.mark.parametrize("field", ["annotations", "audio", "function_call"])
@pytest.mark.parametrize("emitted", [False, True])
def test_history_omits_only_unemitted_sdk_null_bookkeeping(field, emitted):
    """Ensures only absent SDK-created nulls are removed and native emitted nulls remain unchanged."""
    from openai.types.chat import ChatCompletionMessage

    message = ChatCompletionMessage.model_validate(
        {
            "role": "assistant",
            "content": "Answer",
            "annotations": None,
            "audio": None,
            "function_call": None,
        }
    )
    original = message.model_dump()
    delta = {"role": "assistant", "content": "Answer"}
    if emitted:
        delta[field] = None
    history = driver.assistant_history(message, [{"choices": [{"index": 0, "delta": delta}]}])
    assert (field in history) is emitted
    if emitted:
        assert history[field] is None
    assert message.model_dump() == original
    assert field in driver.assistant_history(message)
