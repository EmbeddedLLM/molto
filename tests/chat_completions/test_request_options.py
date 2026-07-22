from molto.providers import OpenAIChatCompletionsProvider
from molto.types.openai.chat_completions import OpenAIChatCompletionRequest


def test_request_omits_implicit_defaults_from_ir() -> None:
    implicit = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
    ).to_ir()
    assert implicit.stream is None
    assert implicit.parallel_tool_calls is None
    assert implicit.tool_choice is None

    explicit = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        parallel_tool_calls=True,
        tool_choice="auto",
    ).to_ir()
    assert explicit.stream is False
    assert explicit.parallel_tool_calls is True
    assert explicit.tool_choice == "auto"


def test_shared_openai_options_are_not_chat_specific() -> None:
    request = OpenAIChatCompletionRequest(
        model="openai/gpt-4.1-nano",
        messages=[{"role": "user", "content": "hi"}],
        store=False,
        prompt_cache_key="chat-cache",
        prompt_cache_retention="24h",
        service_tier="flex",
        extra_body={"chat_only": True},
    )

    ir = request.to_ir()

    assert ir.openai.store is False
    assert ir.openai.prompt_cache_key == "chat-cache"
    assert ir.openai.prompt_cache_retention == "24h"
    assert ir.openai.service_tier == "flex"
    assert ir.extra_body == {"chat_only": True}
    assert "extra_body" in type(ir).model_fields
    assert "extra_body" not in type(ir.responses).model_fields
    assert ir.responses.include is None

    model_settings = OpenAIChatCompletionsProvider._build_model_settings(ir)
    assert model_settings["extra_body"] == {"chat_only": True}
