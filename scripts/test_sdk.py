import asyncio
import base64
from pathlib import Path

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from molto.configs import ENV_CONFIG

IMAGE_PATH = Path("tests") / "files" / "github.png"
IMAGE_DATA = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
IMAGE_URL = f"data:image/png;base64,{IMAGE_DATA}"

openai_client = AsyncOpenAI(
    api_key=ENV_CONFIG.openai_api_key_plain,
    base_url="http://localhost:8080/v1",
    # api_key=ENV_CONFIG.deepseek_api_key_plain,
    # base_url="https://api.deepseek.com",
)
anthropic_client = AsyncAnthropic(
    api_key=ENV_CONFIG.anthropic_api_key_plain,
    base_url="http://localhost:8080",
    # base_url="https://api.anthropic.com",
)

PROMPTS = [
    "How much wood would a woodchuck chuck?",
    "What's your reasoning?",
]


OPENAI_MODEL = "gpt-5.6-luna"
ANTHROPIC_MODEL = "claude-sonnet-5"


async def responses(turns: int = 1):
    if turns not in (1, 2):
        raise ValueError("turns must be 1 or 2")

    input_items = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": PROMPTS[0]},
                {"type": "input_image", "image_url": IMAGE_URL},
            ],
        }
    ]
    for current_turn in range(turns):
        response = await openai_client.responses.create(
            model=OPENAI_MODEL,
            input=input_items,
            # include=["reasoning.encrypted_content"],
            stream=True,
            stream_options=dict(include_obfuscation=False),
            # reasoning=dict(effort="max", summary="detailed"),
            reasoning=dict(effort="none"),
            max_output_tokens=2048,
        )
        current_output = output_type = None
        completed_output = []
        async for event in response:
            if event.type == "response.reasoning_text.delta":
                output_type = event.type
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[94m{event.delta}\033[0m", end="", flush=True)
            elif event.type == "response.reasoning_summary_text.delta":
                output_type = event.type
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[90m{event.delta}\033[0m", end="", flush=True)
            elif event.type == "response.output_text.delta":
                output_type = event.type
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[97m{event.delta}\033[0m", end="", flush=True)
            elif event.type == "response.completed":
                completed_output = event.response.output

        print("\n" + "---" * 6, flush=True)
        if current_turn < turns - 1:
            input_items.extend(item.model_dump(exclude_none=True) for item in completed_output)
            input_items.append(
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": PROMPTS[current_turn + 1]}],
                }
            )


async def chat_completions(turns: int = 1):
    if turns not in (1, 2):
        raise ValueError("turns must be 1 or 2")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPTS[0]},
                {"type": "image_url", "image_url": {"url": IMAGE_URL}},
            ],
        }
    ]
    for current_turn in range(turns):
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            stream=True,
            stream_options=dict(include_obfuscation=False, include_usage=True),
            reasoning_effort="none",
            max_completion_tokens=2048,
        )
        current_output = output_type = None
        reasoning_content = ""
        content = ""
        async for event in response:
            if not event.choices:
                continue

            delta = event.choices[0].delta
            if reasoning_delta := getattr(delta, "reasoning_content", None):
                output_type = "reasoning"
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[94m{reasoning_delta}\033[0m", end="", flush=True)
                reasoning_content += reasoning_delta
            elif delta.content:
                output_type = "text"
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[97m{delta.content}\033[0m", end="", flush=True)
                content += delta.content

        print("\n" + "---" * 6, flush=True)
        if current_turn < turns - 1:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": content,
                        **({"reasoning_content": reasoning_content} if reasoning_content else {}),
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": PROMPTS[current_turn + 1]}],
                    },
                ]
            )


async def messages(turns: int = 1):
    if turns not in (1, 2):
        raise ValueError("turns must be 1 or 2")

    message_history = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPTS[0]},
            ],
        }
    ]
    for current_turn in range(turns):
        response = await anthropic_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2048,
            messages=message_history,
            # thinking={"type": "adaptive", "display": "summarized"},
            # output_config={"effort": "max"},
            thinking={"type": "disabled"},
            stream=True,
        )
        current_output = None
        thinking = ""
        signature = ""
        text = ""
        async for event in response:
            # print(event)
            if event.type != "content_block_delta":
                continue
            delta = event.delta
            if delta.type == "thinking_delta":
                output_type = "thinking"
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[94m{delta.thinking}\033[0m", end="", flush=True)
                thinking += delta.thinking
            elif delta.type == "signature_delta":
                signature += delta.signature
            elif delta.type == "text_delta":
                output_type = "text"
                if current_output is not None and current_output != output_type:
                    print("", flush=True)
                current_output = output_type
                print(f"\033[97m{delta.text}\033[0m", end="", flush=True)
                text += delta.text

        print("\n" + "---" * 6, flush=True)
        print(f"{len(thinking)=}, {len(signature)=}")
        if current_turn < turns - 1:
            message_history.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            [{"type": "thinking", "thinking": thinking, "signature": signature}]
                            if signature
                            else []
                        )
                        + [{"type": "text", "text": text}],
                    },
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": PROMPTS[current_turn + 1]}],
                    },
                ]
            )


async def main():
    # await responses(2)
    # await chat_completions(2)
    await messages(2)


if __name__ == "__main__":
    asyncio.run(main())
