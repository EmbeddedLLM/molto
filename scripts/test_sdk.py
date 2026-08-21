import asyncio
import base64
import json
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

PROMPT = "Call `read_image('a.png')` and `read_image('b.png')` in parallel."
QUESTION = "What are the logos?"
TOOL_CALL_COUNT = 2
READ_IMAGE_FUNCTION = {
    "name": "read_image",
    "description": "Read an image from a file path.",
    "parameters": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the image file.",
            }
        },
        "required": ["file_path"],
        "additionalProperties": False,
    },
    "strict": False,
}
RESPONSES_TOOLS = [{"type": "function", **READ_IMAGE_FUNCTION}]
CHAT_COMPLETIONS_TOOLS = [{"type": "function", "function": READ_IMAGE_FUNCTION}]
ANTHROPIC_TOOLS = [
    {
        "name": READ_IMAGE_FUNCTION["name"],
        "description": READ_IMAGE_FUNCTION["description"],
        "input_schema": READ_IMAGE_FUNCTION["parameters"],
    }
]

OPENAI_MODEL = "gpt-5.6-luna"
ANTHROPIC_MODEL = "claude-sonnet-5"


async def responses():
    input_items = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": PROMPT}],
        }
    ]
    for turn in range(2):
        response = await openai_client.responses.create(
            model=OPENAI_MODEL,
            input=input_items,
            tools=RESPONSES_TOOLS,
            # tool_choice=(
            #     {"type": "function", "name": READ_IMAGE_FUNCTION["name"]} if turn == 0 else "none"
            # ),
            parallel_tool_calls=True,
            reasoning=dict(effort="none"),
            stream=True,
            stream_options=dict(include_obfuscation=False),
            max_output_tokens=4096,
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
                print("", flush=True)
                print(event.response.usage)

        print("\n" + "---" * 6, flush=True)
        input_items.extend(item.model_dump(exclude_none=True) for item in completed_output)
        tool_calls = [item for item in completed_output if item.type == "function_call"]
        if turn == 1 and not tool_calls:
            return
        if len(tool_calls) != TOOL_CALL_COUNT or any(
            tool_call.name != READ_IMAGE_FUNCTION["name"] for tool_call in tool_calls
        ):
            raise RuntimeError(f"Model did not call read_image exactly {TOOL_CALL_COUNT} times")
        input_items.extend(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": [
                    {
                        "type": "input_image",
                        "image_url": IMAGE_URL,
                        "detail": "auto",
                    }
                ],
            }
            for tool_call in tool_calls
        )
        input_items.append(
            {
                "role": "user",
                "content": [{"type": "input_text", "text": QUESTION}],
            }
        )

    raise RuntimeError("Model did not answer after receiving the image")


async def chat_completions():
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    for turn in range(2):
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=CHAT_COMPLETIONS_TOOLS,
            # tool_choice=(
            #     {
            #         "type": "function",
            #         "function": {"name": READ_IMAGE_FUNCTION["name"]},
            #     }
            #     if turn == 0
            #     else "none"
            # ),
            reasoning_effort="none",
            parallel_tool_calls=True,
            stream=True,
            stream_options=dict(include_obfuscation=False, include_usage=True),
            max_completion_tokens=4096,
        )
        current_output = output_type = None
        reasoning_content = ""
        content = ""
        tool_calls = {}
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

            for tool_call_delta in delta.tool_calls or []:
                tool_call = tool_calls.setdefault(
                    tool_call_delta.index,
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    },
                )
                if tool_call_delta.id:
                    tool_call["id"] = tool_call_delta.id
                if tool_call_delta.function:
                    tool_call["function"]["name"] += tool_call_delta.function.name or ""
                    tool_call["function"]["arguments"] += tool_call_delta.function.arguments or ""

        print("\n" + "---" * 6, flush=True)
        ordered_tool_calls = [tool_calls[index] for index in sorted(tool_calls)]
        if turn == 1 and not ordered_tool_calls:
            return
        if len(ordered_tool_calls) != TOOL_CALL_COUNT or any(
            tool_call["function"]["name"] != READ_IMAGE_FUNCTION["name"]
            for tool_call in ordered_tool_calls
        ):
            raise RuntimeError(f"Model did not call read_image exactly {TOOL_CALL_COUNT} times")

        messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": ordered_tool_calls,
            }
        )
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": "The image will be attached as user prompt after all tool responses.",
            }
            for tool_call in ordered_tool_calls
        )
        # Chat tool messages cannot contain images. Keep every tool response
        # contiguous, then relocate image content after the complete tool block.
        for tool_call in ordered_tool_calls:
            try:
                file_path = json.loads(tool_call["function"]["arguments"])["file_path"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise RuntimeError("read_image did not provide a file_path") from error
            if not isinstance(file_path, str):
                raise RuntimeError("read_image file_path must be a string")

            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"Image returned by read_image(file_path={file_path!r}), "
                                f"tool_call_id={tool_call['id']!r}:"
                            ),
                        },
                        {"type": "image_url", "image_url": {"url": IMAGE_URL}},
                    ],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": QUESTION}],
            }
        )

    raise RuntimeError("Model did not answer after receiving the image")


async def messages():
    message_history = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    for turn in range(2):
        async with anthropic_client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            messages=message_history,
            tools=ANTHROPIC_TOOLS,
            # tool_choice=(
            #     {
            #         "type": "tool",
            #         "name": READ_IMAGE_FUNCTION["name"],
            #         "disable_parallel_tool_use": False,
            #     }
            #     if turn == 0
            #     else {"type": "none"}
            # ),
            # thinking={"type": "adaptive", "display": "summarized"},
            # output_config={"effort": "high"},
            thinking={"type": "disabled"},
        ) as response:
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
            message = await response.get_final_message()

        print("\n" + "---" * 6, flush=True)
        tool_calls = [block for block in message.content if block.type == "tool_use"]
        if turn == 1 and not tool_calls:
            return
        if len(tool_calls) != TOOL_CALL_COUNT or any(
            tool_call.name != READ_IMAGE_FUNCTION["name"] for tool_call in tool_calls
        ):
            raise RuntimeError(f"Model did not call read_image exactly {TOOL_CALL_COUNT} times")

        message_history.extend(
            [
                {
                    "role": "assistant",
                    "content": [block.model_dump(exclude_none=True) for block in message.content],
                },
                {
                    "role": "user",
                    "content": [
                        *[
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call.id,
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": "image/png",
                                            "data": IMAGE_DATA,
                                        },
                                    }
                                ],
                            }
                            for tool_call in tool_calls
                        ],
                        {"type": "text", "text": QUESTION},
                    ],
                },
            ]
        )

    raise RuntimeError("Model did not answer after receiving the image")


async def main():
    # await responses()
    # await chat_completions()
    await messages()


if __name__ == "__main__":
    asyncio.run(main())
