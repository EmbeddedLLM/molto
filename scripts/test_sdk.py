import asyncio

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from molto.configs import ENV_CONFIG

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

PROMPT = "Call Bash with command `sleep 5`, then call DateTime and respond with its result. Describe your action before calling each tool."
TOOL_RESULTS = {
    "Bash": """<returncode>

0

</returncode>

<stdout>


</stdout>

<stderr>


</stderr>""",
    "DateTime": """Current UTC datetime (ISO 8601 format): 2026-08-03T04:25:45.279896+00:00

Current UTC day name: Monday""",
}
FUNCTIONS = [
    {
        "name": "Bash",
        "description": "Execute bash commands.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute.",
                    "maxLength": 200,
                    "minLength": 1,
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": False,
    },
    {
        "name": "DateTime",
        "description": "Get the current date and time in UTC (ISO 8601 format).",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        "strict": False,
    },
]
RESPONSES_TOOLS = [{"type": "function", **function} for function in FUNCTIONS]
CHAT_COMPLETIONS_TOOLS = [{"type": "function", "function": function} for function in FUNCTIONS]
ANTHROPIC_TOOLS = [
    {
        "name": function["name"],
        "description": function["description"],
        "input_schema": function["parameters"],
    }
    for function in FUNCTIONS
]

OPENAI_MODEL = "gpt-5.6-luna"
ANTHROPIC_MODEL = "claude-haiku-4-5"


async def responses():
    input_items = [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": PROMPT}],
        }
    ]
    for _ in range(len(FUNCTIONS) + 1):
        response = await openai_client.responses.create(
            model=OPENAI_MODEL,
            input=input_items,
            tools=RESPONSES_TOOLS,
            parallel_tool_calls=False,
            stream=True,
            stream_options=dict(include_obfuscation=False),
            reasoning=dict(effort="none"),
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

        print("\n" + "---" * 6, flush=True)
        input_items.extend(item.model_dump(exclude_none=True) for item in completed_output)
        tool_calls = [item for item in completed_output if item.type == "function_call"]
        if not tool_calls:
            return
        input_items.extend(
            {
                "type": "function_call_output",
                "call_id": tool_call.call_id,
                "output": TOOL_RESULTS[tool_call.name],
            }
            for tool_call in tool_calls
        )

    raise RuntimeError("Model did not finish after the expected tool calls")


async def chat_completions():
    messages = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    for _ in range(len(FUNCTIONS) + 1):
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=CHAT_COMPLETIONS_TOOLS,
            parallel_tool_calls=False,
            stream=True,
            stream_options=dict(include_obfuscation=False, include_usage=True),
            reasoning_effort="none",
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
        if not tool_calls:
            return

        messages.append(
            {
                "role": "assistant",
                "content": content or None,
                "tool_calls": list(tool_calls.values()),
                **({"reasoning_content": reasoning_content} if reasoning_content else {}),
            }
        )
        messages.extend(
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": TOOL_RESULTS[tool_call["function"]["name"]],
            }
            for tool_call in tool_calls.values()
        )

    raise RuntimeError("Model did not finish after the expected tool calls")


async def messages():
    message_history = [{"role": "user", "content": [{"type": "text", "text": PROMPT}]}]
    for _ in range(len(FUNCTIONS) + 1):
        async with anthropic_client.messages.stream(
            model=ANTHROPIC_MODEL,
            max_tokens=4096,
            messages=message_history,
            tools=ANTHROPIC_TOOLS,
            thinking={"type": "disabled"},
        ) as response:
            current_output = None
            async for event in response:
                if event.type != "content_block_delta":
                    continue
                delta = event.delta
                if delta.type == "thinking_delta":
                    output_type = "thinking"
                    if current_output is not None and current_output != output_type:
                        print("", flush=True)
                    current_output = output_type
                    print(f"\033[94m{delta.thinking}\033[0m", end="", flush=True)
                elif delta.type == "text_delta":
                    output_type = "text"
                    if current_output is not None and current_output != output_type:
                        print("", flush=True)
                    current_output = output_type
                    print(f"\033[97m{delta.text}\033[0m", end="", flush=True)
            message = await response.get_final_message()

        print("\n" + "---" * 6, flush=True)
        tool_calls = [block for block in message.content if block.type == "tool_use"]
        if not tool_calls:
            return

        message_history.extend(
            [
                {
                    "role": "assistant",
                    "content": [block.model_dump(exclude_none=True) for block in message.content],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call.id,
                            "content": TOOL_RESULTS[tool_call.name],
                        }
                        for tool_call in tool_calls
                    ],
                },
            ]
        )

    raise RuntimeError("Model did not finish after the expected tool calls")


async def main():
    # await responses()
    # await chat_completions()
    await messages()


if __name__ == "__main__":
    asyncio.run(main())
