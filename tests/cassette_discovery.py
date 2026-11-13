"""Filename discovery shared by public-route replay and matcher tests."""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from molto.providers import DEFAULT_PROVIDER_BY_ENDPOINT, PROVIDER_CAPABILITIES, Endpoint, Provider

CASSETTE_DIR = Path(__file__).parent / "cassettes"
Mode = Literal["stream", "non_stream"]
PROTOCOLS: tuple[Endpoint, ...] = ("responses", "chat/completions", "messages")
ENCRYPTED_SCENARIOS = (
    "reasoning_encrypted",
    "input_image_with_reasoning_encrypted",
    "tool_call_with_reasoning_encrypted",
)
SCENARIO_PROTOCOLS = dict.fromkeys(
    (
        "simple_text",
        "sampling_tests",
        "reasoning_raw",
        "input_image_base64",
        "input_image_remote_url",
        "tool_call",
        "tool_call_choice",
        "tool_call_parallel",
        "tool_call_image_result_parallel",
        "prompt_cache_implicit",
        "prompt_cache_explicit",
        "error-invalid_request_error",
        "web_search",
    ),
    PROTOCOLS,
)
SCENARIO_PROTOCOLS.update(dict.fromkeys(ENCRYPTED_SCENARIOS, ("responses", "messages")))
# Provider features without a translation contract. Their recordings may only
# prove native passthrough, so they are rejected anywhere outside `native/`.
NATIVE_SCENARIO_PROTOCOLS: dict[str, tuple[Endpoint, ...]] = {
    "response_format_json_object": ("chat/completions",),
    "response_format_json_schema": ("chat/completions",),
    "text_format_json_schema": ("responses",),
    "model_settings": ("chat/completions", "responses"),
    "tool_choice_required": ("chat/completions", "responses"),
    "web_search_tool_options": ("responses",),
    "web_search_api_source": ("responses",),
    "conversation_id": ("responses",),
}
_FILENAME = re.compile(
    r"(?P<provider>[a-z0-9_]+)-(?P<endpoint>responses|chat_completions|messages)-"
    r"(?P<mode>stream|non_stream)-(?P<turn>[0-9]+)\.yaml"
)


@dataclass(frozen=True)
class Cassette:
    path: Path
    scenario: str
    provider: Provider
    endpoint: Endpoint
    mode: Mode
    turn: int


def parse_cassette(path: Path) -> Cassette:
    match = _FILENAME.fullmatch(path.name)
    if match is None:
        raise ValueError(
            f"Invalid cassette filename {path}; expected provider-endpoint-mode-turn.yaml"
        )
    provider = match["provider"]
    if provider not in PROVIDER_CAPABILITIES:
        raise ValueError(
            f"Unsupported provider profile {provider!r} in {path}; update PROVIDER_CAPABILITIES"
        )
    return Cassette(
        path,
        path.parent.name,
        cast(Provider, provider),
        cast(Endpoint, match["endpoint"].replace("chat_completions", "chat/completions")),
        cast(Mode, match["mode"]),
        int(match["turn"]),
    )


@dataclass(frozen=True)
class Conversation:
    scenario: str
    provider: Provider
    endpoint: Endpoint
    mode: Mode
    cassettes: tuple[Cassette, ...]

    @property
    def paths(self) -> list[Path]:
        return [cassette.path for cassette in self.cassettes]

    @property
    def turns(self) -> tuple[int, ...]:
        return tuple(cassette.turn for cassette in self.cassettes)

    @property
    def id(self) -> str:
        return f"{self.scenario}-{self.provider}-{self.endpoint}-{self.mode}"


def discover_conversations(
    root: Path, *, scenarios: Iterable[str] | None = None
) -> tuple[Conversation, ...]:
    """Inspect only scenario directories directly inside the requested root."""
    selected = None if scenarios is None else set(scenarios)
    grouped: dict[tuple[str, Provider, Endpoint, Mode], dict[int, Cassette]] = defaultdict(dict)
    for path in sorted(root.glob("*/*.yaml")):
        if selected is not None and path.parent.name not in selected:
            continue
        cassette = parse_cassette(path)
        key = (cassette.scenario, cassette.provider, cassette.endpoint, cassette.mode)
        turns = grouped[key]
        if cassette.turn in turns:
            raise ValueError(
                f"Duplicate cassette identity: {turns[cassette.turn].path} and {path}"
            )
        turns[cassette.turn] = cassette
    conversations = []
    for key, turns in sorted(grouped.items()):
        if sorted(turns) != list(range(max(turns) + 1)):
            raise ValueError(
                f"Missing turns for {key}: found {sorted(turns)}; require consecutive turns from 0"
            )
        conversations.append(Conversation(*key, tuple(turns[turn] for turn in sorted(turns))))
    return tuple(conversations)


def select_conversation(
    conversations: Iterable[Conversation],
    scenario: str,
    endpoint: Endpoint,
    mode: Mode,
    *,
    provider: Provider | None = None,
) -> Conversation:
    candidates = [
        conversation
        for conversation in conversations
        if (conversation.scenario, conversation.endpoint, conversation.mode)
        == (scenario, endpoint, mode)
    ]
    preferred = provider or DEFAULT_PROVIDER_BY_ENDPOINT[endpoint]
    for conversation in candidates:
        if conversation.provider == preferred:
            return conversation
    if provider is None and len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Missing or ambiguous {scenario}/{endpoint}/{mode} counterpart for provider {preferred!r}; "
        f"found {[conversation.id for conversation in candidates]}"
    )


@dataclass(frozen=True)
class RouteCase:
    downstream: Conversation
    upstream: Conversation
    reference: Conversation

    @property
    def id(self) -> str:
        return f"{self.downstream.id}-via-{self.upstream.provider}-{self.upstream.endpoint}"


def _matched_issuer(conversation: Conversation) -> Provider:
    # Only the curated Messages fixtures use the anthropic label for
    # OpenAI-issued opaque state. This is not a cross-provider contract.
    if conversation.endpoint == "messages" and conversation.provider == "anthropic":
        return "openai"
    return conversation.provider


def route_cases(
    conversations: Iterable[Conversation],
    scenario: str,
    mode: Mode,
    routes: Iterable[tuple[Endpoint, Endpoint]],
) -> tuple[RouteCase, ...]:
    """Vary upstream providers against deterministic companions, never provider products."""
    selected = [c for c in conversations if c.scenario == scenario and c.mode == mode]
    cases = []
    for downstream, upstream in routes:
        upstreams = [c for c in selected if c.endpoint == upstream]
        if not upstreams:
            raise ValueError(f"Missing {scenario}/{upstream}/{mode} conversation")
        for source in upstreams:
            target = (
                source
                if downstream == upstream
                else select_conversation(selected, scenario, downstream, mode)
            )
            reference = select_conversation(selected, scenario, "responses", mode)
            for counterpart in (target, reference):
                if source.turns != counterpart.turns:
                    raise ValueError(
                        f"Mismatched turns: {source.id} {source.turns} versus {counterpart.id} {counterpart.turns}"
                    )
                if scenario in ENCRYPTED_SCENARIOS:
                    if _matched_issuer(source) != _matched_issuer(counterpart):
                        raise ValueError(
                            f"Incompatible encrypted-state issuers: {source.id} and {counterpart.id}"
                        )
            cases.append(RouteCase(target, source, reference))
    return tuple(cases)


def public_conversations(root: Path) -> tuple[Conversation, ...]:
    conversations = discover_conversations(root)
    native = root.name == "native"
    scenario_protocols = SCENARIO_PROTOCOLS | (NATIVE_SCENARIO_PROTOCOLS if native else {})
    for conversation in conversations:
        scenario = conversation.scenario
        if scenario not in scenario_protocols:
            raise ValueError(
                f"Unsupported scenario {scenario!r} in {root}; add scenario assertions before replay"
            )
        if native and scenario == "error-invalid_request_error":
            raise ValueError(f"Error scenario in {root}; use matched error assertions instead")
        if conversation.endpoint not in scenario_protocols[scenario]:
            raise ValueError(f"Unsupported endpoint in {conversation.id}")
        thinking = PROVIDER_CAPABILITIES[conversation.provider].thinking
        if (scenario == "reasoning_raw" and thinking != "raw") or (
            scenario in ENCRYPTED_SCENARIOS and thinking != "summary"
        ):
            raise ValueError(
                f"Incompatible provider reasoning profile in {conversation.id}: {thinking}"
            )
        if (
            root.name == "matched"
            and conversation.mode == "non_stream"
            and scenario
            not in {"simple_text", "sampling_tests", "reasoning_encrypted", "web_search"}
        ):
            raise ValueError(
                f"Unsupported matched response mode in {conversation.id}; add scenario assertions"
            )
        if root.name == "matched":
            minimum_turns = {
                "tool_call": 3,
                "tool_call_with_reasoning_encrypted": 3,
                "error-invalid_request_error": 1,
                "web_search": 1,
            }.get(scenario, 2)
            if len(conversation.cassettes) < minimum_turns:
                raise ValueError(
                    f"Missing required turns in {conversation.id}: expected at least {minimum_turns}, found {conversation.turns}"
                )
    return conversations


MATCHED_CONVERSATIONS = public_conversations(CASSETTE_DIR / "matched")
NATIVE_CONVERSATIONS = public_conversations(CASSETTE_DIR / "native")


def matcher_pairs(
    scenario: str, endpoint: Endpoint, mode: Mode = "stream"
) -> tuple[tuple[int, Path, Path], ...]:
    return tuple(
        (source.turn, source.path, reference.path)
        for case in route_cases(MATCHED_CONVERSATIONS, scenario, mode, [("responses", endpoint)])
        for source, reference in zip(
            case.upstream.cassettes, case.reference.cassettes, strict=True
        )
    )
