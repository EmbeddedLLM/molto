from itertools import product
from pathlib import Path

import pytest

from tests.cassette_discovery import (
    PROTOCOLS,
    discover_conversations,
    parse_cassette,
    public_conversations,
    route_cases,
    select_conversation,
)


def _recordings(root: Path, scenario: str, prefix: str, turns: tuple[int, ...] = (0, 1)) -> None:
    directory = root / scenario
    directory.mkdir(parents=True, exist_ok=True)
    for turn in turns:
        (directory / f"{prefix}-{turn}.yaml").touch()


def _matched_text(root: Path) -> None:
    for prefix in (
        "openai-responses-stream",
        "openai-chat_completions-stream",
        "anthropic-messages-stream",
    ):
        _recordings(root, "simple_text", prefix)


@pytest.mark.parametrize("provider", ["vllm", "anthropic"])
def test_second_provider_adds_cases_without_replacing_existing_conversation(
    tmp_path: Path, provider: str
) -> None:
    """Ensures a second provider on one endpoint adds distinct native and route cases."""
    root = tmp_path / "matched"
    _matched_text(root)
    routes = tuple(product(PROTOCOLS, repeat=2))
    original = route_cases(public_conversations(root), "simple_text", "stream", routes)

    _recordings(root, "simple_text", f"{provider}-chat_completions-stream")
    conversations = public_conversations(root)
    expanded = route_cases(conversations, "simple_text", "stream", routes)

    assert len(conversations) == 4
    assert len({conversation.id for conversation in conversations}) == 4
    assert {case.id for case in original} < {case.id for case in expanded}
    assert len(expanded) == len({case.id for case in expanded}) == 12
    extra = [
        case
        for case in expanded
        if case.upstream.provider == provider and case.upstream.endpoint == "chat/completions"
    ]
    assert {case.downstream.endpoint for case in extra} == set(PROTOCOLS)
    assert all(case.reference.provider == "openai" for case in extra)
    native = next(case for case in extra if case.downstream.endpoint == "chat/completions")
    assert native.downstream is native.upstream


def test_extra_responses_provider_preserves_the_canonical_reference(tmp_path: Path) -> None:
    """Ensures adding another Responses provider cannot replace existing parity references."""
    root = tmp_path / "matched"
    _matched_text(root)
    routes = tuple(product(PROTOCOLS, repeat=2))
    original = route_cases(public_conversations(root), "simple_text", "stream", routes)

    _recordings(root, "simple_text", "anthropic-responses-stream")
    expanded = route_cases(public_conversations(root), "simple_text", "stream", routes)

    assert {case.id for case in original} < {case.id for case in expanded}
    assert len(expanded) == len({case.id for case in expanded}) == 12
    assert all(case.reference.provider == "openai" for case in expanded)
    assert {case.reference.paths[0] for case in expanded} == {original[0].reference.paths[0]}


def test_explicit_provider_selection_does_not_fall_back_to_endpoint_default(
    tmp_path: Path,
) -> None:
    """Ensures provider-qualified selection retains both providers' turn zero."""
    root = tmp_path / "matched"
    _matched_text(root)
    _recordings(root, "simple_text", "vllm-chat_completions-stream")
    conversations = public_conversations(root)

    selected = select_conversation(
        conversations, "simple_text", "chat/completions", "stream", provider="vllm"
    )
    canonical = select_conversation(conversations, "simple_text", "chat/completions", "stream")

    assert selected.provider == "vllm"
    assert canonical.provider == "openai"
    assert selected.paths[0] != canonical.paths[0]
    with pytest.raises(ValueError, match="counterpart.*anthropic"):
        select_conversation(
            conversations, "simple_text", "chat/completions", "stream", provider="anthropic"
        )


def test_turns_are_ordered_numerically_and_modes_stay_separate(tmp_path: Path) -> None:
    """Ensures turn ten follows turn nine and response modes form separate conversations."""
    root = tmp_path / "native"
    _recordings(root, "simple_text", "vllm-chat_completions-stream", tuple(range(11)))
    _recordings(root, "simple_text", "vllm-chat_completions-non_stream")

    conversations = public_conversations(root)
    assert len(conversations) == 2
    streamed = select_conversation(conversations, "simple_text", "chat/completions", "stream")
    complete = select_conversation(conversations, "simple_text", "chat/completions", "non_stream")
    assert streamed.turns == tuple(range(11))
    assert [path.stem.rsplit("-", 1)[1] for path in streamed.paths] == [
        str(turn) for turn in range(11)
    ]
    assert complete.turns == (0, 1)
    assert streamed.id != complete.id


def test_discovery_only_reads_the_requested_root_and_direct_scenarios(tmp_path: Path) -> None:
    """Ensures sources, upstream IR, archives, and nested trees cannot become route cases."""
    root = tmp_path / "matched"
    _matched_text(root)
    for excluded in (
        tmp_path / "sources" / "simple_text",
        tmp_path / "upstream_ir" / "simple_text",
        tmp_path / "archive" / "simple_text",
        root / "archive" / "simple_text",
    ):
        excluded.mkdir(parents=True, exist_ok=True)
        (excluded / "malformed.yaml").touch()

    conversations = public_conversations(root)
    assert len(conversations) == 3
    assert all(path.parent.parent == root for c in conversations for path in c.paths)


def test_native_conversation_needs_no_matched_counterpart(tmp_path: Path) -> None:
    """Ensures a faithful provider recording can join native replay without other protocols."""
    root = tmp_path / "native"
    _recordings(root, "tool_call", "vllm-chat_completions-stream", (0, 1, 2))

    conversations = public_conversations(root)
    assert len(conversations) == 1
    assert conversations[0].provider == "vllm"
    assert conversations[0].endpoint == "chat/completions"
    assert conversations[0].turns == (0, 1, 2)


def test_single_turn_search_conversations_support_both_response_modes(tmp_path: Path) -> None:
    """Ensures hosted-search fixtures join discovery without requiring a second turn."""
    root = tmp_path / "matched"
    for mode in ("stream", "non_stream"):
        for prefix in ("openai-responses", "openai-chat_completions", "anthropic-messages"):
            _recordings(root, "web_search", f"{prefix}-{mode}", (0,))

    conversations = public_conversations(root)

    assert len(conversations) == 6
    assert {(c.endpoint, c.mode) for c in conversations} == set(
        product(PROTOCOLS, ("stream", "non_stream"))
    )
    assert all(c.scenario == "web_search" and c.turns == (0,) for c in conversations)


@pytest.mark.parametrize(
    "filename",
    [
        "openai-responses-stream.yaml",
        "openai-responses-stream-negative.yaml",
        "openai-responses-non-stream-00.yaml",
        "openai-chat-stream-00.yaml",
        "responses-stream-00.yaml",
    ],
)
def test_malformed_filenames_fail_with_the_expected_format(tmp_path: Path, filename: str) -> None:
    """Ensures malformed eligible files raise an actionable error instead of disappearing."""
    directory = tmp_path / "simple_text"
    directory.mkdir()
    (directory / filename).touch()

    with pytest.raises(ValueError, match="Invalid cassette filename.*provider-endpoint-mode-turn"):
        discover_conversations(tmp_path)


def test_duplicate_numeric_turn_identity_is_rejected(tmp_path: Path) -> None:
    """Ensures zero padding cannot disguise two files for one provider and turn."""
    _recordings(tmp_path, "simple_text", "openai-responses-stream")
    duplicate = tmp_path / "simple_text" / "openai-responses-stream-00.yaml"
    duplicate.touch()

    with pytest.raises(ValueError, match="Duplicate cassette identity") as error:
        discover_conversations(tmp_path)
    assert "openai-responses-stream-0.yaml" in str(error.value)
    assert duplicate.name in str(error.value)


@pytest.mark.parametrize("turns", [(1,), (0, 2)])
def test_missing_initial_or_intermediate_turn_is_rejected(
    tmp_path: Path, turns: tuple[int, ...]
) -> None:
    """Ensures conversations cannot omit an initial request or an intervening turn."""
    _recordings(tmp_path, "simple_text", "openai-responses-stream", turns)

    with pytest.raises(ValueError, match="Missing turns.*consecutive turns from 0"):
        discover_conversations(tmp_path)


def test_missing_matched_protocol_counterpart_is_rejected(tmp_path: Path) -> None:
    """Ensures missing matched protocols fail case generation rather than reducing routes."""
    _recordings(tmp_path, "simple_text", "openai-responses-stream")
    conversations = discover_conversations(tmp_path)

    with pytest.raises(ValueError, match="Missing.*simple_text/messages/stream"):
        route_cases(conversations, "simple_text", "stream", (("responses", "messages"),))


def test_missing_matched_counterpart_turn_is_rejected(tmp_path: Path) -> None:
    """Ensures equal scenario names cannot hide a truncated counterpart conversation."""
    _recordings(tmp_path, "simple_text", "openai-responses-stream")
    _recordings(tmp_path, "simple_text", "openai-chat_completions-stream", (0,))

    with pytest.raises(ValueError, match="Mismatched turns"):
        route_cases(
            discover_conversations(tmp_path),
            "simple_text",
            "stream",
            (("responses", "chat/completions"),),
        )


def test_missing_responses_reference_is_rejected(tmp_path: Path) -> None:
    """Ensures matched comparisons always have an independent Responses reference."""
    _recordings(tmp_path, "simple_text", "openai-chat_completions-stream")
    _recordings(tmp_path, "simple_text", "anthropic-messages-stream")

    with pytest.raises(ValueError, match="Missing or ambiguous simple_text/responses/stream"):
        route_cases(
            discover_conversations(tmp_path),
            "simple_text",
            "stream",
            (("messages", "chat/completions"),),
        )


def test_unknown_provider_requires_a_registered_capability_profile(tmp_path: Path) -> None:
    """Ensures a new filename cannot silently use the endpoint's default provider profile."""
    with pytest.raises(
        ValueError, match="Unsupported provider profile.*unregistered.*PROVIDER_CAPABILITIES"
    ):
        parse_cassette(tmp_path / "simple_text" / "unregistered-responses-stream-00.yaml")


def test_unsupported_scenario_requires_explicit_assertions(tmp_path: Path) -> None:
    """Ensures filename discovery does not invent coverage for an unknown scenario."""
    _recordings(tmp_path, "unsupported_feature", "openai-responses-stream")

    with pytest.raises(ValueError, match="Unsupported scenario.*add scenario assertions"):
        public_conversations(tmp_path)


@pytest.mark.parametrize(
    ("scenario", "provider"),
    [("reasoning_raw", "openai"), ("reasoning_encrypted", "vllm")],
)
def test_reasoning_profile_must_match_the_scenario(
    tmp_path: Path, scenario: str, provider: str
) -> None:
    """Ensures discovered reasoning fixtures use the scenario's raw or summary profile."""
    _recordings(tmp_path, scenario, f"{provider}-responses-stream")

    with pytest.raises(ValueError, match="Incompatible provider reasoning profile"):
        public_conversations(tmp_path)


@pytest.mark.parametrize(
    "scenario",
    [
        "reasoning_encrypted",
        "input_image_with_reasoning_encrypted",
        "tool_call_with_reasoning_encrypted",
    ],
)
def test_encrypted_messages_alias_preserves_the_single_issuer_contract(
    tmp_path: Path, scenario: str
) -> None:
    """Ensures curated anthropic-labelled Messages fixtures retain OpenAI-issued history."""
    root = tmp_path / "matched"
    turns = (0, 1, 2) if scenario == "tool_call_with_reasoning_encrypted" else (0, 1)
    _recordings(root, scenario, "openai-responses-stream", turns)
    _recordings(root, scenario, "anthropic-messages-stream", turns)

    cases = route_cases(
        public_conversations(root),
        scenario,
        "stream",
        tuple(product(("responses", "messages"), repeat=2)),
    )

    assert len(cases) == 4
    assert all(case.reference.provider == "openai" for case in cases)
    assert {case.upstream.provider for case in cases} == {"openai", "anthropic"}


def test_encrypted_history_rejects_other_cross_provider_issuer_pairs(tmp_path: Path) -> None:
    """Ensures matching summary profiles cannot authorize opaque state from another issuer."""
    root = tmp_path / "matched"
    _recordings(root, "reasoning_encrypted", "anthropic-responses-stream")
    _recordings(root, "reasoning_encrypted", "openai-messages-stream")

    with pytest.raises(ValueError, match="Incompatible encrypted-state issuers"):
        route_cases(
            public_conversations(root),
            "reasoning_encrypted",
            "stream",
            (("responses", "messages"),),
        )


def test_scenario_endpoint_restrictions_are_not_inferred_from_filename(tmp_path: Path) -> None:
    """Ensures correctly named Chat fixtures cannot bypass encrypted-scenario restrictions."""
    root = tmp_path / "matched"
    _recordings(root, "reasoning_encrypted", "openai-chat_completions-stream")

    with pytest.raises(ValueError, match="Unsupported endpoint.*reasoning_encrypted"):
        public_conversations(root)


@pytest.mark.parametrize(
    ("scenario", "turns"),
    [
        ("simple_text", (0,)),
        ("tool_call", (0, 1)),
        ("tool_call_with_reasoning_encrypted", (0, 1)),
    ],
)
def test_matched_conversation_preserves_required_scenario_turns(
    tmp_path: Path, scenario: str, turns: tuple[int, ...]
) -> None:
    """Ensures equally truncated protocol files cannot erase required continuation coverage."""
    root = tmp_path / "matched"
    _recordings(root, scenario, "openai-responses-stream", turns)

    with pytest.raises(ValueError, match="Missing required turns"):
        public_conversations(root)


def test_native_deviation_can_reproduce_in_a_single_turn(tmp_path: Path) -> None:
    """Ensures native SDK evidence does not require unrelated continuation recordings."""
    root = tmp_path / "native"
    _recordings(root, "simple_text", "vllm-chat_completions-stream", (0,))

    conversations = public_conversations(root)
    assert len(conversations) == 1
    assert conversations[0].turns == (0,)


def test_unsupported_matched_response_mode_requires_scenario_assertions(tmp_path: Path) -> None:
    """Ensures an unhandled complete-response scenario does not join unrelated tests."""
    root = tmp_path / "matched"
    _recordings(root, "tool_call", "openai-responses-non_stream", (0, 1, 2))

    with pytest.raises(
        ValueError, match="Unsupported matched response mode.*add scenario assertions"
    ):
        public_conversations(root)


def test_incomplete_results_are_excluded_from_registered_exception_cases(tmp_path: Path) -> None:
    """Ensures Responses-only incomplete results require no Chat or Messages error fixtures."""
    root = tmp_path / "errors"
    for prefix in (
        "openai-responses-stream",
        "openai-chat_completions-stream",
        "anthropic-messages-stream",
    ):
        _recordings(root, "model_not_found", prefix, (0,))
    _recordings(root, "incomplete_response", "openai-responses-stream", (0,))
    _recordings(root, "incomplete_response", "openai-responses-non_stream", (0,))
    _recordings(root, "unregistered_error", "unregistered-responses-stream", (0,))

    conversations = discover_conversations(root, scenarios=("model_not_found",))
    cases = route_cases(
        conversations,
        "model_not_found",
        "stream",
        tuple(product(PROTOCOLS, repeat=2)),
    )

    assert len(cases) == 9
    assert {conversation.scenario for conversation in conversations} == {"model_not_found"}
