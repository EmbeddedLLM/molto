import pytest
from pydantic import ValidationError

from molto.types.normalized import NormalizedRequest, TextInput, UserInput


def test_user_input_preserves_message_boundary_around_text_parts():
    first = UserInput(content=[TextInput(content="Part one"), TextInput(content="Part two")])
    second = UserInput(content=[TextInput(content="Follow-up")])

    request = NormalizedRequest(input=[first, second])

    assert request.input == [first, second]


def test_text_input_is_not_a_top_level_input_item():
    with pytest.raises(ValidationError, match="union_tag_invalid"):
        NormalizedRequest(input=[TextInput(content="Unwrapped")])
