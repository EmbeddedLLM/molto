from pydantic import BaseModel

from molto.utils.io import dumps_sse, load_sse


class ExampleEvent(BaseModel):
    type: str
    detail: str | None = None
    sequence_number: int = 0


def test_load_sse_joins_data_lines_and_ignores_other_fields():
    sse = (
        "event: response.output_text.delta\n"
        "id: event-1\n"
        'data: {"type": "response.output_text.delta",\n'
        'data: "delta": "Hello"}\n\n'
    )

    assert load_sse(sse) == {
        "type": "response.output_text.delta",
        "delta": "Hello",
    }


def test_dump_sse_serializes_dict_with_event_type():
    assert dumps_sse({"type": "response.completed"}, "response.completed") == (
        'event: response.completed\ndata: {"type":"response.completed"}\n\n'
    )


def test_dump_sse_serializes_only_set_model_fields():
    event = ExampleEvent(type="response.created", detail=None)

    assert dumps_sse(event) == 'data: {"type":"response.created","detail":null}\n\n'


def test_sse_round_trip():
    event = {
        "type": "response.output_text.delta",
        "delta": "Hello, world!",
        "metadata": {"sequence_number": 1, "final": False},
    }

    assert load_sse(dumps_sse(event, event["type"])) == event
