import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from molto.utils.io import dump_yaml, dumps_sse, load_sse, read_json, read_yaml

# U+201D encodes to UTF-8 bytes E2 80 9D; 0x9D is undefined in cp1252, so this
# text crashes any read that falls back to the Windows default encoding.
NON_ASCII_TEXT = "\u201cquoted\u201d \u2014 it\u2019s"


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


def test_read_yaml_decodes_utf8_independently_of_platform_encoding(tmp_path: Path):
    """Ensures cassette reads use UTF-8, not the platform default (cp1252 on Windows)."""
    path = tmp_path / "cassette.yaml"
    path.write_bytes(f"text: {NON_ASCII_TEXT}\n".encode())

    assert read_yaml(str(path)) == {"text": NON_ASCII_TEXT}


def test_read_json_decodes_utf8_independently_of_platform_encoding(tmp_path: Path):
    """Ensures JSON reads use UTF-8, not the platform default (cp1252 on Windows)."""
    path = tmp_path / "data.json"
    path.write_bytes(f'{{"text": "{NON_ASCII_TEXT}"}}'.encode())

    assert read_json(str(path)) == {"text": NON_ASCII_TEXT}


def test_dump_yaml_writes_utf8_independently_of_platform_encoding(tmp_path: Path):
    """Ensures written cassettes are UTF-8 bytes, so matcher scripts do not re-encode them."""
    path = tmp_path / "cassette.yaml"

    dump_yaml({"text": NON_ASCII_TEXT}, str(path))

    # Strict UTF-8 decoding is the check; newlines are normalized by .gitattributes.
    assert path.read_text(encoding="utf-8") == f"text: {NON_ASCII_TEXT}\n"


def test_import_and_json_helpers_work_without_pyyaml(tmp_path: Path):
    """Ensures `import molto` and the JSON helpers do not require PyYAML, a test-only dependency."""
    path = tmp_path / "data.json"
    path.write_bytes(b'{"text": "ok"}')
    # A fresh interpreter, because this one has already imported yaml.
    code = "\n".join(
        [
            "import sys; sys.modules['yaml'] = None",
            "import molto",
            "from molto.utils.io import json_dumps, read_json",
            f"assert read_json({str(path)!r}) == {{'text': 'ok'}}",
            """assert json_dumps({'a': 1}) == '{"a":1}'""",
        ]
    )

    subprocess.run([sys.executable, "-c", code], check=True)


def test_yaml_round_trip(tmp_path: Path):
    """Ensures `dump_yaml` output reads back unchanged through `read_yaml`."""
    path = tmp_path / "cassette.yaml"
    data = {"request": {"body": {"text": NON_ASCII_TEXT}}, "items": [1, 2]}

    dump_yaml(data, str(path))

    assert read_yaml(str(path)) == data
