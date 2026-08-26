from itertools import product
from pathlib import Path
from typing import Literal

from molto.types.exceptions import ModelNotFoundError

CASSETTE_DIR = Path(__file__).parents[1] / "cassettes" / "errors"
Protocol = Literal["responses", "chat/completions", "messages"]
PROTOCOLS: tuple[Protocol, ...] = ("responses", "chat/completions", "messages")
ROUTES: tuple[tuple[Protocol, Protocol], ...] = tuple(product(PROTOCOLS, PROTOCOLS))


def test_model_not_found():
    # TODO: Load cassettes, for each protocol, load the downstream request body, and replay upstream response
    # Assert the exception cause is a ModelNotFoundError
    pass
