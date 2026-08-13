import inspect

from molto.upstream.base import BaseClient
from molto.upstream.openai import OpenAIChatCompletionsClient, OpenAIResponsesClient


def test_base_client_defines_abstract_streaming_contract():
    assert inspect.isabstract(BaseClient)
    assert BaseClient.__abstractmethods__ == {"events", "native_events"}
    assert not inspect.isabstract(OpenAIResponsesClient)
    assert not inspect.isabstract(OpenAIChatCompletionsClient)
