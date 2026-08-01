from molto.types.ir.portable import ResponseMetadata, ResponseStatus


def test_openai_response_metadata_is_strict_and_round_trips() -> None:
    metadata = ResponseMetadata(model="gpt-test", object="response", choice_index=0)

    assert ResponseMetadata.model_validate_json(metadata.model_dump_json()) == metadata


def test_response_status_remains_openai_compatible() -> None:
    assert ResponseStatus.COMPLETED.value == "completed"
