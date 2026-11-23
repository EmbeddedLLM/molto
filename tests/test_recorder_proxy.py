import httpx
import pytest

from scripts import recorder_proxy


@pytest.mark.parametrize(
    ("value", "masked"),
    [
        ("Bearer sk-secret", "Bearer ***"),
        ("bearer sk-secret", "Bearer ***"),
        ("Basic dXNlcjpzZWNyZXQ=", "***"),
        ("sk-secret", "***"),
        ("", ""),
    ],
)
def test_mask_authorization_never_retains_credentials(value: str, masked: str):
    """Ensures recorded Authorization values keep at most the scheme, never key material."""
    assert recorder_proxy.mask_authorization(value) == masked


def test_filter_headers_retains_only_whitelisted_headers_and_masks_authorization():
    """Ensures recordings drop unlisted headers, such as API keys, and mask Authorization."""
    headers = httpx.Headers(
        {
            "Authorization": "Bearer sk-secret",
            "Content-Type": "application/json",
            "X-Run-Id": "run-1",
            "X-API-Key": "sk-ant-secret",
            "OpenAI-Organization": "org-secret",
            "Cookie": "session=secret",
        }
    )

    assert recorder_proxy.filter_headers(headers) == {
        "authorization": "Bearer ***",
        "content-type": "application/json",
        "x-run-id": "run-1",
    }
