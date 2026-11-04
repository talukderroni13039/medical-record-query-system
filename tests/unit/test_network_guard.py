import os
import socket

import pytest


def test_outbound_network_is_blocked_in_default_tests() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="Network access blocked"):
            sock.connect(("api.anthropic.com", 443))
    finally:
        sock.close()


def test_default_tests_have_no_api_key_and_provider_none() -> None:
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert os.environ["ANSWER_PROVIDER"] == "none"
    assert os.environ["APP_ENV"] == "test"
