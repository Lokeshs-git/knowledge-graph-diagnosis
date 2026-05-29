"""Tests for the LLM client.

Uses unittest.mock to avoid real API calls. Real integration tests
should live in `tests/integration/` and be opt-in (e.g. marked `@pytest.mark.integration`).
"""

from unittest.mock import MagicMock, patch

from quickstart.llm import LLMClient, _extract_text


def test_extract_text_concatenates_text_blocks() -> None:
    """Multiple text blocks are joined with newlines."""
    msg = MagicMock()
    block1 = MagicMock()
    block1.type = "text"
    block1.text = "Hello"
    block2 = MagicMock()
    block2.type = "text"
    block2.text = "world"
    msg.content = [block1, block2]

    assert _extract_text(msg) == "Hello\nworld"


def test_extract_text_skips_non_text_blocks() -> None:
    """Tool use blocks etc. are ignored."""
    msg = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "answer"
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    msg.content = [text_block, tool_block]

    assert _extract_text(msg) == "answer"


@patch("quickstart.llm.Anthropic")
def test_complete_returns_text(mock_anthropic_class: MagicMock) -> None:
    """LLMClient.complete returns the joined text from the API response."""
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.type = "text"
    mock_block.text = "Paris"
    mock_response.content = [mock_block]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    client = LLMClient(api_key="test-key")
    result = client.complete("What is the capital of France?")

    assert result == "Paris"
    mock_client.messages.create.assert_called_once()
    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["messages"] == [
        {"role": "user", "content": "What is the capital of France?"}
    ]


@patch("quickstart.llm.Anthropic")
def test_complete_uses_system_prompt(mock_anthropic_class: MagicMock) -> None:
    """System prompt is forwarded when set on the client."""
    mock_response = MagicMock()
    mock_response.content = []
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response
    mock_anthropic_class.return_value = mock_client

    client = LLMClient(api_key="test-key", system="Be terse.")
    client.complete("hi")

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Be terse."
