"""Anthropic provider implementation."""

import json
from typing import Optional, List, Dict, Any, AsyncIterator, Iterator

import httpx

from .base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    Message,
    ProviderError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ServerError,
)


class AnthropicProvider(BaseProvider):
    """Anthropic API provider."""

    provider_name = "anthropic"
    default_model = "claude-sonnet-4-20250514"
    available_models = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229",
        "claude-3-haiku-20240307",
    ]

    DEFAULT_BASE_URL = "https://api.anthropic.com"
    DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
    API_VERSION = "2023-06-01"

    def __init__(self, config: Optional[ProviderConfig] = None):
        super().__init__(config)
        if self.config.api_key_env is None:
            self.config.api_key_env = self.DEFAULT_API_KEY_ENV

    def _get_base_url(self) -> str:
        """Get API base URL."""
        return self.config.base_url or self.DEFAULT_BASE_URL

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        api_key = self.config.get_api_key()
        if not api_key:
            raise AuthenticationError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY or provide api_key in config.",
                provider=self.provider_name,
            )

        headers = {
            "x-api-key": api_key,
            "anthropic-version": self.API_VERSION,
            "Content-Type": "application/json",
        }
        headers.update(self.config.extra_headers)
        return headers

    def _create_client(self) -> httpx.Client:
        """Create synchronous HTTP client."""
        return httpx.Client(
            base_url=self._get_base_url(),
            headers=self._get_headers(),
            timeout=self.config.timeout,
        )

    def _create_async_client(self) -> httpx.AsyncClient:
        """Create asynchronous HTTP client."""
        return httpx.AsyncClient(
            base_url=self._get_base_url(),
            headers=self._get_headers(),
            timeout=self.config.timeout,
        )

    def _handle_error(self, response: httpx.Response) -> None:
        """Handle API error responses."""
        status_code = response.status_code

        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", response.text)
        except Exception:
            error_message = response.text

        if status_code == 401:
            raise AuthenticationError(
                f"Authentication failed: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
                response=response,
            )
        elif status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(
                f"Rate limit exceeded: {error_message}",
                provider=self.provider_name,
                retry_after=float(retry_after) if retry_after else None,
                status_code=status_code,
                response=response,
            )
        elif status_code == 400:
            raise InvalidRequestError(
                f"Invalid request: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
                response=response,
            )
        elif status_code >= 500:
            raise ServerError(
                f"Server error: {error_message}",
                provider=self.provider_name,
                status_code=status_code,
                response=response,
            )
        else:
            raise ProviderError(
                f"API error ({status_code}): {error_message}",
                provider=self.provider_name,
                status_code=status_code,
                response=response,
            )

    def _convert_messages(self, messages: List[Message]) -> tuple[Optional[str], List[Dict[str, str]]]:
        """
        Convert messages to Anthropic format.

        Anthropic uses a separate 'system' parameter instead of a system message.
        Returns (system_prompt, messages).
        """
        system_prompt = None
        converted_messages = []

        for msg in messages:
            if msg.role == "system":
                # Anthropic handles system prompt separately
                system_prompt = msg.content
            else:
                converted_messages.append({
                    "role": msg.role,
                    "content": msg.content,
                })

        return system_prompt, converted_messages

    def _build_request_body(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request body for messages API."""
        system_prompt, converted_messages = self._convert_messages(messages)

        body: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": converted_messages,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": stream,
        }

        if system_prompt:
            body["system"] = system_prompt

        # Add optional parameters
        temperature = kwargs.get("temperature", self.config.temperature)
        if temperature is not None:
            body["temperature"] = temperature

        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            body["top_k"] = kwargs["top_k"]
        if "stop_sequences" in kwargs:
            body["stop_sequences"] = kwargs["stop_sequences"]

        # Add any extra params from config
        body.update(self.config.extra_params)

        return body

    def _parse_response(self, data: Dict[str, Any], model: str) -> ProviderResponse:
        """Parse API response into ProviderResponse."""
        content_blocks = data.get("content", [])
        text_content = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text_content += block.get("text", "")

        usage = data.get("usage")
        if usage:
            usage = {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
            }

        return ProviderResponse(
            content=text_content,
            model=data.get("model", model),
            provider=self.provider_name,
            finish_reason=data.get("stop_reason"),
            usage=usage,
            raw_response=data,
        )

    def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion synchronously."""
        body = self._build_request_body(messages, model, stream=False, **kwargs)

        response = self.client.post("/v1/messages", json=body)

        if response.status_code != 200:
            self._handle_error(response)

        data = response.json()
        return self._parse_response(data, model or self.default_model)

    async def complete_async(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion asynchronously."""
        body = self._build_request_body(messages, model, stream=False, **kwargs)

        response = await self.async_client.post("/v1/messages", json=body)

        if response.status_code != 200:
            self._handle_error(response)

        data = response.json()
        return self._parse_response(data, model or self.default_model)

    def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream a completion synchronously."""
        body = self._build_request_body(messages, model, stream=True, **kwargs)

        with self.client.stream("POST", "/v1/messages", json=body) as response:
            if response.status_code != 200:
                response.read()
                self._handle_error(response)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if not data_str.strip():
                        continue

                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue

    async def stream_async(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion asynchronously."""
        body = self._build_request_body(messages, model, stream=True, **kwargs)

        async with self.async_client.stream("POST", "/v1/messages", json=body) as response:
            if response.status_code != 200:
                await response.aread()
                self._handle_error(response)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if not data_str.strip():
                        continue

                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type")

                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            if delta.get("type") == "text_delta":
                                text = delta.get("text", "")
                                if text:
                                    yield text
                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue
