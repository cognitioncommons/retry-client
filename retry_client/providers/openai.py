"""OpenAI provider implementation."""

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


class OpenAIProvider(BaseProvider):
    """OpenAI API provider."""

    provider_name = "openai"
    default_model = "gpt-4o"
    available_models = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-4",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o1-preview",
    ]

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"

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
                "OpenAI API key not found. Set OPENAI_API_KEY or provide api_key in config.",
                provider=self.provider_name,
            )

        headers = {
            "Authorization": f"Bearer {api_key}",
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

    def _build_request_body(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build request body for chat completion."""
        body = {
            "model": model or self.default_model,
            "messages": [m.to_dict() for m in messages],
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "stream": stream,
        }

        # Add optional parameters
        if "top_p" in kwargs:
            body["top_p"] = kwargs["top_p"]
        if "frequency_penalty" in kwargs:
            body["frequency_penalty"] = kwargs["frequency_penalty"]
        if "presence_penalty" in kwargs:
            body["presence_penalty"] = kwargs["presence_penalty"]
        if "stop" in kwargs:
            body["stop"] = kwargs["stop"]

        # Add any extra params from config
        body.update(self.config.extra_params)

        return body

    def _parse_response(self, data: Dict[str, Any], model: str) -> ProviderResponse:
        """Parse API response into ProviderResponse."""
        choice = data["choices"][0]
        message = choice.get("message", {})

        usage = data.get("usage")
        if usage:
            usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        return ProviderResponse(
            content=message.get("content", ""),
            model=data.get("model", model),
            provider=self.provider_name,
            finish_reason=choice.get("finish_reason"),
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

        response = self.client.post("/chat/completions", json=body)

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

        response = await self.async_client.post("/chat/completions", json=body)

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

        with self.client.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code != 200:
                # Read the full response for error handling
                response.read()
                self._handle_error(response)

            for line in response.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
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

        async with self.async_client.stream("POST", "/chat/completions", json=body) as response:
            if response.status_code != 200:
                await response.aread()
                self._handle_error(response)

            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        choices = data.get("choices", [])
                        if choices:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        continue
