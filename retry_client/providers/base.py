"""Base provider class for LLM providers."""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator, Iterator


@dataclass
class Message:
    """A chat message."""

    role: str  # "system", "user", "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary."""
        return {"role": self.role, "content": self.content}

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role="assistant", content=content)


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    api_key: Optional[str] = None
    api_key_env: Optional[str] = None  # Environment variable name for API key
    base_url: Optional[str] = None
    timeout: float = 60.0
    max_tokens: int = 4096
    temperature: float = 0.7
    extra_headers: Dict[str, str] = field(default_factory=dict)
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get_api_key(self) -> Optional[str]:
        """Get API key from config or environment."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class ProviderResponse:
    """Response from a provider."""

    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": x, "completion_tokens": y, "total_tokens": z}
    raw_response: Optional[Any] = None

    @property
    def prompt_tokens(self) -> int:
        """Get prompt token count."""
        if self.usage:
            return self.usage.get("prompt_tokens", 0)
        return 0

    @property
    def completion_tokens(self) -> int:
        """Get completion token count."""
        if self.usage:
            return self.usage.get("completion_tokens", 0)
        return 0

    @property
    def total_tokens(self) -> int:
        """Get total token count."""
        if self.usage:
            return self.usage.get("total_tokens", 0)
        return self.prompt_tokens + self.completion_tokens


class ProviderError(Exception):
    """Base exception for provider errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code
        self.response = response


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""

    pass


class RateLimitError(ProviderError):
    """Raised when rate limit is hit."""

    def __init__(
        self,
        message: str,
        provider: str,
        retry_after: Optional[float] = None,
        status_code: Optional[int] = None,
        response: Optional[Any] = None,
    ):
        super().__init__(message, provider, status_code, response)
        self.retry_after = retry_after


class InvalidRequestError(ProviderError):
    """Raised when request is invalid."""

    pass


class ServerError(ProviderError):
    """Raised when server returns an error."""

    pass


class BaseProvider(ABC):
    """Base class for LLM providers."""

    provider_name: str = "base"
    default_model: str = ""
    available_models: List[str] = []

    def __init__(self, config: Optional[ProviderConfig] = None):
        self.config = config or ProviderConfig()
        self._client: Optional[Any] = None
        self._async_client: Optional[Any] = None

    @abstractmethod
    def _create_client(self) -> Any:
        """Create the synchronous HTTP client."""
        pass

    @abstractmethod
    def _create_async_client(self) -> Any:
        """Create the asynchronous HTTP client."""
        pass

    @property
    def client(self) -> Any:
        """Get or create synchronous client."""
        if self._client is None:
            self._client = self._create_client()
        return self._client

    @property
    def async_client(self) -> Any:
        """Get or create asynchronous client."""
        if self._async_client is None:
            self._async_client = self._create_async_client()
        return self._async_client

    @abstractmethod
    def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion synchronously.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to provider's default)
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse with the completion
        """
        pass

    @abstractmethod
    async def complete_async(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion asynchronously.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to provider's default)
            **kwargs: Additional provider-specific parameters

        Returns:
            ProviderResponse with the completion
        """
        pass

    @abstractmethod
    def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream a completion synchronously.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to provider's default)
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the completion text
        """
        pass

    @abstractmethod
    async def stream_async(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion asynchronously.

        Args:
            messages: List of chat messages
            model: Model to use (defaults to provider's default)
            **kwargs: Additional provider-specific parameters

        Yields:
            Chunks of the completion text
        """
        pass

    def health_check(self) -> Dict[str, Any]:
        """
        Check provider health synchronously.

        Returns:
            Dict with health status information
        """
        try:
            # Simple completion to test connectivity
            response = self.complete(
                messages=[Message.user("Say 'ok'")],
                max_tokens=10,
            )
            return {
                "provider": self.provider_name,
                "status": "healthy",
                "model": response.model,
            }
        except Exception as e:
            return {
                "provider": self.provider_name,
                "status": "unhealthy",
                "error": str(e),
            }

    async def health_check_async(self) -> Dict[str, Any]:
        """
        Check provider health asynchronously.

        Returns:
            Dict with health status information
        """
        try:
            response = await self.complete_async(
                messages=[Message.user("Say 'ok'")],
                max_tokens=10,
            )
            return {
                "provider": self.provider_name,
                "status": "healthy",
                "model": response.model,
            }
        except Exception as e:
            return {
                "provider": self.provider_name,
                "status": "unhealthy",
                "error": str(e),
            }

    def close(self) -> None:
        """Close synchronous client."""
        if self._client is not None:
            if hasattr(self._client, "close"):
                self._client.close()
            self._client = None

    async def close_async(self) -> None:
        """Close asynchronous client."""
        if self._async_client is not None:
            if hasattr(self._async_client, "aclose"):
                await self._async_client.aclose()
            elif hasattr(self._async_client, "close"):
                self._async_client.close()
            self._async_client = None

    def __enter__(self) -> "BaseProvider":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "BaseProvider":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()
