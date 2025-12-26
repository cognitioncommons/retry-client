"""
retry-client - Robust LLM client with retry, fallback, and circuit breaker.

This package provides a resilient wrapper for LLM API calls with:
- Exponential backoff with jitter for retries
- Fallback model chains across providers
- Circuit breaker pattern to prevent cascading failures
- Rate limit detection and handling

Basic usage:
    from retry_client import RetryClient, Message

    client = RetryClient()
    response = client.chat("Hello, how are you?")
    print(response)

With configuration:
    from retry_client import RetryClient, ClientConfig, RetryConfig

    config = ClientConfig(
        default_provider="anthropic",
        retry_config=RetryConfig(max_retries=5),
        fallback_enabled=True,
    )
    client = RetryClient(config)

Using decorators:
    from retry_client import with_retry, circuit_breaker, RetryConfig

    @with_retry(RetryConfig(max_retries=3))
    def my_function():
        ...

    @circuit_breaker("my_service")
    def my_service_call():
        ...
"""

__version__ = "0.1.0"

# Main client
from .client import (
    RetryClient,
    ClientConfig,
    create_client,
)

# Retry module
from .retry import (
    RetryConfig,
    RetryState,
    RetryExhaustedError,
    RateLimitError,
    with_retry,
    with_async_retry,
    retry_sync,
    retry_async,
    is_rate_limit_error,
    extract_retry_after,
)

# Circuit breaker module
from .circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    CircuitOpenError,
    CircuitBreakerRegistry,
    circuit_breaker,
    async_circuit_breaker,
    get_circuit_breaker,
    get_registry,
)

# Fallback module
from .fallback import (
    FallbackChain,
    FallbackChainConfig,
    FallbackResult,
    FallbackReason,
    FallbackExhaustedError,
    ModelSpec,
    create_openai_chain,
    create_anthropic_chain,
    create_mixed_chain,
    register_provider,
)

# Provider base classes
from .providers.base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    Message,
    ProviderError,
    AuthenticationError,
    InvalidRequestError,
    ServerError,
)
from .providers.base import RateLimitError as ProviderRateLimitError

# Providers
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider

__all__ = [
    # Version
    "__version__",
    # Main client
    "RetryClient",
    "ClientConfig",
    "create_client",
    # Retry
    "RetryConfig",
    "RetryState",
    "RetryExhaustedError",
    "RateLimitError",
    "with_retry",
    "with_async_retry",
    "retry_sync",
    "retry_async",
    "is_rate_limit_error",
    "extract_retry_after",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreakerRegistry",
    "circuit_breaker",
    "async_circuit_breaker",
    "get_circuit_breaker",
    "get_registry",
    # Fallback
    "FallbackChain",
    "FallbackChainConfig",
    "FallbackResult",
    "FallbackReason",
    "FallbackExhaustedError",
    "ModelSpec",
    "create_openai_chain",
    "create_anthropic_chain",
    "create_mixed_chain",
    "register_provider",
    # Provider base
    "BaseProvider",
    "ProviderConfig",
    "ProviderResponse",
    "Message",
    "ProviderError",
    "AuthenticationError",
    "InvalidRequestError",
    "ServerError",
    "ProviderRateLimitError",
    # Providers
    "OpenAIProvider",
    "AnthropicProvider",
]
