"""Main client wrapper with retry, fallback, and circuit breaker."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Iterator, AsyncIterator

from .providers.base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    Message,
    ProviderError,
)
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .retry import (
    RetryConfig,
    RetryState,
    RetryExhaustedError,
    RateLimitError as RetryRateLimitError,
    is_rate_limit_error,
    extract_retry_after,
)
from .circuit import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitOpenError,
    get_circuit_breaker,
)
from .fallback import (
    FallbackChain,
    FallbackChainConfig,
    FallbackResult,
    FallbackExhaustedError,
    ModelSpec,
    create_openai_chain,
    create_anthropic_chain,
    create_mixed_chain,
)


@dataclass
class ClientConfig:
    """Configuration for the retry client."""

    # Default provider settings
    default_provider: str = "openai"
    default_model: Optional[str] = None

    # Provider configs
    openai_config: Optional[ProviderConfig] = None
    anthropic_config: Optional[ProviderConfig] = None

    # Retry settings
    retry_enabled: bool = True
    retry_config: Optional[RetryConfig] = None

    # Circuit breaker settings
    circuit_enabled: bool = True
    circuit_config: Optional[CircuitBreakerConfig] = None

    # Fallback settings
    fallback_enabled: bool = True
    fallback_models: Optional[List[tuple[str, str]]] = None  # [(provider, model), ...]

    # Callbacks
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
    on_fallback: Optional[Callable[[str, str, Exception], None]] = None


class RetryClient:
    """
    Robust LLM client with retry, fallback, and circuit breaker patterns.

    This client wraps multiple LLM providers and provides:
    - Automatic retries with exponential backoff and jitter
    - Fallback to alternative models/providers on failure
    - Circuit breaker to prevent cascading failures
    - Rate limit detection and handling
    """

    def __init__(self, config: Optional[ClientConfig] = None):
        self.config = config or ClientConfig()

        # Initialize providers
        self._providers: Dict[str, BaseProvider] = {}
        self._init_providers()

        # Initialize circuit breakers
        self._circuits: Dict[str, CircuitBreaker] = {}

        # Initialize fallback chain if enabled
        self._fallback_chain: Optional[FallbackChain] = None
        if self.config.fallback_enabled:
            self._init_fallback_chain()

    def _init_providers(self) -> None:
        """Initialize provider instances."""
        # OpenAI
        openai_config = self.config.openai_config or ProviderConfig()
        self._providers["openai"] = OpenAIProvider(openai_config)

        # Anthropic
        anthropic_config = self.config.anthropic_config or ProviderConfig()
        self._providers["anthropic"] = AnthropicProvider(anthropic_config)

    def _init_fallback_chain(self) -> None:
        """Initialize fallback chain."""
        if self.config.fallback_models:
            # Use custom fallback chain
            specs = [
                ModelSpec(
                    provider=provider,
                    model=model,
                    config=self._get_provider_config(provider),
                    retry_config=self.config.retry_config,
                    circuit_config=self.config.circuit_config,
                )
                for provider, model in self.config.fallback_models
            ]
            chain_config = FallbackChainConfig(
                models=specs,
                on_fallback=self.config.on_fallback,
            )
            self._fallback_chain = FallbackChain(chain_config)
        else:
            # Use default mixed chain
            self._fallback_chain = create_mixed_chain(
                openai_config=self.config.openai_config,
                anthropic_config=self.config.anthropic_config,
            )

    def _get_provider_config(self, provider: str) -> Optional[ProviderConfig]:
        """Get provider config by name."""
        if provider == "openai":
            return self.config.openai_config
        elif provider == "anthropic":
            return self.config.anthropic_config
        return None

    def _get_provider(self, provider: Optional[str] = None) -> BaseProvider:
        """Get provider instance."""
        provider_name = provider or self.config.default_provider
        if provider_name not in self._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        return self._providers[provider_name]

    def _get_circuit(self, provider: str, model: str) -> CircuitBreaker:
        """Get or create circuit breaker for provider/model."""
        circuit_name = f"{provider}:{model}"
        if circuit_name not in self._circuits:
            config = self.config.circuit_config or CircuitBreakerConfig()
            self._circuits[circuit_name] = CircuitBreaker(circuit_name, config)
        return self._circuits[circuit_name]

    def _execute_with_retry(
        self,
        func: Callable[[], ProviderResponse],
        provider: str,
        model: str,
    ) -> ProviderResponse:
        """Execute a function with retry logic."""
        if not self.config.retry_enabled:
            return func()

        retry_config = self.config.retry_config or RetryConfig()
        state = RetryState(retry_config)

        while True:
            try:
                return func()
            except Exception as e:
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(e)
                    if retry_after and retry_after > retry_config.max_delay:
                        raise RetryRateLimitError(str(e), retry_after) from e

                if not state.should_retry(e):
                    raise RetryExhaustedError(
                        f"Retry exhausted after {state.attempt} attempts",
                        last_exception=e,
                        attempts=state.attempt,
                    ) from e

                state.increment(e)
                delay = state.get_delay()

                # Use retry-after if available
                retry_after = extract_retry_after(e)
                if retry_after and retry_after <= retry_config.max_delay:
                    delay = retry_after

                if self.config.on_retry:
                    self.config.on_retry(state.attempt, e, delay)

                import time
                time.sleep(delay)

    async def _execute_with_retry_async(
        self,
        func: Callable[[], Any],
        provider: str,
        model: str,
    ) -> ProviderResponse:
        """Execute an async function with retry logic."""
        if not self.config.retry_enabled:
            return await func()

        retry_config = self.config.retry_config or RetryConfig()
        state = RetryState(retry_config)

        while True:
            try:
                return await func()
            except Exception as e:
                if is_rate_limit_error(e):
                    retry_after = extract_retry_after(e)
                    if retry_after and retry_after > retry_config.max_delay:
                        raise RetryRateLimitError(str(e), retry_after) from e

                if not state.should_retry(e):
                    raise RetryExhaustedError(
                        f"Retry exhausted after {state.attempt} attempts",
                        last_exception=e,
                        attempts=state.attempt,
                    ) from e

                state.increment(e)
                delay = state.get_delay()

                # Use retry-after if available
                retry_after = extract_retry_after(e)
                if retry_after and retry_after <= retry_config.max_delay:
                    delay = retry_after

                if self.config.on_retry:
                    self.config.on_retry(state.attempt, e, delay)

                await asyncio.sleep(delay)

    def complete(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        use_fallback: Optional[bool] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion with retry and fallback.

        Args:
            messages: List of chat messages
            model: Model to use
            provider: Provider to use (defaults to config default)
            use_fallback: Override fallback setting
            **kwargs: Additional parameters for the provider

        Returns:
            ProviderResponse with the completion
        """
        use_fallback = use_fallback if use_fallback is not None else self.config.fallback_enabled

        if use_fallback and self._fallback_chain:
            result = self._fallback_chain.complete(messages, **kwargs)
            return result.response

        # Single provider mode
        provider_name = provider or self.config.default_provider
        provider_instance = self._get_provider(provider_name)
        model_name = model or self.config.default_model or provider_instance.default_model

        if self.config.circuit_enabled:
            circuit = self._get_circuit(provider_name, model_name)
            if circuit.is_open:
                raise CircuitOpenError(
                    f"Circuit open for {provider_name}:{model_name}",
                    remaining_timeout=circuit.get_stats()["remaining_timeout"],
                )

            def do_complete() -> ProviderResponse:
                return self._execute_with_retry(
                    lambda: provider_instance.complete(messages, model=model_name, **kwargs),
                    provider_name,
                    model_name,
                )

            return circuit.call(do_complete)
        else:
            return self._execute_with_retry(
                lambda: provider_instance.complete(messages, model=model_name, **kwargs),
                provider_name,
                model_name,
            )

    async def complete_async(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        use_fallback: Optional[bool] = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """
        Generate a completion asynchronously with retry and fallback.

        Args:
            messages: List of chat messages
            model: Model to use
            provider: Provider to use (defaults to config default)
            use_fallback: Override fallback setting
            **kwargs: Additional parameters for the provider

        Returns:
            ProviderResponse with the completion
        """
        use_fallback = use_fallback if use_fallback is not None else self.config.fallback_enabled

        if use_fallback and self._fallback_chain:
            result = await self._fallback_chain.complete_async(messages, **kwargs)
            return result.response

        # Single provider mode
        provider_name = provider or self.config.default_provider
        provider_instance = self._get_provider(provider_name)
        model_name = model or self.config.default_model or provider_instance.default_model

        if self.config.circuit_enabled:
            circuit = self._get_circuit(provider_name, model_name)
            if circuit.is_open:
                raise CircuitOpenError(
                    f"Circuit open for {provider_name}:{model_name}",
                    remaining_timeout=circuit.get_stats()["remaining_timeout"],
                )

            async def do_complete() -> ProviderResponse:
                return await self._execute_with_retry_async(
                    lambda: provider_instance.complete_async(messages, model=model_name, **kwargs),
                    provider_name,
                    model_name,
                )

            return await circuit.call_async(do_complete)
        else:
            return await self._execute_with_retry_async(
                lambda: provider_instance.complete_async(messages, model=model_name, **kwargs),
                provider_name,
                model_name,
            )

    def chat(
        self,
        message: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Simple chat interface - send a message and get a response.

        Args:
            message: User message
            system: Optional system prompt
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters

        Returns:
            Response text
        """
        messages = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(message))

        response = self.complete(messages, model=model, provider=provider, **kwargs)
        return response.content

    async def chat_async(
        self,
        message: str,
        system: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Async simple chat interface.

        Args:
            message: User message
            system: Optional system prompt
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters

        Returns:
            Response text
        """
        messages = []
        if system:
            messages.append(Message.system(system))
        messages.append(Message.user(message))

        response = await self.complete_async(messages, model=model, provider=provider, **kwargs)
        return response.content

    def stream(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """
        Stream a completion (no retry/fallback for streaming).

        Args:
            messages: List of chat messages
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters

        Yields:
            Chunks of the response
        """
        provider_name = provider or self.config.default_provider
        provider_instance = self._get_provider(provider_name)
        model_name = model or self.config.default_model or provider_instance.default_model

        yield from provider_instance.stream(messages, model=model_name, **kwargs)

    async def stream_async(
        self,
        messages: List[Message],
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Stream a completion asynchronously (no retry/fallback for streaming).

        Args:
            messages: List of chat messages
            model: Model to use
            provider: Provider to use
            **kwargs: Additional parameters

        Yields:
            Chunks of the response
        """
        provider_name = provider or self.config.default_provider
        provider_instance = self._get_provider(provider_name)
        model_name = model or self.config.default_model or provider_instance.default_model

        async for chunk in provider_instance.stream_async(messages, model=model_name, **kwargs):
            yield chunk

    def health_check(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Check health of provider(s).

        Args:
            provider: Specific provider to check (or all if None)

        Returns:
            Health status dictionary
        """
        if provider:
            return self._get_provider(provider).health_check()

        results = {}
        for name, prov in self._providers.items():
            try:
                results[name] = prov.health_check()
            except Exception as e:
                results[name] = {
                    "provider": name,
                    "status": "error",
                    "error": str(e),
                }
        return results

    async def health_check_async(self, provider: Optional[str] = None) -> Dict[str, Any]:
        """
        Check health of provider(s) asynchronously.

        Args:
            provider: Specific provider to check (or all if None)

        Returns:
            Health status dictionary
        """
        if provider:
            return await self._get_provider(provider).health_check_async()

        results = {}
        for name, prov in self._providers.items():
            try:
                results[name] = await prov.health_check_async()
            except Exception as e:
                results[name] = {
                    "provider": name,
                    "status": "error",
                    "error": str(e),
                }
        return results

    def get_circuit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        stats = {name: circuit.get_stats() for name, circuit in self._circuits.items()}
        if self._fallback_chain:
            stats.update(self._fallback_chain.get_circuit_stats())
        return stats

    def reset_circuits(self) -> None:
        """Reset all circuit breakers."""
        for circuit in self._circuits.values():
            circuit.reset()
        if self._fallback_chain:
            self._fallback_chain.reset_circuits()

    def close(self) -> None:
        """Close all resources."""
        for provider in self._providers.values():
            provider.close()
        if self._fallback_chain:
            self._fallback_chain.close()

    async def close_async(self) -> None:
        """Close all resources asynchronously."""
        for provider in self._providers.values():
            await provider.close_async()
        if self._fallback_chain:
            await self._fallback_chain.close_async()

    def __enter__(self) -> "RetryClient":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "RetryClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()


# Convenience function for quick usage
def create_client(
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    retry_enabled: bool = True,
    fallback_enabled: bool = True,
) -> RetryClient:
    """
    Create a retry client with simple configuration.

    Args:
        provider: Default provider ("openai" or "anthropic")
        model: Default model
        api_key: API key (uses environment variable if not provided)
        retry_enabled: Enable retry logic
        fallback_enabled: Enable fallback chain

    Returns:
        Configured RetryClient
    """
    provider_config = ProviderConfig(api_key=api_key) if api_key else None

    config = ClientConfig(
        default_provider=provider,
        default_model=model,
        retry_enabled=retry_enabled,
        fallback_enabled=fallback_enabled,
    )

    if provider == "openai":
        config.openai_config = provider_config
    elif provider == "anthropic":
        config.anthropic_config = provider_config

    return RetryClient(config)
