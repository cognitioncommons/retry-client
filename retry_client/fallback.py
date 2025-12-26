"""Fallback model chains for resilient LLM requests."""

import asyncio
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Type, Union
from enum import Enum

from .providers.base import (
    BaseProvider,
    ProviderConfig,
    ProviderResponse,
    Message,
    ProviderError,
    RateLimitError,
)
from .providers.openai import OpenAIProvider
from .providers.anthropic import AnthropicProvider
from .retry import RetryConfig, RetryState, RetryExhaustedError
from .circuit import CircuitBreaker, CircuitBreakerConfig, CircuitOpenError


class FallbackReason(Enum):
    """Reason for falling back to next provider."""

    NONE = "none"  # No fallback occurred
    CIRCUIT_OPEN = "circuit_open"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    RETRY_EXHAUSTED = "retry_exhausted"


@dataclass
class FallbackResult:
    """Result from a fallback chain execution."""

    response: ProviderResponse
    attempts: List[Dict[str, Any]]  # List of attempt info
    fallback_reason: FallbackReason
    final_provider: str
    final_model: str


@dataclass
class ModelSpec:
    """Specification for a model in a fallback chain."""

    provider: str  # "openai" or "anthropic"
    model: str
    config: Optional[ProviderConfig] = None
    retry_config: Optional[RetryConfig] = None
    circuit_config: Optional[CircuitBreakerConfig] = None

    @property
    def circuit_name(self) -> str:
        """Generate circuit breaker name for this model."""
        return f"{self.provider}:{self.model}"


@dataclass
class FallbackChainConfig:
    """Configuration for a fallback chain."""

    models: List[ModelSpec] = field(default_factory=list)
    stop_on_rate_limit: bool = False  # If True, don't fallback on rate limit
    on_fallback: Optional[Callable[[str, str, Exception], None]] = None  # (from_model, to_model, error)


# Provider registry
PROVIDER_CLASSES: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """Register a custom provider."""
    PROVIDER_CLASSES[name] = provider_class


def create_provider(spec: ModelSpec) -> BaseProvider:
    """Create a provider instance from a ModelSpec."""
    provider_class = PROVIDER_CLASSES.get(spec.provider)
    if not provider_class:
        raise ValueError(f"Unknown provider: {spec.provider}")

    return provider_class(config=spec.config)


class FallbackChain:
    """
    Fallback chain for resilient LLM requests.

    Tries each model in sequence until one succeeds or all fail.
    Integrates with retry and circuit breaker patterns.
    """

    def __init__(self, config: FallbackChainConfig):
        self.config = config
        self._providers: Dict[str, BaseProvider] = {}
        self._circuits: Dict[str, CircuitBreaker] = {}

    def _get_provider(self, spec: ModelSpec) -> BaseProvider:
        """Get or create a provider for the given spec."""
        key = f"{spec.provider}:{id(spec.config)}"
        if key not in self._providers:
            self._providers[key] = create_provider(spec)
        return self._providers[key]

    def _get_circuit(self, spec: ModelSpec) -> CircuitBreaker:
        """Get or create a circuit breaker for the given spec."""
        circuit_name = spec.circuit_name
        if circuit_name not in self._circuits:
            config = spec.circuit_config or CircuitBreakerConfig()
            self._circuits[circuit_name] = CircuitBreaker(circuit_name, config)
        return self._circuits[circuit_name]

    def _should_fallback(self, error: Exception, spec: ModelSpec) -> bool:
        """Determine if we should fallback based on the error."""
        if isinstance(error, CircuitOpenError):
            return True

        if isinstance(error, RateLimitError):
            return not self.config.stop_on_rate_limit

        if isinstance(error, RetryExhaustedError):
            return True

        if isinstance(error, ProviderError):
            return True

        return True  # Default to fallback

    def complete(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> FallbackResult:
        """
        Execute completion with fallback chain synchronously.

        Args:
            messages: List of chat messages
            **kwargs: Additional parameters passed to provider

        Returns:
            FallbackResult with response and attempt info
        """
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None

        for i, spec in enumerate(self.config.models):
            provider = self._get_provider(spec)
            circuit = self._get_circuit(spec)

            attempt_info: Dict[str, Any] = {
                "provider": spec.provider,
                "model": spec.model,
                "index": i,
            }

            try:
                # Check circuit first
                if circuit.is_open:
                    raise CircuitOpenError(
                        f"Circuit open for {spec.circuit_name}",
                        remaining_timeout=circuit.get_stats()["remaining_timeout"],
                    )

                # Execute with circuit breaker
                def do_complete() -> ProviderResponse:
                    return provider.complete(messages, model=spec.model, **kwargs)

                response = circuit.call(do_complete)

                attempt_info["success"] = True
                attempts.append(attempt_info)

                return FallbackResult(
                    response=response,
                    attempts=attempts,
                    fallback_reason=FallbackReason.NONE if i == 0 else self._get_fallback_reason(last_error),
                    final_provider=spec.provider,
                    final_model=spec.model,
                )

            except Exception as e:
                last_error = e
                attempt_info["success"] = False
                attempt_info["error"] = str(e)
                attempt_info["error_type"] = type(e).__name__
                attempts.append(attempt_info)

                # Check if we should fallback
                if not self._should_fallback(e, spec):
                    raise

                # Call fallback callback
                if self.config.on_fallback and i < len(self.config.models) - 1:
                    next_spec = self.config.models[i + 1]
                    self.config.on_fallback(
                        f"{spec.provider}:{spec.model}",
                        f"{next_spec.provider}:{next_spec.model}",
                        e,
                    )

        # All models failed
        raise FallbackExhaustedError(
            f"All {len(self.config.models)} models in fallback chain failed",
            attempts=attempts,
            last_error=last_error,
        )

    async def complete_async(
        self,
        messages: List[Message],
        **kwargs: Any,
    ) -> FallbackResult:
        """
        Execute completion with fallback chain asynchronously.

        Args:
            messages: List of chat messages
            **kwargs: Additional parameters passed to provider

        Returns:
            FallbackResult with response and attempt info
        """
        attempts: List[Dict[str, Any]] = []
        last_error: Optional[Exception] = None

        for i, spec in enumerate(self.config.models):
            provider = self._get_provider(spec)
            circuit = self._get_circuit(spec)

            attempt_info: Dict[str, Any] = {
                "provider": spec.provider,
                "model": spec.model,
                "index": i,
            }

            try:
                # Check circuit first
                if circuit.is_open:
                    raise CircuitOpenError(
                        f"Circuit open for {spec.circuit_name}",
                        remaining_timeout=circuit.get_stats()["remaining_timeout"],
                    )

                # Execute with circuit breaker
                async def do_complete() -> ProviderResponse:
                    return await provider.complete_async(messages, model=spec.model, **kwargs)

                response = await circuit.call_async(do_complete)

                attempt_info["success"] = True
                attempts.append(attempt_info)

                return FallbackResult(
                    response=response,
                    attempts=attempts,
                    fallback_reason=FallbackReason.NONE if i == 0 else self._get_fallback_reason(last_error),
                    final_provider=spec.provider,
                    final_model=spec.model,
                )

            except Exception as e:
                last_error = e
                attempt_info["success"] = False
                attempt_info["error"] = str(e)
                attempt_info["error_type"] = type(e).__name__
                attempts.append(attempt_info)

                # Check if we should fallback
                if not self._should_fallback(e, spec):
                    raise

                # Call fallback callback
                if self.config.on_fallback and i < len(self.config.models) - 1:
                    next_spec = self.config.models[i + 1]
                    self.config.on_fallback(
                        f"{spec.provider}:{spec.model}",
                        f"{next_spec.provider}:{next_spec.model}",
                        e,
                    )

        # All models failed
        raise FallbackExhaustedError(
            f"All {len(self.config.models)} models in fallback chain failed",
            attempts=attempts,
            last_error=last_error,
        )

    def _get_fallback_reason(self, error: Optional[Exception]) -> FallbackReason:
        """Determine fallback reason from error."""
        if error is None:
            return FallbackReason.NONE
        if isinstance(error, CircuitOpenError):
            return FallbackReason.CIRCUIT_OPEN
        if isinstance(error, RateLimitError):
            return FallbackReason.RATE_LIMITED
        if isinstance(error, RetryExhaustedError):
            return FallbackReason.RETRY_EXHAUSTED
        return FallbackReason.ERROR

    def get_circuit_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers in the chain."""
        return {name: circuit.get_stats() for name, circuit in self._circuits.items()}

    def reset_circuits(self) -> None:
        """Reset all circuit breakers."""
        for circuit in self._circuits.values():
            circuit.reset()

    def close(self) -> None:
        """Close all providers."""
        for provider in self._providers.values():
            provider.close()
        self._providers.clear()

    async def close_async(self) -> None:
        """Close all providers asynchronously."""
        for provider in self._providers.values():
            await provider.close_async()
        self._providers.clear()

    def __enter__(self) -> "FallbackChain":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> "FallbackChain":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close_async()


class FallbackExhaustedError(Exception):
    """Raised when all models in fallback chain fail."""

    def __init__(
        self,
        message: str,
        attempts: Optional[List[Dict[str, Any]]] = None,
        last_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.attempts = attempts or []
        self.last_error = last_error


# Convenience functions for creating common fallback chains

def create_openai_chain(
    models: Optional[List[str]] = None,
    config: Optional[ProviderConfig] = None,
) -> FallbackChain:
    """
    Create a fallback chain using OpenAI models.

    Args:
        models: List of model names (defaults to gpt-4o -> gpt-4o-mini -> gpt-3.5-turbo)
        config: Provider configuration

    Returns:
        FallbackChain configured for OpenAI
    """
    if models is None:
        models = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]

    specs = [
        ModelSpec(provider="openai", model=model, config=config)
        for model in models
    ]

    return FallbackChain(FallbackChainConfig(models=specs))


def create_anthropic_chain(
    models: Optional[List[str]] = None,
    config: Optional[ProviderConfig] = None,
) -> FallbackChain:
    """
    Create a fallback chain using Anthropic models.

    Args:
        models: List of model names (defaults to claude-sonnet-4 -> claude-3-5-haiku)
        config: Provider configuration

    Returns:
        FallbackChain configured for Anthropic
    """
    if models is None:
        models = ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"]

    specs = [
        ModelSpec(provider="anthropic", model=model, config=config)
        for model in models
    ]

    return FallbackChain(FallbackChainConfig(models=specs))


def create_mixed_chain(
    models: Optional[List[tuple[str, str]]] = None,
    openai_config: Optional[ProviderConfig] = None,
    anthropic_config: Optional[ProviderConfig] = None,
) -> FallbackChain:
    """
    Create a fallback chain mixing providers.

    Args:
        models: List of (provider, model) tuples
        openai_config: Config for OpenAI
        anthropic_config: Config for Anthropic

    Returns:
        FallbackChain with mixed providers
    """
    if models is None:
        models = [
            ("anthropic", "claude-sonnet-4-20250514"),
            ("openai", "gpt-4o"),
            ("anthropic", "claude-3-5-haiku-20241022"),
            ("openai", "gpt-4o-mini"),
        ]

    configs = {
        "openai": openai_config,
        "anthropic": anthropic_config,
    }

    specs = [
        ModelSpec(provider=provider, model=model, config=configs.get(provider))
        for provider, model in models
    ]

    return FallbackChain(FallbackChainConfig(models=specs))
