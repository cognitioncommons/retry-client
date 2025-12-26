"""Exponential backoff with jitter for retry logic."""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar, Optional, Any
from functools import wraps

T = TypeVar("T")


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: float = 0.5  # Random jitter factor (0-1)
    retryable_exceptions: tuple = field(default_factory=lambda: (Exception,))
    retryable_status_codes: tuple = field(default_factory=lambda: (429, 500, 502, 503, 504))


class RetryState:
    """Tracks retry state across attempts."""

    def __init__(self, config: RetryConfig):
        self.config = config
        self.attempt = 0
        self.last_exception: Optional[Exception] = None
        self.total_delay = 0.0

    def should_retry(self, exception: Exception) -> bool:
        """Determine if we should retry based on exception type."""
        if self.attempt >= self.config.max_retries:
            return False
        return isinstance(exception, self.config.retryable_exceptions)

    def get_delay(self) -> float:
        """Calculate delay with exponential backoff and jitter."""
        # Exponential backoff
        delay = self.config.base_delay * (self.config.exponential_base ** self.attempt)

        # Cap at max delay
        delay = min(delay, self.config.max_delay)

        # Add jitter: random value between (1 - jitter) and (1 + jitter) of delay
        jitter_range = delay * self.config.jitter
        delay = delay + random.uniform(-jitter_range, jitter_range)

        # Ensure delay is positive
        delay = max(0.01, delay)

        self.total_delay += delay
        return delay

    def increment(self, exception: Exception) -> None:
        """Increment attempt counter and store exception."""
        self.attempt += 1
        self.last_exception = exception


class RateLimitError(Exception):
    """Raised when rate limit is detected."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class RetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""

    def __init__(self, message: str, last_exception: Optional[Exception] = None, attempts: int = 0):
        super().__init__(message)
        self.last_exception = last_exception
        self.attempts = attempts


def extract_retry_after(exception: Exception) -> Optional[float]:
    """Extract retry-after header from various exception types."""
    # Check for httpx response
    if hasattr(exception, 'response') and exception.response is not None:
        response = exception.response
        if hasattr(response, 'headers'):
            retry_after = response.headers.get('retry-after') or response.headers.get('Retry-After')
            if retry_after:
                try:
                    return float(retry_after)
                except ValueError:
                    pass

    # Check for rate limit info in exception attributes
    if hasattr(exception, 'retry_after'):
        return exception.retry_after

    return None


def is_rate_limit_error(exception: Exception, status_code: Optional[int] = None) -> bool:
    """Detect if exception is a rate limit error."""
    # Check status code
    if status_code == 429:
        return True

    # Check exception response
    if hasattr(exception, 'response') and exception.response is not None:
        if hasattr(exception.response, 'status_code'):
            if exception.response.status_code == 429:
                return True

    # Check exception message
    message = str(exception).lower()
    rate_limit_indicators = ['rate limit', 'ratelimit', 'too many requests', 'quota exceeded']
    return any(indicator in message for indicator in rate_limit_indicators)


def with_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Decorator for adding retry logic to synchronous functions.

    Args:
        config: Retry configuration
        on_retry: Callback called before each retry (attempt, exception, delay)
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            state = RetryState(config)

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if is_rate_limit_error(e):
                        retry_after = extract_retry_after(e)
                        if retry_after and retry_after > config.max_delay:
                            raise RateLimitError(str(e), retry_after) from e

                    if not state.should_retry(e):
                        raise RetryExhaustedError(
                            f"Retry exhausted after {state.attempt} attempts",
                            last_exception=e,
                            attempts=state.attempt,
                        ) from e

                    state.increment(e)
                    delay = state.get_delay()

                    # Use retry-after if available and reasonable
                    retry_after = extract_retry_after(e)
                    if retry_after and retry_after <= config.max_delay:
                        delay = retry_after

                    if on_retry:
                        on_retry(state.attempt, e, delay)

                    time.sleep(delay)

        return wrapper
    return decorator


def with_async_retry(
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
):
    """
    Decorator for adding retry logic to async functions.

    Args:
        config: Retry configuration
        on_retry: Callback called before each retry (attempt, exception, delay)
    """
    if config is None:
        config = RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            state = RetryState(config)

            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if is_rate_limit_error(e):
                        retry_after = extract_retry_after(e)
                        if retry_after and retry_after > config.max_delay:
                            raise RateLimitError(str(e), retry_after) from e

                    if not state.should_retry(e):
                        raise RetryExhaustedError(
                            f"Retry exhausted after {state.attempt} attempts",
                            last_exception=e,
                            attempts=state.attempt,
                        ) from e

                    state.increment(e)
                    delay = state.get_delay()

                    # Use retry-after if available and reasonable
                    retry_after = extract_retry_after(e)
                    if retry_after and retry_after <= config.max_delay:
                        delay = retry_after

                    if on_retry:
                        on_retry(state.attempt, e, delay)

                    await asyncio.sleep(delay)

        return wrapper
    return decorator


async def retry_async(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        on_retry: Callback called before each retry
        **kwargs: Keyword arguments for func

    Returns:
        Result from func
    """
    if config is None:
        config = RetryConfig()

    state = RetryState(config)

    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if is_rate_limit_error(e):
                retry_after = extract_retry_after(e)
                if retry_after and retry_after > config.max_delay:
                    raise RateLimitError(str(e), retry_after) from e

            if not state.should_retry(e):
                raise RetryExhaustedError(
                    f"Retry exhausted after {state.attempt} attempts",
                    last_exception=e,
                    attempts=state.attempt,
                ) from e

            state.increment(e)
            delay = state.get_delay()

            # Use retry-after if available and reasonable
            retry_after = extract_retry_after(e)
            if retry_after and retry_after <= config.max_delay:
                delay = retry_after

            if on_retry:
                on_retry(state.attempt, e, delay)

            await asyncio.sleep(delay)


def retry_sync(
    func: Callable[..., T],
    *args: Any,
    config: Optional[RetryConfig] = None,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
    **kwargs: Any,
) -> T:
    """
    Execute a synchronous function with retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for func
        config: Retry configuration
        on_retry: Callback called before each retry
        **kwargs: Keyword arguments for func

    Returns:
        Result from func
    """
    if config is None:
        config = RetryConfig()

    state = RetryState(config)

    while True:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if is_rate_limit_error(e):
                retry_after = extract_retry_after(e)
                if retry_after and retry_after > config.max_delay:
                    raise RateLimitError(str(e), retry_after) from e

            if not state.should_retry(e):
                raise RetryExhaustedError(
                    f"Retry exhausted after {state.attempt} attempts",
                    last_exception=e,
                    attempts=state.attempt,
                ) from e

            state.increment(e)
            delay = state.get_delay()

            # Use retry-after if available and reasonable
            retry_after = extract_retry_after(e)
            if retry_after and retry_after <= config.max_delay:
                delay = retry_after

            if on_retry:
                on_retry(state.attempt, e, delay)

            time.sleep(delay)
