"""Circuit breaker pattern for fault tolerance."""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, TypeVar, Optional, Any, Dict
from functools import wraps

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Circuit tripped, requests fail fast
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes in half-open before closing
    timeout: float = 30.0  # Seconds before trying half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open
    exclude_exceptions: tuple = field(default_factory=tuple)  # Don't count these as failures


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    def __init__(self, message: str, remaining_timeout: float = 0):
        super().__init__(message)
        self.remaining_timeout = remaining_timeout


class CircuitBreaker:
    """
    Circuit breaker implementation.

    States:
    - CLOSED: Normal operation. Failures increment counter.
    - OPEN: All requests fail fast. Timer runs.
    - HALF_OPEN: Limited requests allowed to test recovery.
    """

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_calls = 0

        self._lock = threading.RLock()
        self._async_lock: Optional[asyncio.Lock] = None

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, checking for timeout transition."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count

    @property
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (failing fast)."""
        return self.state == CircuitState.OPEN

    @property
    def is_half_open(self) -> bool:
        """Check if circuit is half-open (testing)."""
        return self.state == CircuitState.HALF_OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._last_failure_time is None:
            return True
        return time.time() - self._last_failure_time >= self.config.timeout

    def _get_remaining_timeout(self) -> float:
        """Get remaining time until half-open transition."""
        if self._last_failure_time is None:
            return 0
        elapsed = time.time() - self._last_failure_time
        return max(0, self.config.timeout - elapsed)

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0
        self._success_count = 0

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
        self._half_open_calls = 0

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    def _can_execute(self) -> bool:
        """Check if a request can be executed."""
        state = self.state  # This may transition open -> half-open

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # Half-open: limit concurrent calls
        if state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls

        return False

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                self._half_open_calls = max(0, self._half_open_calls - 1)

                if self._success_count >= self.config.success_threshold:
                    self._transition_to_closed()

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if isinstance(exception, self.config.exclude_exceptions):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls = max(0, self._half_open_calls - 1)
                self._transition_to_open()

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    self._transition_to_open()

    def _begin_call(self) -> None:
        """Mark beginning of a call (for half-open tracking)."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_calls += 1

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from func

        Raises:
            CircuitOpenError: If circuit is open
        """
        with self._lock:
            if not self._can_execute():
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open",
                    remaining_timeout=self._get_remaining_timeout(),
                )
            self._begin_call()

        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    async def call_async(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """
        Execute an async function through the circuit breaker.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from func

        Raises:
            CircuitOpenError: If circuit is open
        """
        # Use sync lock for state checks (quick operations)
        with self._lock:
            if not self._can_execute():
                raise CircuitOpenError(
                    f"Circuit '{self.name}' is open",
                    remaining_timeout=self._get_remaining_timeout(),
                )
            self._begin_call()

        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise

    def reset(self) -> None:
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to_closed()

    def trip(self) -> None:
        """Manually trip the circuit breaker to open state."""
        with self._lock:
            self._transition_to_open()

    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "remaining_timeout": self._get_remaining_timeout() if self._state == CircuitState.OPEN else 0,
            }


class CircuitBreakerRegistry:
    """Registry for managing multiple circuit breakers."""

    def __init__(self):
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._lock = threading.RLock()

    def get_or_create(
        self, name: str, config: Optional[CircuitBreakerConfig] = None
    ) -> CircuitBreaker:
        """Get existing circuit breaker or create new one."""
        with self._lock:
            if name not in self._breakers:
                self._breakers[name] = CircuitBreaker(name, config)
            return self._breakers[name]

    def get(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self._breakers.get(name)

    def all(self) -> Dict[str, CircuitBreaker]:
        """Get all circuit breakers."""
        return dict(self._breakers)

    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all circuit breakers."""
        return {name: breaker.get_stats() for name, breaker in self._breakers.items()}


# Global registry
_registry = CircuitBreakerRegistry()


def get_circuit_breaker(
    name: str, config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a circuit breaker from the global registry."""
    return _registry.get_or_create(name, config)


def get_registry() -> CircuitBreakerRegistry:
    """Get the global circuit breaker registry."""
    return _registry


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
):
    """
    Decorator for adding circuit breaker to synchronous functions.

    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
    """
    breaker = get_circuit_breaker(name, config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return breaker.call(func, *args, **kwargs)

        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator


def async_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
):
    """
    Decorator for adding circuit breaker to async functions.

    Args:
        name: Circuit breaker name
        config: Circuit breaker configuration
    """
    breaker = get_circuit_breaker(name, config)

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            return await breaker.call_async(func, *args, **kwargs)

        wrapper._circuit_breaker = breaker  # type: ignore
        return wrapper

    return decorator
