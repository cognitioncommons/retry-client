# retry-client

A robust LLM client wrapper with retry, fallback, and circuit breaker patterns.

## Features

- **Exponential Backoff with Jitter**: Automatic retries with configurable backoff
- **Fallback Model Chains**: Automatically fall back to alternative models/providers
- **Circuit Breaker Pattern**: Prevent cascading failures when services are down
- **Rate Limit Detection**: Intelligent handling of rate limits with retry-after headers
- **Multi-Provider Support**: OpenAI and Anthropic out of the box

## Installation

```bash
pip install retry-client
```

Or install from source:

```bash
pip install -e /opt/cognitioncommons-tools/retry-client
```

## Quick Start

### Python API

```python
from retry_client import RetryClient, Message

# Simple usage with defaults
client = RetryClient()
response = client.chat("Hello, how are you?")
print(response)

# With message list
messages = [
    Message.system("You are a helpful assistant."),
    Message.user("What is 2+2?"),
]
response = client.complete(messages)
print(response.content)
```

### CLI

```bash
# Send a message
retry-client send "What is 2+2?"

# Use a specific provider
retry-client -p anthropic send "Hello"

# Interactive chat
retry-client chat

# Check provider health
retry-client health --all

# List available models
retry-client models
```

## Configuration

### Client Configuration

```python
from retry_client import RetryClient, ClientConfig, RetryConfig, CircuitBreakerConfig

config = ClientConfig(
    # Provider settings
    default_provider="anthropic",  # or "openai"
    default_model="claude-sonnet-4-20250514",

    # Retry settings
    retry_enabled=True,
    retry_config=RetryConfig(
        max_retries=5,
        base_delay=1.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=0.5,
    ),

    # Circuit breaker settings
    circuit_enabled=True,
    circuit_config=CircuitBreakerConfig(
        failure_threshold=5,
        success_threshold=2,
        timeout=30.0,
    ),

    # Fallback settings
    fallback_enabled=True,
    fallback_models=[
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
        ("anthropic", "claude-3-5-haiku-20241022"),
        ("openai", "gpt-4o-mini"),
    ],

    # Callbacks
    on_retry=lambda attempt, error, delay: print(f"Retry {attempt}: {error}"),
    on_fallback=lambda from_m, to_m, error: print(f"Fallback: {from_m} -> {to_m}"),
)

client = RetryClient(config)
```

### Using Decorators

```python
from retry_client import with_retry, circuit_breaker, RetryConfig

# Retry decorator for sync functions
@with_retry(RetryConfig(max_retries=3))
def my_api_call():
    # Your code here
    pass

# Async retry decorator
@with_async_retry(RetryConfig(max_retries=3))
async def my_async_call():
    # Your async code here
    pass

# Circuit breaker decorator
@circuit_breaker("my_service")
def call_external_service():
    # Your code here
    pass
```

## Retry Logic

The retry module implements exponential backoff with jitter:

```python
from retry_client import RetryConfig, retry_sync

config = RetryConfig(
    max_retries=5,        # Maximum retry attempts
    base_delay=1.0,       # Initial delay in seconds
    max_delay=60.0,       # Maximum delay cap
    exponential_base=2.0, # Backoff multiplier
    jitter=0.5,           # Random jitter factor (0-1)
)

# Use with any function
result = retry_sync(my_function, arg1, arg2, config=config)
```

## Fallback Chains

Create resilient chains that try multiple models:

```python
from retry_client import (
    FallbackChain, FallbackChainConfig, ModelSpec,
    create_openai_chain, create_anthropic_chain, create_mixed_chain,
)

# Quick chain creation
chain = create_mixed_chain()
result = chain.complete([Message.user("Hello")])
print(result.response.content)

# Custom chain
specs = [
    ModelSpec(provider="anthropic", model="claude-sonnet-4-20250514"),
    ModelSpec(provider="openai", model="gpt-4o"),
]
chain = FallbackChain(FallbackChainConfig(models=specs))
```

## Circuit Breaker

Prevent cascading failures:

```python
from retry_client import CircuitBreaker, CircuitBreakerConfig

config = CircuitBreakerConfig(
    failure_threshold=5,    # Failures before opening
    success_threshold=2,    # Successes to close
    timeout=30.0,           # Seconds before half-open
)

breaker = CircuitBreaker("my_service", config)

try:
    result = breaker.call(my_function, *args)
except CircuitOpenError:
    # Circuit is open, fail fast
    pass
```

## Providers

### OpenAI

```python
from retry_client import OpenAIProvider, ProviderConfig

provider = OpenAIProvider(ProviderConfig(
    api_key="sk-...",  # Or use OPENAI_API_KEY env var
    timeout=60.0,
    max_tokens=4096,
))

response = provider.complete([Message.user("Hello")])
```

### Anthropic

```python
from retry_client import AnthropicProvider, ProviderConfig

provider = AnthropicProvider(ProviderConfig(
    api_key="sk-ant-...",  # Or use ANTHROPIC_API_KEY env var
    timeout=60.0,
    max_tokens=4096,
))

response = provider.complete([Message.user("Hello")])
```

## Async Support

All operations support async:

```python
import asyncio
from retry_client import RetryClient

async def main():
    client = RetryClient()

    # Async completion
    response = await client.complete_async([Message.user("Hello")])

    # Async streaming
    async for chunk in client.stream_async([Message.user("Tell me a story")]):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

## CLI Reference

```bash
# Global options
retry-client --provider/-p [openai|anthropic]  # Provider to use
retry-client --model/-m MODEL                   # Model to use
retry-client --api-key/-k KEY                   # API key
retry-client --no-retry                         # Disable retries
retry-client --no-fallback                      # Disable fallback
retry-client --verbose/-v                       # Verbose output

# Commands
retry-client send MESSAGE          # Send a single message
retry-client chat                  # Interactive chat
retry-client health                # Check provider health
retry-client circuits              # Show circuit breaker status
retry-client reset                 # Reset circuit breakers
retry-client models                # List available models
```

## Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key

## License

MIT
