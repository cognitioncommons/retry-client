"""Command-line interface for retry-client."""

import json
import sys
from typing import Optional

import click

from .client import RetryClient, ClientConfig, create_client
from .providers.base import Message, ProviderConfig


@click.group()
@click.version_option(version="0.1.0", prog_name="retry-client")
@click.option("--provider", "-p", type=click.Choice(["openai", "anthropic"]), default="openai",
              help="LLM provider to use")
@click.option("--model", "-m", help="Model to use")
@click.option("--api-key", "-k", envvar=["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
              help="API key (or set via environment variable)")
@click.option("--no-retry", is_flag=True, help="Disable retry logic")
@click.option("--no-fallback", is_flag=True, help="Disable fallback chain")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx: click.Context, provider: str, model: Optional[str], api_key: Optional[str],
        no_retry: bool, no_fallback: bool, verbose: bool) -> None:
    """Robust LLM client with retry, fallback, and circuit breaker."""
    ctx.ensure_object(dict)
    ctx.obj["provider"] = provider
    ctx.obj["model"] = model
    ctx.obj["api_key"] = api_key
    ctx.obj["retry_enabled"] = not no_retry
    ctx.obj["fallback_enabled"] = not no_fallback
    ctx.obj["verbose"] = verbose


def get_client(ctx: click.Context) -> RetryClient:
    """Create client from context."""
    return create_client(
        provider=ctx.obj["provider"],
        model=ctx.obj["model"],
        api_key=ctx.obj["api_key"],
        retry_enabled=ctx.obj["retry_enabled"],
        fallback_enabled=ctx.obj["fallback_enabled"],
    )


@cli.command()
@click.argument("message")
@click.option("--system", "-s", help="System prompt")
@click.option("--max-tokens", "-t", type=int, default=1024, help="Maximum tokens in response")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def send(ctx: click.Context, message: str, system: Optional[str], max_tokens: int,
         temperature: float, json_output: bool) -> None:
    """Send a single message and get a response.

    Example:
        retry-client send "What is 2+2?"
        retry-client -p anthropic send "Hello" --system "You are helpful."
    """
    try:
        with get_client(ctx) as client:
            messages = []
            if system:
                messages.append(Message.system(system))
            messages.append(Message.user(message))

            if ctx.obj["verbose"]:
                click.echo(f"Provider: {ctx.obj['provider']}", err=True)
                click.echo(f"Model: {ctx.obj['model'] or 'default'}", err=True)
                click.echo(f"Retry: {ctx.obj['retry_enabled']}", err=True)
                click.echo(f"Fallback: {ctx.obj['fallback_enabled']}", err=True)
                click.echo("---", err=True)

            response = client.complete(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            if json_output:
                output = {
                    "content": response.content,
                    "model": response.model,
                    "provider": response.provider,
                    "finish_reason": response.finish_reason,
                    "usage": response.usage,
                }
                click.echo(json.dumps(output, indent=2))
            else:
                click.echo(response.content)

    except Exception as e:
        if ctx.obj["verbose"]:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--system", "-s", help="System prompt")
@click.option("--max-tokens", "-t", type=int, default=1024, help="Maximum tokens in response")
@click.option("--temperature", type=float, default=0.7, help="Sampling temperature")
@click.pass_context
def chat(ctx: click.Context, system: Optional[str], max_tokens: int, temperature: float) -> None:
    """Interactive chat session.

    Start a conversation with the LLM. Type 'quit' or 'exit' to end.

    Example:
        retry-client chat
        retry-client -p anthropic chat --system "You are a pirate."
    """
    click.echo("Starting chat session. Type 'quit' or 'exit' to end.")
    click.echo(f"Provider: {ctx.obj['provider']}, Model: {ctx.obj['model'] or 'default'}")
    click.echo("---")

    conversation: list[Message] = []
    if system:
        conversation.append(Message.system(system))

    try:
        with get_client(ctx) as client:
            while True:
                try:
                    user_input = click.prompt("You", prompt_suffix=": ")
                except (EOFError, KeyboardInterrupt):
                    click.echo("\nGoodbye!")
                    break

                if user_input.lower() in ("quit", "exit", "q"):
                    click.echo("Goodbye!")
                    break

                if not user_input.strip():
                    continue

                conversation.append(Message.user(user_input))

                try:
                    response = client.complete(
                        conversation,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    assistant_message = response.content
                    conversation.append(Message.assistant(assistant_message))

                    click.echo(f"\nAssistant: {assistant_message}\n")

                except Exception as e:
                    click.echo(f"\nError: {e}\n", err=True)
                    # Remove the failed user message
                    conversation.pop()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--all", "-a", "check_all", is_flag=True, help="Check all providers")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def health(ctx: click.Context, check_all: bool, json_output: bool) -> None:
    """Check provider health.

    Example:
        retry-client health
        retry-client health --all
        retry-client -p anthropic health
    """
    try:
        with get_client(ctx) as client:
            if check_all:
                results = client.health_check()
            else:
                results = client.health_check(ctx.obj["provider"])

            if json_output:
                click.echo(json.dumps(results, indent=2))
            else:
                if isinstance(results, dict) and "status" in results:
                    # Single provider result
                    results = {ctx.obj["provider"]: results}

                for provider_name, status in results.items():
                    if isinstance(status, dict):
                        state = status.get("status", "unknown")
                        icon = "[OK]" if state == "healthy" else "[FAIL]"
                        model = status.get("model", "N/A")
                        error = status.get("error", "")

                        click.echo(f"{icon} {provider_name}")
                        if state == "healthy":
                            click.echo(f"    Model: {model}")
                        else:
                            click.echo(f"    Error: {error}")
                    else:
                        click.echo(f"[?] {provider_name}: {status}")

    except Exception as e:
        if ctx.obj["verbose"]:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
@click.pass_context
def circuits(ctx: click.Context, json_output: bool) -> None:
    """Show circuit breaker status.

    Example:
        retry-client circuits
    """
    try:
        with get_client(ctx) as client:
            stats = client.get_circuit_stats()

            if json_output:
                click.echo(json.dumps(stats, indent=2))
            else:
                if not stats:
                    click.echo("No circuit breakers active.")
                else:
                    for name, data in stats.items():
                        state = data.get("state", "unknown")
                        failures = data.get("failure_count", 0)
                        remaining = data.get("remaining_timeout", 0)

                        icon = "[CLOSED]" if state == "closed" else ("[OPEN]" if state == "open" else "[HALF]")
                        click.echo(f"{icon} {name}")
                        click.echo(f"    Failures: {failures}")
                        if state == "open" and remaining > 0:
                            click.echo(f"    Timeout: {remaining:.1f}s remaining")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def reset(ctx: click.Context) -> None:
    """Reset all circuit breakers.

    Example:
        retry-client reset
    """
    try:
        with get_client(ctx) as client:
            client.reset_circuits()
            click.echo("All circuit breakers reset.")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def models() -> None:
    """List available models by provider.

    Example:
        retry-client models
    """
    from .providers.openai import OpenAIProvider
    from .providers.anthropic import AnthropicProvider

    click.echo("OpenAI Models:")
    for model in OpenAIProvider.available_models:
        default = " (default)" if model == OpenAIProvider.default_model else ""
        click.echo(f"  - {model}{default}")

    click.echo("\nAnthropic Models:")
    for model in AnthropicProvider.available_models:
        default = " (default)" if model == AnthropicProvider.default_model else ""
        click.echo(f"  - {model}{default}")


def main() -> None:
    """Entry point for CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
