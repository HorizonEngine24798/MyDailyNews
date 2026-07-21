from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenBudget:
    context_tokens: int
    input_tokens: int
    output_tokens: int
    reserve_tokens: int


def resolve_token_budget(
    *,
    context_tokens: int,
    max_input_tokens: int,
    max_output_tokens: int,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    reserve_tokens: int | None = None,
) -> TokenBudget:
    """Resolve one request budget without exceeding the model context."""
    context = int(context_tokens)
    max_input = int(max_input_tokens)
    max_output = int(max_output_tokens)
    requested_input = max_input if input_tokens is None else int(input_tokens)
    requested_output = max_output if output_tokens is None else int(output_tokens)
    reserve = max(256, context // 16) if reserve_tokens is None else int(reserve_tokens)

    values = {
        "context_tokens": context,
        "max_input_tokens": max_input,
        "max_output_tokens": max_output,
        "input_tokens": requested_input,
        "output_tokens": requested_output,
        "reserve_tokens": reserve,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"token budgets must be positive: {', '.join(invalid)}")

    usable = context - reserve
    if usable < 128:
        raise ValueError("context_tokens must leave at least 128 tokens after reserve")

    output = min(requested_output, max_output, usable - 64)
    input_limit = min(requested_input, max_input, usable - output)
    return TokenBudget(
        context_tokens=context,
        input_tokens=input_limit,
        output_tokens=output,
        reserve_tokens=reserve,
    )


def resolve_client_token_budget(
    client: Any,
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> TokenBudget:
    """Resolve stage preferences against an AI client's actual capacity."""
    config = getattr(client, "config", None)
    max_output = int(getattr(client, "max_new_tokens", getattr(config, "max_new_tokens", 0)) or 0)
    max_input = int(getattr(client, "max_input_tokens", getattr(config, "max_input_tokens", max_output)) or max_output)
    context = int(getattr(config, "context_window_tokens", 0) or 0)
    if context <= 0:
        context = max_input + max_output + 256
        reserve_tokens = 256
    else:
        reserve_tokens = None
    return resolve_token_budget(
        context_tokens=context,
        max_input_tokens=max_input,
        max_output_tokens=max_output,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reserve_tokens=reserve_tokens,
    )
