"""WASP arms (report tool, monitor announcement) for collusion experiments."""

from __future__ import annotations

# The report tool is not a variant: it comes with WaspJiraTicketChoiceEnvironment,
# selected through sweeps[].environments.
WASP_VARIANTS = ("none", "monitor")


def normalize_wasp_variant(value: str) -> str:
    variant = value.strip().lower() or "none"
    if variant not in WASP_VARIANTS:
        known = ", ".join(WASP_VARIANTS)
        raise ValueError(f"Unknown wasp_variant={variant!r}. Known: {known}.")
    return variant


def wasp_has_monitor(variant: str) -> bool:
    return variant == "monitor"
