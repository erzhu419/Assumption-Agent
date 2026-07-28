"""Production-only outer closure for the frozen QuAC P1 formal study."""

from .runner import (
    CONFIG_SCHEMA,
    OUTER_FAILURE_SCHEMA,
    OUTER_TERMINAL_SCHEMA,
    QuacP1FormalOuterError,
    SourceFreeFormalConfig,
    load_config,
    main,
    run_formal_production,
)

__all__ = [
    "CONFIG_SCHEMA",
    "OUTER_FAILURE_SCHEMA",
    "OUTER_TERMINAL_SCHEMA",
    "QuacP1FormalOuterError",
    "SourceFreeFormalConfig",
    "load_config",
    "main",
    "run_formal_production",
]
