"""Public Q0.5b qualification-wire import surface.

The implementation lives in the explicitly qualification-only module below;
this stable name is what isolated Python/Rust/host adapters bind.  Importing it
does not start Q1 or alter any formal gate/root state.
"""

from .phase3_q05b_wire_qualification_contract_v1 import *  # noqa: F403
from .phase3_q05b_wire_qualification_contract_v1 import __all__
