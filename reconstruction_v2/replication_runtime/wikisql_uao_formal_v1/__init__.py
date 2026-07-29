"""One-shot outer controller package for the frozen WikiSQL UAO P4 study.

The package deliberately does not import :mod:`runner`: the production entry
point uses ``python -m ...runner``, and eager re-export would execute that
module once during package import and then a second time through ``runpy``.
"""

__all__: list[str] = []
