"""Offline one-shot runtime for the frozen BioASQ P1 formal study.

The package intentionally performs no eager imports.  In particular, this
keeps ``python -m replication_runtime.bioasq_p1_formal_v1.runner`` from
loading the runner twice and keeps the source-free canary import surface
narrow.
"""

__all__: list[str] = []
