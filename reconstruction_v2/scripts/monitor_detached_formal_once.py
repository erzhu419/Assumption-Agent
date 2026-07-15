#!/usr/bin/env python3
from __future__ import annotations

"""Read the durable status of a detached one-shot formal run."""

from launch_detached_formal_once import main


if __name__ == "__main__":
    raise SystemExit(main(["status", *__import__("sys").argv[1:]]))
