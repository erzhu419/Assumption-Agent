"""
Harness runtime: loads a harness module, provides it with an LLM client +
KB access, and exposes a uniform `solve(problem: str) -> str` interface.

A harness is a single Python file that defines:
    def solve(problem: str, ctx: HarnessContext) -> str

ctx provides:
    ctx.generate(prompt, max_tokens=800, temperature=0.3) -> str
        — wraps Gemini Flash
    ctx.kb : Dict[sid, strategy dict]
        — all 27 strategies from phase zero/kb/strategies
    ctx.trace_samples : List[Dict]
        — a sample of 50 recent execution traces (compact form)
    ctx.log(msg) -> None
        — optional diagnostic logging

The harness CAN NOT:
    - import network/filesystem modules directly (sandboxed by convention)
    - do recursive LLM calls without bound (we track total calls per solve)
"""

from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
from llm_client import create_client  # noqa: E402

import mh_config as cfg  # noqa: E402


@dataclass
class HarnessContext:
    _client: object
    kb: Dict[str, Dict]
    trace_samples: List[Dict]
    call_count: int = 0
    max_calls: int = 10
    logs: List[str] = field(default_factory=list)

    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        if self.call_count >= self.max_calls:
            raise RuntimeError(f"Harness exceeded {self.max_calls} LLM calls per solve")
        self.call_count += 1
        resp = self._client.generate(prompt, max_tokens=max_tokens, temperature=temperature)
        return resp["text"]

    def log(self, msg: str):
        self.logs.append(str(msg))


def load_kb() -> Dict[str, Dict]:
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    return kb


def load_trace_samples(n: int = 50) -> List[Dict]:
    """Compact sample of recent execution traces for the proposer to reference."""
    all_paths = sorted(cfg.EXECUTIONS_DIR.glob("exec_*.json"))
    if not all_paths:
        return []
    random.seed(0)
    chosen = random.sample(all_paths, min(n, len(all_paths)))
    samples = []
    for p in chosen:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            samples.append({
                "task_id": d["task"]["task_id"],
                "domain": d["task"].get("domain", ""),
                "difficulty": d["task"].get("difficulty", ""),
                "selected_strategy": d["strategy_selection"]["selected_strategy"],
                "success": d["outcome"]["success"],
                "features": d["task"].get("complexity_features", {}),
            })
        except Exception:
            continue
    return samples


def load_harness(harness_path: Path) -> Callable:
    """Dynamically load a harness module. Returns its solve() function."""
    spec = importlib.util.spec_from_file_location(
        f"harness_{harness_path.stem}", str(harness_path)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {harness_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise AttributeError(f"{harness_path} missing solve() function")
    return module.solve


def make_context(client: Optional[object] = None, max_calls: int = 10) -> HarnessContext:
    if client is None:
        client = create_client()
    return HarnessContext(
        _client=client, kb=load_kb(),
        trace_samples=load_trace_samples(), max_calls=max_calls,
    )


def run_harness(harness_path: Path, problem: str,
                ctx: HarnessContext) -> Dict:
    """Run a harness on a single problem. Returns answer + metadata."""
    # Reset ctx per-problem counters
    ctx.call_count = 0
    ctx.logs = []
    solve = load_harness(harness_path)
    t0 = time.time()
    try:
        answer = solve(problem, ctx)
    except Exception as e:
        return {
            "answer": f"[HARNESS ERROR] {e}",
            "llm_calls": ctx.call_count,
            "seconds": time.time() - t0,
            "error": str(e),
        }
    return {
        "answer": (answer or "").strip(),
        "llm_calls": ctx.call_count,
        "seconds": time.time() - t0,
        "error": None,
        "logs": ctx.logs,
    }
