#!/usr/bin/env python3
"""
Orthogonal Decomposition Gate v1
================================
Evaluates wisdom candidates across four independent validity components:
  1. reframe_depth             – cognitive reframing beyond surface rephrasing
  2. substantive_content_delta – real content change, not just formatting
  3. wisdom_problem_alignment  – semantic relevance of wisdom to problem
  4. antipattern_avoidance     – absence of known validity-threatening patterns

This script is fully self-contained. It attempts to import an external data
API (exp21_data_api) but falls back to reading canonical JSON files from disk.

Usage:
    python exp21_agent_gate.py [--data-dir PATH] [--output PATH]
"""

import sys
import os
import json
import math
import re
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Orthogonal Decomposition Gate")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Root directory containing data files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Lightweight text utilities (no external ML dependencies required)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    if not text:
        return []
    text = text.lower().strip()
    tokens = re.findall(r"[a-z0-9\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff]+", text)
    return tokens


def _char_ngrams(text: str, n: int = 3) -> Dict[str, int]:
    """Character n-gram frequency vector."""
    text = text.lower().strip()
    grams: Dict[str, int] = {}
    for i in range(max(0, len(text) - n + 1)):
        g = text[i:i + n]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _word_ngrams(text: str, n: int = 1) -> Dict[str, int]:
    """Word n-gram frequency vector."""
    tokens = tokenize(text)
    grams: Dict[str, int] = {}
    for i in range(max(0, len(tokens) - n + 1)):
        g = " ".join(tokens[i:i + n])
        grams[g] = grams.get(g, 0) + 1
    return grams


def cosine_sim_vectors(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors represented as dicts."""
    if not a or not b:
        return 0.0
    keys = set(a.keys()) | set(b.keys())
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a < 1e-12 or norm_b < 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def text_cosine_distance(text_a: str, text_b: str) -> float:
    """Approximate cosine distance using combined char-3-gram + word-unigram vectors."""
    if not text_a or not text_b:
        return 1.0
    ca = _char_ngrams(text_a, 3)
    cb = _char_ngrams(text_b, 3)
    sim_char = cosine_sim_vectors(ca, cb)

    wa = _word_ngrams(text_a, 1)
    wb = _word_ngrams(text_b, 1)
    sim_word = cosine_sim_vectors(wa, wb)

    ba = _word_ngrams(text_a, 2)
    bb = _word_ngrams(text_b, 2)
    sim_bi = cosine_sim_vectors(ba, bb)

    sim = 0.3 * sim_char + 0.4 * sim_word + 0.3 * sim_bi
    return max(0.0, min(1.0, 1.0 - sim))


def text_cosine_similarity(text_a: str, text_b: str) -> float:
    return 1.0 - text_cosine_distance(text_a, text_b)


def levenshtein_ratio(s1: str, s2: str) -> float:
    """Levenshtein edit distance ratio (token-level)."""
    t1 = tokenize(s1)
    t2 = tokenize(s2)
    if not t1 and not t2:
        return 0.0
    max_len = max(len(t1), len(t2))
    if max_len == 0:
        return 0.0

    m, n = len(t1), len(t2)
    if m > 500 or n > 500:
        s1_set = set(t1)
        s2_set = set(t2)
        if not s1_set and not s2_set:
            return 0.0
        jaccard = len(s1_set & s2_set) / max(1, len(s1_set | s2_set))
        return 1.0 - jaccard

    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if t1[i - 1] == t2[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n] / max_len


def median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


# ---------------------------------------------------------------------------
# Data loading layer
# ---------------------------------------------------------------------------

class DataLoader:
    """Loads experiment data from disk JSON files."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = self._find_data_dir(data_dir)
        self._wisdom_lib = None
        self._candidates_cache: Dict[str, Any] = {}
        self._records_cache: Dict[str, List[Dict]] = {}

    @staticmethod
    def _find_data_dir(hint: Optional[str] = None) -> Path:
        """Search for the data directory."""
        script_dir = Path(__file__).resolve().parent
        candidates_dirs = []
        if hint:
            candidates_dirs.append(Path(hint).resolve())
        for base in [script_dir, script_dir.parent, script_dir.parent.parent,
                     script_dir.parent.parent.parent]:
            for sub in ["", "data", "phase four/data", "phase four",
                        "phase zero/data", "phase zero", "generated"]:
                candidates_dirs.append((base / sub).resolve())

        for d in candidates_dirs:
            if not d.is_dir():
                continue
            if ((d / "wisdom_library.json").is_file() or
                (d / "candidates").is_dir() or
                (d / "answers").is_dir() or
                any(d.glob("*_answers.json")) or
                any(d.glob("*_meta.json"))):
                return d
        return script_dir

    def wisdom_library(self) -> Dict[str, Any]:
        if self._wisdom_lib is not None:
            return self._wisdom_lib
        for name in ["wisdom_library.json", "wisdoms.json", "wisdom.json"]:
            fp = self.data_dir / name
            if fp.is_file():
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    self._wisdom_lib = {str(i): v for i, v in enumerate(data)}
                elif isinstance(data, dict):
                    self._wisdom_lib = data
                else:
                    self._wisdom_lib = {}
                return self._wisdom_lib

        for parent in [self.data_dir.parent, self.data_dir.parent.parent]:
            for name in ["wisdom_library.json", "wisdoms.json"]:
                fp = parent / name
                if fp.is_file():
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        self._wisdom_lib = {str(i): v for i, v in enumerate(data)}
                    elif isinstance(data, dict):
                        self._wisdom_lib = data
                    else:
                        self._wisdom_lib = {}
                    return self._wisdom_lib

        self._wisdom_lib = {}
        return self._wisdom_lib

    def list_candidates(self) -> List[str]:
        lib = self.wisdom_library()
        if lib:
            return list(lib.keys())
        cdir = self.data_dir / "candidates"
        if cdir.is_dir():
            return [d.name for d in cdir.iterdir() if d.is_dir()]
        return []

    def candidate_info(self, cid: str) -> Dict[str, Any]:
        if cid in self._candidates_cache:
            return self._candidates_cache[cid]
        lib = self.wisdom_library()
        entry = lib.get(cid, lib.get(str(cid), {}))
        if isinstance(entry, str):
            entry = {"text": entry, "aphorism": entry}
        if not isinstance(entry, dict):
            entry = {"raw": entry}
        self._candidates_cache[cid] = entry
        return entry

    def _get_wisdom_text(self, cid: str) -> str:
        info = self.candidate_info(cid)
        for key in ["unpacked", "text", "aphorism", "wisdom", "content", "description"]:
            if key in info and isinstance(info[key], str):
                return info[key]
        parts = [str(v) for v in info.values() if isinstance(v, str)]
        return " ".join(parts) if parts else str(cid)

    def per_pid_records(self, cid: str) -> List[Dict[str, Any]]:
        """Load per-prompt-id records for a candidate."""
        if cid in self._records_cache:
            return self._records_cache[cid]

        records: List[Dict[str, Any]] = []

        # Preferred strategy: external API if available
        try:
            import exp21_data_api  # type: ignore
            if hasattr(exp21_data_api, "per_pid_records"):
                api_records = exp21_data_api.per_pid_records(cid)
                if isinstance(api_records, list):
                    self._records_cache[cid] = api_records
                    return api_records
        except Exception:
            pass

        search_dirs = [
            self.data_dir / "answers",
            self.data_dir / "candidates" / str(cid),
            self.data_dir / "candidates" / str(cid) / "answers",
            self.data_dir,
        ]

        answers_data = None
        meta_data = None

        for sdir in search_dirs:
            if not sdir.is_dir():
                continue
            for fp in sorted(sdir.glob("*.json")):
                fname = fp.name.lower()
                if str(cid).lower() in fname or "answer" in fname:
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if "answer" in fname:
                            answers_data = data
                        if "meta" in fname:
                            meta_data = data
                    except (json.JSONDecodeError, IOError):
                        pass

        info = self.candidate_info(cid)
        if "records" in info and isinstance(info["records"], list):
            records = info["records"]
        elif "prompts" in info and isinstance(info["prompts"], list):
            records = info["prompts"]
        elif "per_pid" in info and isinstance(info["per_pid"], list):
            records = info["per_pid"]

        if answers_data:
            if isinstance(answers_data, list):
                for item in answers_data:
                    if isinstance(item, dict):
                        records.append(item)
            elif isinstance(answers_data, dict):
                if "records" in answers_data:
                    records.extend(answers_data["records"])
                elif "prompts" in answers_data:
                    records.extend(answers_data["prompts"])
                else:
                    for pid, val in answers_data.items():
                        if isinstance(val, dict):
                            val["prompt_id"] = pid
                            records.append(val)

        if meta_data and isinstance(meta_data, dict):
            meta_by_pid = {}
            if isinstance(meta_data, list):
                for m in meta_data:
                    if isinstance(m, dict) and "prompt_id" in m:
                        meta_by_pid[m["prompt_id"]] = m
            elif isinstance(meta_data, dict):
                for pid, val in meta_data.items():
                    if isinstance(val, dict):
                        meta_by_pid[pid] = val
            for rec in records:
                pid = rec.get("prompt_id", rec.get("pid", ""))
                if pid in meta_by_pid:
                    rec["meta"] = meta_by_pid[pid]

        self._records_cache[cid] = records
        return records

    def get_all_wisdom_texts(self) -> Dict[str, str]:
        """Return {cid: wisdom_text} for all candidates."""
        result = {}
        for cid in self.list_candidates():
            result[cid] = self._get_wisdom_text(cid)
        return result


# ---------------------------------------------------------------------------
# Gate Component Evaluators
# ---------------------------------------------------------------------------

class ReframeDepthEvaluator:
    """Component 1: Measures cognitive reframing depth."""

    THRESHOLD_DISTANCE = 0.25
    MIN_PROMPTS = 5

    def evaluate(self, cid: str, loader: DataLoader) -> Dict[str, Any]:
        records = loader.per_pid_records(cid)
        distances = []

        for rec in records:
            # FIX: use canonical API field name `problem`
            prompt_text = rec.get("problem", "")
            # FIX: use canonical API field names `ext_what_changed` / `base_what_changed`
            reframe = (rec.get("ext_what_changed", "") or
                       rec.get("base_what_changed", ""))

            if prompt_text and reframe:
                dist = text_cosine_distance(prompt_text, reframe)
                distances.append(dist)

        n_prompts = len(distances)
        med_dist = median(distances) if distances else 0.0

        passed = (med_dist >= self.THRESHOLD_DISTANCE and n_prompts >= self.MIN_PROMPTS)

        return {
            "component": "reframe_depth",
            "passed": passed,
            "median_reframe_distance": round(med_dist, 4),
            "n_prompts": n_prompts,
            "threshold_distance": self.THRESHOLD_DISTANCE,
            "min_prompts": self.MIN_PROMPTS,
            "details": f"median_dist={med_dist:.4f}, n={n_prompts}"
        }


class SubstantiveContentDeltaEvaluator:
    """Component 2: Measures substantive content change between base and ext answers."""

    THRESHOLD_MEDIAN = 0.20
    THRESHOLD_FRACTION = 0.60
    THRESHOLD_PER_PROMPT = 0.15

    def evaluate(self, cid: str, loader: DataLoader) -> Dict[str, Any]:
        records = loader.per_pid_records(cid)
        deltas = []

        for rec in records:
            base = rec.get("base_answer", "")
            ext = rec.get("ext_answer", "")
            if base and ext:
                delta = text_cosine_distance(base, ext)
                deltas.append(delta)

        med_delta = median(deltas) if deltas else 0.0
        frac_above = (sum(1 for d in deltas if d >= self.THRESHOLD_PER_PROMPT) / len(deltas)) if deltas else 0.0
        passed = (med_delta >= self.THRESHOLD_MEDIAN and frac_above >= self.THRESHOLD_FRACTION)

        return {
            "component": "substantive_content_delta",
            "passed": passed,
            "median_content_delta": round(med_delta, 4),
            "fraction_above_threshold": round(frac_above, 4),
            "n_prompts": len(deltas),
            "threshold_median": self.THRESHOLD_MEDIAN,
            "threshold_fraction": self.THRESHOLD_FRACTION,
            "threshold_per_prompt": self.THRESHOLD_PER_PROMPT,
            "details": f"median_delta={med_delta:.4f}, frac_above={frac_above:.4f}, n={len(deltas)}"
        }


class WisdomProblemAlignmentEvaluator:
    """Component 3: Measures semantic relevance of wisdom to each problem."""

    MIN_ALIGNMENT_Z = -1.5

    def evaluate(self, cid: str, loader: DataLoader) -> Dict[str, Any]:
        wisdom_text = loader._get_wisdom_text(cid)
        records = loader.per_pid_records(cid)

        # FIX: use canonical API field name `problem`
        raw_sims = []
        for rec in records:
            problem = rec.get("problem", "")
            if problem and wisdom_text:
                raw_sims.append(text_cosine_similarity(wisdom_text, problem))

        if not raw_sims:
            return {
                "component": "wisdom_problem_alignment",
                "passed": False,
                "min_alignment_z": -999,
                "mean_alignment": 0.0,
                "n_prompts": 0,
                "threshold_min_z": self.MIN_ALIGNMENT_Z,
                "details": "no valid problem/wisdom pairs"
            }

        all_wisdom = loader.get_all_wisdom_texts()
        baseline_sims: List[float] = []
        for other_cid, other_text in all_wisdom.items():
            if not other_text:
                continue
            for rec in records:
                # FIX: use canonical API field name `problem`
                problem = rec.get("problem", "")
                if problem:
                    baseline_sims.append(text_cosine_similarity(other_text, problem))

        mu = mean(baseline_sims) if baseline_sims else 0.0
        sigma = std(baseline_sims) if baseline_sims else 0.0
        if sigma < 1e-12:
            z_scores = [0.0 for _ in raw_sims]
        else:
            z_scores = [(s - mu) / sigma for s in raw_sims]

        min_z = min(z_scores) if z_scores else -999
        mean_alignment = mean(raw_sims) if raw_sims else 0.0
        passed = min_z >= self.MIN_ALIGNMENT_Z

        return {
            "component": "wisdom_problem_alignment",
            "passed": passed,
            "min_alignment_z": round(min_z, 4),
            "mean_alignment": round(mean_alignment, 4),
            "n_prompts": len(raw_sims),
            "threshold_min_z": self.MIN_ALIGNMENT_Z,
            "details": f"min_z={min_z:.4f}, mean_alignment={mean_alignment:.4f}, n={len(raw_sims)}"
        }


class AntipatternAvoidanceEvaluator:
    """Component 4: Detects anti-patterns explicitly recorded in metadata."""

    MAX_FLAGGED_FRACTION = 0.34

    def evaluate(self, cid: str, loader: DataLoader) -> Dict[str, Any]:
        records = loader.per_pid_records(cid)

        flagged = 0
        total = 0
        examples = []

        for rec in records:
            anti = rec.get("ext_anti_patterns", rec.get("base_anti_patterns", ""))
            if anti is None:
                anti = ""
            anti_text = anti if isinstance(anti, str) else json.dumps(anti, ensure_ascii=False)
            total += 1

            bad = False
            if anti_text.strip():
                low = anti_text.lower()
                if not re.fullmatch(r"(none|no|n/?a|nil|nothing|[]{}\",?\s*)+", low.strip()):
                    bad = True

            if bad:
                flagged += 1
                if len(examples) < 5:
                    examples.append(anti_text[:200])

        frac = (flagged / total) if total else 1.0
        passed = frac <= self.MAX_FLAGGED_FRACTION

        return {
            "component": "antipattern_avoidance",
            "passed": passed,
            "flagged_fraction": round(frac, 4),
            "n_prompts": total,
            "n_flagged": flagged,
            "threshold_max_fraction": self.MAX_FLAGGED_FRACTION,
            "examples": examples,
            "details": f"flagged={flagged}/{total}, frac={frac:.4f}"
        }


# ---------------------------------------------------------------------------
# Gate runner
# ---------------------------------------------------------------------------

class OrthogonalDecompositionGate:
    def __init__(self):
        self.components = [
            ReframeDepthEvaluator(),
            SubstantiveContentDeltaEvaluator(),
            WisdomProblemAlignmentEvaluator(),
            AntipatternAvoidanceEvaluator(),
        ]

    def evaluate_candidate(self, cid: str, loader: DataLoader) -> Dict[str, Any]:
        component_results = []
        for comp in self.components:
            component_results.append(comp.evaluate(cid, loader))

        passed_components = [r["component"] for r in component_results if r.get("passed", False)]
        failed_components = [r["component"] for r in component_results if not r.get("passed", False)]

        overall_pass = len(failed_components) == 0

        return {
            "candidate_id": cid,
            "overall_pass": overall_pass,
            "n_components": len(component_results),
            "n_passed": len(passed_components),
            "n_failed": len(failed_components),
            "passed_components": passed_components,
            "failed_components": failed_components,
            "components": component_results,
        }

    def evaluate_all(self, loader: DataLoader, verbose: bool = False) -> Dict[str, Any]:
        cids = loader.list_candidates()
        results = []

        for i, cid in enumerate(cids):
            if verbose:
                print(f"[{i+1}/{len(cids)}] evaluating {cid}", file=sys.stderr)
            results.append(self.evaluate_candidate(cid, loader))

        pass_rate = (sum(1 for r in results if r["overall_pass"]) / len(results)) if results else 0.0

        return {
            "summary": {
                "n_candidates": len(results),
                "n_passed": sum(1 for r in results if r["overall_pass"]),
                "n_failed": sum(1 for r in results if not r["overall_pass"]),
                "pass_rate": round(pass_rate, 4),
            },
            "results": results,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    loader = DataLoader(args.data_dir)
    gate = OrthogonalDecompositionGate()
    output = gate.evaluate_all(loader, verbose=args.verbose)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    else:
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()