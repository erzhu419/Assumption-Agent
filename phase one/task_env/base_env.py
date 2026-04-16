"""
Phase 1: Task Environment.
Loads Phase 0 benchmark problems and executes them with strategies via LLM.
"""

import json
import random
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))
from llm_client import create_client, parse_json_from_llm


@dataclass
class TaskObservation:
    problem_id: str
    domain: str
    description: str
    difficulty: str
    reference_strategies: List[str]  # ground truth (hidden from dispatcher)


@dataclass
class ExecutionOutcome:
    success: bool
    partial_success: bool = False
    evaluation_score: float = 0.0
    strategy_used: str = ""
    consistency_score: float = 1.0
    steps_taken: int = 0
    failure_reason: str = ""
    llm_output: str = ""
    is_simulated: bool = False


class TaskEnvironment:
    """
    Loads Phase 0 problems and provides an execute-with-strategy interface.

    For Phase 1 training, we evaluate strategy selection by checking if
    the chosen strategy matches the annotated optimal strategies (proxy reward),
    since actually executing complex tasks end-to-end is too expensive.
    """

    def __init__(self, problems_dir: Path = None, annotations_dir: Path = None,
                 strategy_kb: Dict = None, use_llm_execution: bool = False):
        import _config as settings
        self.problems_dir = problems_dir or settings.PROBLEMS_DIR
        self.annotations_dir = annotations_dir or settings.ANNOTATIONS_DIR
        self.strategy_kb = strategy_kb or {}
        self.use_llm_execution = use_llm_execution

        self._problems: List[Dict] = []
        self._annotations: Dict[str, Dict] = {}
        self._current_task: Optional[Dict] = None
        self._client = None

        self._load_data()

    def _load_data(self):
        """Load all problems and annotations."""
        # Problems
        for f in sorted(self.problems_dir.glob("*.json")):
            if "error" in f.name:
                continue
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._problems.extend(data)

        # Annotations
        for f in sorted(self.annotations_dir.glob("*.json")):
            ann = json.loads(f.read_text(encoding="utf-8"))
            self._annotations[ann["problem_id"]] = ann

        # Separate into train / eval
        random.seed(42)
        random.shuffle(self._problems)
        n = len(self._problems)
        self.train_problems = self._problems[:int(n * 0.7)]
        self.val_problems = self._problems[int(n * 0.7):int(n * 0.85)]
        self.test_problems = self._problems[int(n * 0.85):]

    @property
    def client(self):
        if self._client is None:
            self._client = create_client()
        return self._client

    def sample_task(self, split: str = "train") -> TaskObservation:
        """Sample a random task from the specified split."""
        if split == "train":
            pool = self.train_problems
        elif split == "val":
            pool = self.val_problems
        else:
            pool = self.test_problems

        task = random.choice(pool)
        self._current_task = task

        ref = task.get("reference_answer", {})
        return TaskObservation(
            problem_id=task["problem_id"],
            domain=task.get("domain", "unknown"),
            description=task["description"],
            difficulty=task.get("difficulty", "medium"),
            reference_strategies=ref.get("optimal_strategies", []) +
                                 ref.get("acceptable_strategies", []),
        )

    def evaluate_strategy_selection(
        self,
        problem_id: str,
        selected_strategy: str,
        confidence: float = 0.5,
    ) -> ExecutionOutcome:
        """
        Evaluate whether the selected strategy is correct.

        Two modes:
        1. Annotation-based (fast, default): Check against Phase 0 annotations
        2. LLM-based (expensive, optional): Actually execute via LLM

        For RL training, mode 1 is sufficient — the "real environment" is
        the annotation ground truth, and the "world model" is the statistical
        prediction of whether annotation will match.
        """
        ann = self._annotations.get(problem_id)
        task = self._current_task

        if ann is None:
            # No annotation — treat as unknown
            return ExecutionOutcome(
                success=False,
                evaluation_score=0.0,
                strategy_used=selected_strategy,
                failure_reason="no_annotation",
            )

        ref = ann.get("reference_answer", {})
        optimal = set(ref.get("optimal_strategies", []))
        acceptable = set(ref.get("acceptable_strategies", []))

        # Check against reference
        if selected_strategy in optimal:
            success = True
            eval_score = 1.0
        elif selected_strategy in acceptable:
            success = True
            eval_score = 0.7
        else:
            success = False
            eval_score = 0.0

        # Check annotator agreement: does majority agree with this choice?
        annotator_votes = []
        for a in ann.get("annotations", []):
            top1 = a["selected_strategies"][0] if a.get("selected_strategies") else None
            if top1:
                annotator_votes.append(top1)

        if annotator_votes:
            agreement = sum(1 for v in annotator_votes if v == selected_strategy) / len(annotator_votes)
            # Bonus for annotator agreement
            if agreement > 0.5:
                eval_score = min(1.0, eval_score + 0.2)
            consistency = agreement
        else:
            consistency = 0.5

        return ExecutionOutcome(
            success=success,
            partial_success=(not success and eval_score > 0),
            evaluation_score=eval_score,
            strategy_used=selected_strategy,
            consistency_score=consistency,
            steps_taken=1,
        )

    def get_all_problems(self, split: str = "train") -> List[Dict]:
        if split == "train":
            return self.train_problems
        elif split == "val":
            return self.val_problems
        return self.test_problems

    def stats(self) -> Dict:
        return {
            "total_problems": len(self._problems),
            "train": len(self.train_problems),
            "val": len(self.val_problems),
            "test": len(self.test_problems),
            "annotated": len(self._annotations),
        }
