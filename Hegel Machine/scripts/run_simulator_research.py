"""Run or resume the bounded executable observation-simulator research example."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hegel_machine.cognition.simulator_research import run_simulator_research


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    previous = json.loads(args.resume.read_text()) if args.resume else None
    result = run_simulator_research(state=previous, max_steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, ensure_ascii=False, indent=2)
    print(json.dumps({key: result[key] for key in ("status", "phase", "costs", "selected_probe_changed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
