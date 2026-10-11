"""Reuse the live continuation runner with the full connection-repair runtime."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SAVED = ROOT / 'artifacts/response_research_connections_20261010'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tests'), str(ROOT)]


def main(variant, directory):
    import hegel_machine.cognition as cognition
    cognition.__path__.insert(0, str(SAVED / 'runtime_before'))
    if variant == 'after':
        cognition.__path__.insert(0, str(SAVED / 'runtime_after'))

    # Import after selecting the package path. This loads the full executor,
    # solver, provider and apparatus used by the existing live runner.
    from scripts import run_response_matched_adoption
    directory = Path(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    sources = {}
    for name, module in list(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if filename is None:
            continue
        source = Path(filename).resolve()
        if source.suffix != '.py' or not source.is_relative_to(ROOT):
            continue
        sources[name] = str(source.relative_to(ROOT))
        # The cognition runtime is already frozen; save only the common Python
        # dependencies actually imported, instead of copying every old test.
        if not source.is_relative_to(SAVED):
            target = OUT / 'common_sources' / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_text(source.read_text())
    (directory.parent / (variant + '_runtime.json')).write_text(json.dumps(sources, indent=2))

    spec = importlib.util.spec_from_file_location('connection_continuation', OUT / 'replay.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.OUT = OUT
    result = runner.run(variant, 'hegel_failure', directory, live=True)
    result['runtime_selection'] = [str(path) for path in cognition.__path__[:2]]
    (directory / 'summary.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('variant', choices=('before', 'after'))
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    main(args.variant, args.directory)
