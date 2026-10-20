"""Convert the finite historical plan into shared blocks; restore any saved unit."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from hegel_machine.cognition.effective_history import HistoryStore, compact_unit, restore_unit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--plan', required=True)
    prepare.add_argument('--store', required=True)
    prepare.add_argument('--receipt', required=True)
    prepare.add_argument('--work-root', required=True)
    prepare.add_argument('--only', action='append')
    restore = commands.add_parser('restore')
    restore.add_argument('--store', required=True)
    restore.add_argument('--unit', required=True)
    restore.add_argument('--destination', required=True)
    restore.add_argument('--member')
    args = parser.parse_args()
    if args.command == 'restore':
        store = HistoryStore(args.store, readonly=True)
        try:
            count = restore_unit(store, args.unit, args.destination, args.member)
            print(json.dumps({'unit': args.unit, 'restored_members': count}))
        finally:
            store.close()
        return
    plan = json.loads(Path(args.plan).read_text())
    Path(args.work_root).mkdir(parents=True, exist_ok=True)
    store = HistoryStore(args.store)
    log = Path(args.receipt)
    done = set()
    if log.exists():
        done = {json.loads(line)['unit'] for line in log.read_text().splitlines() if line.strip()}
    try:
        with log.open('a') as output:
            for item in plan['units']:
                if args.only and item['unit'] not in args.only:
                    continue
                if item['unit'] in done:
                    continue
                receipt = compact_unit(store, item, args.work_root,
                    progress=lambda event: print(json.dumps(event), flush=True))
                output.write(json.dumps(receipt, ensure_ascii=False) + '\n')
                output.flush()
                print(json.dumps(receipt, ensure_ascii=False), flush=True)
    finally:
        store.close()


if __name__ == '__main__':
    main()
