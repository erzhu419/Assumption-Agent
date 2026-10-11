"""Include within-task revisits retained in complete independent task files."""
import argparse
import gzip
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--group', required=True)
args = parser.parse_args()
output = Path(__file__).resolve().parent / args.group
protocol = json.loads((output / 'protocol.json').read_text())
summary_path = output / 'summary.json'
summary = json.loads(summary_path.read_text())
for arm, settings in protocol['arms'].items():
    if settings['knowledge'] != 'independent':
        continue
    revisits = 0
    for root in protocol['roots']:
        for path in (output / 'runs' / arm / root).glob('task*.json.gz'):
            with gzip.open(path, 'rt') as stream:
                revisits += len(json.load(stream)['revisits'])
    summary['totals'][arm]['revisits'] = revisits
summary['independent_revisit_accounting'] = 'Within-task revisits are counted from taskNN.json.gz; the combined independent checkpoint resets its top-level revisit list. Research and terminal fees/CPU already include all within-task work.'
summary_path.write_text(json.dumps(summary, indent=2) + '\n')
print({arm: total['revisits'] for arm, total in summary['totals'].items()})
