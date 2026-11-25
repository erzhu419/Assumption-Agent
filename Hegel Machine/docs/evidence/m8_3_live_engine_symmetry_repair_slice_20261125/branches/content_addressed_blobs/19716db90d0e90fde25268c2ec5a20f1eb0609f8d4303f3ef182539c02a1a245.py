"""Audit circular plasmid family prevalence and depth-normalized copy number."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    paths = sorted(Path(cohort_dir).glob('isolate_*.json'))
    if not paths:
        raise ValueError('No isolate bundles found')

    cohort_copies = defaultdict(list)
    for path in paths:
        contigs = json.loads(path.read_text(encoding='utf-8'))['contigs']
        chromosome_depth = median(c['mean_depth'] for c in contigs if not c['circular'])
        isolate_copies = defaultdict(list)
        for contig in contigs:
            if contig['circular'] and len(contig['sequence']) >= 2400:
                copies = contig['mean_depth'] / chromosome_depth
                if 3.2 <= copies <= 7.5:
                    isolate_copies[contig['family_tag']].append(copies)
        for family, copies in isolate_copies.items():
            cohort_copies[family].append(median(copies))

    shared = [family for family, copies in cohort_copies.items()
              if 10 * len(copies) >= 9 * len(paths)]
    if len(shared) != 1:
        raise ValueError(f'Expected one family with at least 90% support; found {len(shared)}')
    family = shared[0]
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': float(median(cohort_copies[family])),
    }
