"""Cohort-wide plasmid family and copy-number QC using contig metadata.

Sequence content is not analyzed; the sequence field supplies length only.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    """Return the family tag and median copy number of the unique shared candidate."""
    paths = sorted(Path(cohort_dir).glob('isolate_*.json'))
    if len(paths) < 12:
        raise ValueError('Expected at least 12 isolate bundles')

    by_family = defaultdict(list)
    for path in paths:
        with path.open(encoding='utf-8') as stream:
            contigs = json.load(stream)['contigs']

        # In the public bundle schema the linear contigs are chromosome segments.
        chromosome_depths = [c['mean_depth'] for c in contigs if not c['circular']]
        if not chromosome_depths or (baseline := median(chromosome_depths)) <= 0:
            raise ValueError(f'{path.name}: missing positive chromosome depth baseline')

        local = defaultdict(list)
        for contig in contigs:
            if contig['circular'] and len(contig['sequence']) >= 2400:
                ratio = contig['mean_depth'] / baseline
                if 3.2 <= ratio <= 7.5:
                    local[contig['family_tag']].append(ratio)
        for family, ratios in local.items():
            by_family[family].append(median(ratios))

    shared = [family for family, ratios in by_family.items()
              if 10 * len(ratios) >= 9 * len(paths)]
    if len(shared) != 1:
        raise ValueError(f'Expected one family at 90% prevalence; found {len(shared)}')
    family = shared[0]
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': float(median(by_family[family])),
    }
