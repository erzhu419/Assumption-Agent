"""Summarize circular contig families by cohort prevalence and copy number."""

import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import median


def audit_copy_number_cohort(cohort_dir: str) -> dict:
    """Return metadata summaries for families present in at least 90% of bundles."""
    paths = [p for p in Path(cohort_dir).glob('*.json')
             if p.is_file() and p.name != 'reference_panel.json']
    if not paths:
        raise ValueError('No isolate bundles found')

    families = defaultdict(list)
    for path in paths:
        contigs = json.loads(path.read_text(encoding='utf-8'))['contigs']
        linear_depths = [float(c['mean_depth']) for c in contigs if not c['circular']]
        if not linear_depths or not all(map(isfinite, linear_depths)):
            raise ValueError(f'No finite linear-contig depth baseline in {path.name}')
        baseline = median(linear_depths)
        if not isfinite(baseline) or baseline <= 0:
            raise ValueError(f'No positive finite depth baseline in {path.name}')

        per_isolate = defaultdict(list)
        for contig in contigs:
            if not contig['circular'] or len(contig['sequence']) < 2400:
                continue
            ratio = float(contig['mean_depth']) / baseline
            if 3.2 <= ratio <= 7.5:
                per_isolate[contig['family_tag']].append(ratio)

        for tag, ratios in per_isolate.items():
            families[tag].append(median(ratios))

    n = len(paths)
    return {
        'isolate_count': n,
        'candidate_families': {
            tag: {
                'supporting_isolates': len(ratios),
                'median_copy_number': float(median(ratios)),
            }
            for tag, ratios in families.items()
            if 10 * len(ratios) >= 9 * n
        },
    }
