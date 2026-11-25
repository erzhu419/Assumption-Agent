"""Audit cohort prevalence and normalized copy number from contig metadata."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median


def audit_copy_number_cohort(cohort_dir: str) -> dict:
    """Report circular contig families meeting the cohort copy-number criteria."""
    files = sorted(Path(cohort_dir).glob('isolate_*.json'))
    if not files:
        raise ValueError('No isolate bundles found')

    families = defaultdict(list)
    for path in files:
        contigs = json.loads(path.read_text())['contigs']
        linear_depths = [c['mean_depth'] for c in contigs if not c['circular']]
        if not linear_depths or (baseline := median(linear_depths)) <= 0:
            raise ValueError(f'No positive linear-contig depth baseline in {path.name}')
        per_isolate = defaultdict(list)
        for contig in contigs:
            if contig['circular'] and len(contig['sequence']) >= 2400:
                ratio = contig['mean_depth'] / baseline
                if 3.2 <= ratio <= 7.5:
                    per_isolate[contig['family_tag']].append(ratio)
        for tag, ratios in per_isolate.items():
            families[tag].append(median(ratios))

    return {
        'isolate_count': len(files),
        'candidate_families': {
            tag: {'supporting_isolates': len(ratios), 'median_copy_number': median(ratios)}
            for tag, ratios in sorted(families.items())
            if 10 * len(ratios) >= 9 * len(files)
        },
    }
