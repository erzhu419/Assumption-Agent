"""Audit circular contig family prevalence and depth across an isolate cohort."""

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
        chromosome_depths = [c['mean_depth'] for c in contigs
                             if not c['circular'] and c['family_tag'].startswith('CHR_SEG_')]
        if not chromosome_depths or (baseline := median(chromosome_depths)) <= 0:
            raise ValueError(f'No positive chromosome depth baseline in {path.name}')
        isolate_copies = defaultdict(list)
        for contig in contigs:
            if contig['circular'] and len(contig['sequence']) >= 2400:
                copies = contig['mean_depth'] / baseline
                if 3.2 <= copies <= 7.5:
                    isolate_copies[contig['family_tag']].append(copies)
        for tag, values in isolate_copies.items():
            cohort_copies[tag].append(median(values))

    qualifying = [(tag, values) for tag, values in cohort_copies.items()
                  if 10 * len(values) >= 9 * len(paths)]
    if len(qualifying) != 1:
        raise ValueError(f'Expected one qualifying family, found {len(qualifying)}')
    tag, values = qualifying[0]
    return {
        'shared_plasmid_family_tag': tag,
        'median_plasmid_copy_number': float(median(values)),
        'amino_acid_substitution': None,
    }
