"""Audit an outbreak cohort for a shared multicopy plasmid allele."""

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median


_CORE_LENGTH = 36
_HALF_CORE = _CORE_LENGTH // 2
_COMPLEMENT = str.maketrans('ACGT', 'TGCA')
_CODE = dict(zip(
    map(''.join, product('TCAG', repeat=3)),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
))


def _read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def _core_hits(sequence, reference_cores):
    """Yield reference offset, exact mutant core, and amino-acid change."""
    for strand in (sequence, sequence.translate(_COMPLEMENT)[::-1]):
        extended = strand + strand[:_CORE_LENGTH - 1]
        for offset, wildtype in reference_cores:
            # With only one changed codon, at least one codon-aligned half is exact.
            starts = set()
            for shift in (0, _HALF_CORE):
                anchor = wildtype[shift:shift + _HALF_CORE]
                pos = extended.find(anchor, shift)
                while pos != -1 and pos - shift < len(strand):
                    starts.add(pos - shift)
                    pos = extended.find(anchor, pos + 1)
            for start in starts:
                core = extended[start:start + _CORE_LENGTH]
                changed = [i for i in range(0, _CORE_LENGTH, 3)
                           if wildtype[i:i + 3] != core[i:i + 3]]
                if len(changed) != 1:
                    continue
                i = changed[0]
                old = _CODE.get(wildtype[i:i + 3])
                new = _CODE.get(core[i:i + 3])
                if old and new and old != new:
                    position = (offset + i) // 3 + 1
                    yield offset, core, f'{old}{position}{new}'


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    """Return the plasmid family, median copy number, and shared substitution."""
    directory = Path(cohort_dir)
    reference = _read_json(directory / 'reference_panel.json')[
        'reference_wildtype_cds'].upper()
    reference_cores = [(i, reference[i:i + _CORE_LENGTH])
                       for i in range(0, len(reference) - _CORE_LENGTH + 1, 3)]
    isolates = sorted(p for p in directory.glob('isolate_*.json')
                      if p.stem.removeprefix('isolate_').isdigit())
    if not isolates:
        raise ValueError('No isolate bundles found')

    support = defaultdict(list)
    for path in isolates:
        contigs = _read_json(path)['contigs']
        chromosome_depths = [c['mean_depth'] for c in contigs
                             if c['family_tag'].startswith('CHR_')]
        if not chromosome_depths or (baseline := median(chromosome_depths)) <= 0:
            raise ValueError(f'No positive median chromosomal depth in {path.name}')

        isolate_hits = defaultdict(list)
        for contig in contigs:
            if not contig['circular'] or len(contig['sequence']) < 2400:
                continue
            copy = contig['mean_depth'] / baseline
            if not 3.2 <= copy <= 7.5:
                continue
            for offset, core, substitution in set(_core_hits(
                    contig['sequence'].upper(), reference_cores)):
                key = contig['family_tag'], offset, core, substitution
                isolate_hits[key].append(copy)
        for key, copies in isolate_hits.items():
            support[key].append(median(copies))

    required = (9 * len(isolates) + 9) // 10
    best_support = max(map(len, support.values()), default=0)
    if best_support < required:
        raise ValueError('No plasmid family and mutant core shared by 90% of isolates')
    # Overlapping reference windows can report the same substitution and copies.
    calls = {(family, substitution, float(median(copies)))
             for (family, _, _, substitution), copies in support.items()
             if len(copies) == best_support}
    if len(calls) != 1:
        raise ValueError('Ambiguous shared plasmid consensus')
    family, substitution, copy = calls.pop()
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': copy,
        'amino_acid_substitution': substitution,
    }
