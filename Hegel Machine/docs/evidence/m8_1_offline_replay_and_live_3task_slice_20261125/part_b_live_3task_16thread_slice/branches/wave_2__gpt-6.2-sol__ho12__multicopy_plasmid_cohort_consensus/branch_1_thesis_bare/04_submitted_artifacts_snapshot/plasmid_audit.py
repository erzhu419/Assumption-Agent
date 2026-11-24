"""Identify a shared multicopy plasmid from isolate assembly and depth data."""

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median


_CORE_LENGTH = 36
_COMPLEMENT = str.maketrans('ACGT', 'TGCA')
_CODE = dict(zip(
    (''.join(bases) for bases in product('TCAG', repeat=3)),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
))


def _core_queries(reference):
    """Split each in-frame core around the one codon allowed to differ."""
    for start in range(0, len(reference) - _CORE_LENGTH + 1, 3):
        core = reference[start:start + _CORE_LENGTH]
        for offset in range(0, _CORE_LENGTH, 3):
            wild_aa = _CODE[core[offset:offset + 3]]
            position = (start + offset) // 3 + 1
            yield start, position, wild_aa, core[:offset], core[offset + 3:]


def _core_hits(sequence, queries):
    """Find exact cores apart from one nonsynonymous codon, on either strand."""
    size = len(sequence)
    hits = set()
    for strand in (sequence, sequence.translate(_COMPLEMENT)[::-1]):
        circular = strand + strand[:_CORE_LENGTH - 1]
        for start, position, wild_aa, left, right in queries:
            # Seed on the longer unchanged stretch, then verify the other flank.
            seed, shift = (left, 0) if len(left) >= len(right) else (right, len(left) + 3)
            hit = circular.find(seed, shift)
            while hit != -1 and hit - shift < size:
                pos = hit - shift
                codon_start = pos + len(left)
                if (circular.startswith(left, pos)
                        and circular.startswith(right, codon_start + 3)):
                    codon = circular[codon_start:codon_start + 3]
                    variant_aa = _CODE.get(codon)
                    if variant_aa is not None and variant_aa != wild_aa:
                        hits.add((start, codon, f'{wild_aa}{position}{variant_aa}'))
                hit = circular.find(seed, hit + 1)
    return hits


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    """Return the family, carrier median copy number, and core substitution."""
    directory = Path(cohort_dir)
    with (directory / 'reference_panel.json').open(encoding='utf-8') as handle:
        reference = json.load(handle)['reference_wildtype_cds'].upper()
    queries = tuple(_core_queries(reference))
    paths = sorted(directory.glob('isolate_*.json'))
    # An unchanged core offset and variant codon identify the exact same motif.
    # Each isolate votes once per family and motif, even if multiple contigs match.
    support = defaultdict(list)
    for path in paths:
        with path.open(encoding='utf-8') as handle:
            contigs = json.load(handle)['contigs']
        chromosomal_depth = median(
            contig['mean_depth'] for contig in contigs
            if contig['family_tag'].startswith('CHR_')
        )
        isolate_hits = defaultdict(list)
        for contig in contigs:
            sequence = contig['sequence']
            if not contig['circular'] or len(sequence) < 2400:
                continue
            copies = contig['mean_depth'] / chromosomal_depth
            if 3.2 <= copies <= 7.5:
                for motif in _core_hits(sequence.upper(), queries):
                    isolate_hits[(contig['family_tag'], *motif)].append(copies)
        for key, copies in isolate_hits.items():
            support[key].append(median(copies))

    eligible = [key for key, copies in support.items() if 10 * len(copies) >= 9 * len(paths)]
    if not eligible:
        raise ValueError('No plasmid family and variant core are shared by at least 90% of isolates')
    winner = min(eligible, key=lambda key: (-len(support[key]), key))
    family, _, _, substitution = winner
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': float(median(support[winner])),
        'amino_acid_substitution': substitution,
    }
