"""Audit cohort-wide plasmid prevalence, copy number, and missense variation."""

import json
from collections import defaultdict
from math import isfinite
from pathlib import Path
from statistics import median


_CORE_LENGTH = 36
_SEED_OFFSETS = (0, 12, 24)
_BASES = 'TCAG'
_AA = dict(zip(
    (a + b + c for a in _BASES for b in _BASES for c in _BASES),
    'FFLLSSSSYY**CC*W' 'LLLLPPPPHHQQRRRR'
    'IIIMTTTTNNKKSSRR' 'VVVVAAAADDEEGGGG',
))
_COMPLEMENT = str.maketrans('ACGT', 'TGCA')


def _dna(value):
    return ''.join(value.split()).upper()


def _reference_index(cds):
    windows = {}
    seeds = {offset: defaultdict(list) for offset in _SEED_OFFSETS}
    for start in range(len(cds) - _CORE_LENGTH + 1):
        window = cds[start:start + _CORE_LENGTH]
        if any(base not in _BASES for base in window):
            continue
        windows[start] = window
        for offset, index in seeds.items():
            index[window[offset:offset + 12]].append(start)
    return windows, seeds


def _core_hits(sequence, windows, seeds):
    """Yield (one-based codon, wild-type AA, observed AA, oriented core)."""
    reverse = sequence.translate(_COMPLEMENT)[::-1]
    for strand in (sequence, reverse):
        # Include windows that cross the arbitrary origin of a circular contig.
        extended = strand + strand[:_CORE_LENGTH - 1]
        for start in range(len(strand)):
            window = extended[start:start + _CORE_LENGTH]
            candidates = set()
            # A change within one codon can disrupt at most two of three seeds.
            for offset, index in seeds.items():
                candidates.update(index.get(window[offset:offset + 12], ()))
            for ref_start in candidates:
                reference = windows[ref_start]
                first_diff = next((i for i, (a, b) in enumerate(zip(reference, window))
                                   if a != b), None)
                if first_diff is None:
                    continue
                # Frame comes from the CDS, not from the contig or window start.
                codon_start = ((ref_start + first_diff) // 3) * 3
                left = codon_start - ref_start
                right = left + 3
                if left < 0 or right > _CORE_LENGTH:
                    continue
                if reference[:left] != window[:left] or reference[right:] != window[right:]:
                    continue
                old = _AA[reference[left:right]]
                new = _AA.get(window[left:right])
                if new is not None and old != new and '*' not in (old, new):
                    yield codon_start // 3 + 1, old, new, window


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    directory = Path(cohort_dir)
    panel_path = directory / 'reference_panel.json'
    panel = json.loads(panel_path.read_text(encoding='utf-8'))
    cds = _dna(panel['reference_wildtype_cds'])
    windows, seeds = _reference_index(cds)
    if not windows:
        raise ValueError('Reference CDS has no unambiguous 36 bp window')

    paths = [path for path in directory.glob('*.json') if path != panel_path]
    if not paths:
        raise ValueError('No isolate bundles found')
    min_support = (9 * len(paths) + 9) // 10
    families = defaultdict(dict)
    for isolate, path in enumerate(paths):
        contigs = json.loads(path.read_text(encoding='utf-8'))['contigs']
        # In these bundles linear contigs supply the chromosomal depth baseline.
        # Family tags are opaque labels, not a source of chromosome annotation.
        linear_depths = [float(c['mean_depth']) for c in contigs if not c['circular']]
        if not linear_depths:
            raise ValueError(f'No chromosome depth baseline in {path.name}')
        baseline = median(linear_depths)
        if not isfinite(baseline) or baseline <= 0:
            raise ValueError(f'No positive finite chromosome depth baseline in {path.name}')
        for contig in contigs:
            if not contig['circular']:
                continue
            copy = float(contig['mean_depth']) / baseline
            if not 3.2 <= copy <= 7.5:
                continue
            sequence = _dna(contig['sequence'])
            if len(sequence) >= 2400:
                families[contig['family_tag']].setdefault(isolate, []).append((sequence, copy))

    consensus = {}
    for tag, members in families.items():
        if len(members) < min_support:
            continue
        core_carriers = defaultdict(lambda: defaultdict(set))
        for isolate, contigs in members.items():
            for index, (sequence, _) in enumerate(contigs):
                for hit in set(_core_hits(sequence, windows, seeds)):
                    core_carriers[hit][isolate].add(index)
        for (position, old, new, _), carriers in core_carriers.items():
            if len(carriers) < min_support:
                continue
            # Overlapping conserved cores can describe the same substitution.
            confirmed = consensus.setdefault((tag, position, old, new), defaultdict(set))
            for isolate, indices in carriers.items():
                confirmed[isolate].update(indices)

    if len(consensus) != 1:
        raise ValueError(f'Expected one cohort-supported family and substitution, found {len(consensus)}')
    ((tag, position, old, new), carriers), = consensus.items()
    per_isolate = [median(families[tag][isolate][index][1] for index in indices)
                   for isolate, indices in carriers.items()]
    return {
        'shared_plasmid_family_tag': tag,
        'median_plasmid_copy_number': float(median(per_isolate)),
        'amino_acid_substitution': f'{old}{position}{new}',
    }
