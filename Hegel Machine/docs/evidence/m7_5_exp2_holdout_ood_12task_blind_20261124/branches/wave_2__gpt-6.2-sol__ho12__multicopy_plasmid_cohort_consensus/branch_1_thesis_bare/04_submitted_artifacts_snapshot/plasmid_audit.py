"""Cohort-level audit of multicopy plasmids in assembled isolate bundles."""

import json
from collections import defaultdict
from pathlib import Path
from statistics import median


_CORE_LENGTH = 36
_COMPLEMENT = str.maketrans('ACGT', 'TGCA')
_GENETIC_CODE = dict(zip(
    (a + b + c for a in 'TCAG' for b in 'TCAG' for c in 'TCAG'),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
))


def _load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _motif_hits(sequence, reference_windows):
    """Yield (CDS offset, observed core, amino acid substitution) evidence."""
    half = _CORE_LENGTH // 2  # Both halves contain whole codons.
    n = len(sequence)
    for strand in (sequence, sequence.translate(_COMPLEMENT)[::-1]):
        circular = strand + strand[:_CORE_LENGTH - 1]
        for offset, wildtype in reference_windows:
            starts = set()
            # With only one changed codon, at least one half is an exact anchor.
            for shift in (0, half):
                anchor = wildtype[shift:shift + half]
                hit = circular.find(anchor, shift, n + shift + half - 1)
                while hit != -1:
                    starts.add(hit - shift)
                    hit = circular.find(anchor, hit + 1, n + shift + half - 1)
            for start in starts:
                core = circular[start:start + _CORE_LENGTH]
                changed = [i for i in range(0, _CORE_LENGTH, 3)
                           if wildtype[i:i + 3] != core[i:i + 3]]
                if len(changed) != 1:
                    continue
                i = changed[0]
                before = _GENETIC_CODE[wildtype[i:i + 3]]
                after = _GENETIC_CODE.get(core[i:i + 3])
                if after is not None and after != before:
                    yield offset, core, f'{before}{(offset + i) // 3 + 1}{after}'


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    """Return the shared plasmid family, carrier median copy number, and variant."""
    root = Path(cohort_dir)
    files = sorted(root.glob('isolate_*.json'))
    reference = _load(root / 'reference_panel.json')['reference_wildtype_cds'].upper()
    windows = [(i, reference[i:i + _CORE_LENGTH])
               for i in range(0, len(reference) - _CORE_LENGTH + 1, 3)]

    carriers = defaultdict(dict)
    for path in files:
        contigs = _load(path)['contigs']
        chromosomal_depth = median(c['mean_depth'] for c in contigs
                                   if c['family_tag'].startswith('CHR_'))
        for index, contig in enumerate(contigs):
            if not contig['circular'] or len(contig['sequence']) < 2400:
                continue
            copy = contig['mean_depth'] / chromosomal_depth
            if not 3.2 <= copy <= 7.5:
                continue
            for offset, core, substitution in _motif_hits(contig['sequence'].upper(), windows):
                key = contig['family_tag'], offset, core, substitution
                carriers[key].setdefault(path, {})[index] = copy

    winners = []
    best_support = 0
    for key, isolates in carriers.items():
        support = len(isolates)
        if 10 * support < 9 * len(files) or support < best_support:
            continue
        if support > best_support:
            winners.clear()
            best_support = support
        winners.append(key)

    calls = {(key[0], key[3]) for key in winners}
    if len(calls) != 1:
        raise ValueError('Expected one cohort-wide plasmid family and substitution')
    family, substitution = calls.pop()
    # Overlapping reference windows can describe the same call. Count each
    # contig once, and give each carrier isolate one vote in the copy median.
    copies = defaultdict(dict)
    for key in winners:
        for isolate, contig_copies in carriers[key].items():
            copies[isolate].update(contig_copies)
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': float(median(median(c.values()) for c in copies.values())),
        'amino_acid_substitution': substitution,
    }
