"""Audit existing circular plasmid contigs for cohort and coding consensus."""

import json
from collections import defaultdict
from itertools import product
from pathlib import Path
from statistics import median


# Standard genetic code, in T/C/A/G order at each codon position.
_CODE = dict(zip(map(''.join, product('TCAG', repeat=3)),
                 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRR'
                 'IIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'))
_COMPLEMENT = str.maketrans('ACGT', 'TGCA')
_CORE = 36
_HALF = _CORE // 2


def _variant_cores(sequence, reference_cores):
    """Yield (reference offset, observed core, AA change) for both strands."""
    for strand in (sequence, sequence.translate(_COMPLEMENT)[::-1]):
        extended = strand + strand[:_CORE - 1]
        for ref_start, wild in reference_cores:
            # One codon differs, so at least one codon-aligned half is exact.
            for shift in (0, _HALF):
                anchor = wild[shift:shift + _HALF]
                hit = extended.find(anchor, shift)
                while hit != -1 and hit - shift < len(sequence):
                    start = hit - shift
                    observed = extended[start:start + _CORE]
                    changed = [i for i in range(0, _CORE, 3)
                               if wild[i:i + 3] != observed[i:i + 3]]
                    if len(changed) == 1:
                        i = changed[0]
                        before = _CODE.get(wild[i:i + 3])
                        after = _CODE.get(observed[i:i + 3])
                        if before and after and before != after:
                            position = (ref_start + i) // 3 + 1
                            yield ref_start, observed, f'{before}{position}{after}'
                    hit = extended.find(anchor, hit + 1)


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    """Return the qualifying family, median copy number, and consensus change."""
    directory = Path(cohort_dir)
    panel = json.loads((directory / 'reference_panel.json').read_text())
    reference = panel['reference_wildtype_cds'].upper()
    reference_cores = [(i, reference[i:i + _CORE])
                       for i in range(0, len(reference) - _CORE + 1, 3)]
    bundles = sorted(directory.glob('isolate_*.json'))
    # A motif's support is counted in distinct isolate bundles, not hits.
    evidence = defaultdict(lambda: defaultdict(dict))
    for path in bundles:
        contigs = json.loads(path.read_text())['contigs']
        chromosomal_depth = median(c['mean_depth'] for c in contigs
                                   if c['family_tag'].startswith('CHR_SEG_'))
        for index, contig in enumerate(contigs):
            if not contig['circular'] or len(contig['sequence']) < 2400:
                continue
            copy = contig['mean_depth'] / chromosomal_depth
            if not 3.2 <= copy <= 7.5:
                continue
            for offset, core, change in _variant_cores(
                    contig['sequence'].upper(), reference_cores):
                key = (contig['family_tag'], change, offset, core)
                evidence[key][path][index] = copy

    # Several overlapping cores may describe the same change. Keep the most
    # widely supported core(s) and count each supporting contig only once.
    candidates = {}
    for (family, change, _, _), isolates in evidence.items():
        support = len(isolates)
        if 10 * support < 9 * len(bundles):
            continue
        key = family, change
        best, supporting_contigs = candidates.get(key, (0, {}))
        if support < best:
            continue
        if support > best:
            supporting_contigs = {}
        for path, copies in isolates.items():
            supporting_contigs.setdefault(path, {}).update(copies)
        candidates[key] = support, supporting_contigs

    if not candidates:
        raise ValueError('No plasmid variant core meets the 90% cohort threshold')
    best_support = max(support for support, _ in candidates.values())
    winners = [(key, copies) for key, (support, copies) in candidates.items()
               if support == best_support]
    if len(winners) != 1:
        raise ValueError('Multiple plasmid variants share the highest cohort support')
    (family, change), copies = winners[0]
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': float(median(median(values.values())
                                                 for values in copies.values())),
        'amino_acid_substitution': change,
    }
