"""Audit shared plasmid families and conserved coding changes in an isolate cohort."""

import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import median


_MOTIF_LENGTH = 36
_HALF_MOTIF = _MOTIF_LENGTH // 2
_COMPLEMENT = str.maketrans('ACGT', 'TGCA')
_GENETIC_CODE = dict(zip(
    (''.join(bases) for bases in product('TCAG', repeat=3)),
    'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG',
))


def _reference_anchors(reference):
    anchors = defaultdict(list)
    for start in range(0, len(reference) - _MOTIF_LENGTH + 1, 3):
        for shift in (0, _HALF_MOTIF):
            anchor_start = start + shift
            anchor = reference[anchor_start:anchor_start + _HALF_MOTIF]
            anchors[anchor].append((start, shift))
    return anchors


def _motif_hits(sequence, reference, anchors):
    """Yield reference-aligned mutant motifs from both strands of a circle."""
    sequence = sequence.upper()
    for strand in (sequence, sequence.translate(_COMPLEMENT)[::-1]):
        length = len(strand)
        wrapped = strand + strand[:_MOTIF_LENGTH - 1]
        # Every start is considered, covering all three frames on this strand.
        for start in range(length):
            for shift in (0, _HALF_MOTIF):
                pos = start + shift
                anchor = wrapped[pos:pos + _HALF_MOTIF]
                for ref_start, ref_shift in anchors.get(anchor, ()):
                    if shift != ref_shift:
                        continue
                    motif = wrapped[start:start + _MOTIF_LENGTH]
                    wildtype = reference[ref_start:ref_start + _MOTIF_LENGTH]
                    # One changed codon must leave one whole half unchanged.
                    other_half = _HALF_MOTIF - shift
                    changed = [i for i in range(other_half, other_half + _HALF_MOTIF, 3)
                               if motif[i:i + 3] != wildtype[i:i + 3]]
                    if len(changed) != 1:
                        continue
                    i = changed[0]
                    old = _GENETIC_CODE[wildtype[i:i + 3]]
                    new = _GENETIC_CODE.get(motif[i:i + 3])
                    if new is not None and new != old:
                        codon_number = (ref_start + i) // 3 + 1
                        yield ref_start, motif, f'{old}{codon_number}{new}'


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    directory = Path(cohort_dir)
    panel = directory / 'reference_panel.json'
    paths = sorted(path for path in directory.glob('*.json') if path != panel)
    if not paths:
        raise ValueError('No isolate bundles found')

    reference = json.loads(panel.read_text(encoding='utf-8'))['reference_wildtype_cds'].upper()
    if len(reference) < _MOTIF_LENGTH or len(reference) % 3 or set(reference) - set('ACGT'):
        raise ValueError('reference_wildtype_cds must contain at least 12 complete ACGT codons')
    anchors = _reference_anchors(reference)

    cohort_copies = defaultdict(list)
    motif_prevalence = Counter()
    for path in paths:
        contigs = json.loads(path.read_text(encoding='utf-8'))['contigs']
        depths = [c['mean_depth'] for c in contigs if not c['circular']]
        if not depths or (baseline := median(depths)) <= 0:
            raise ValueError(f'No positive chromosome depth baseline in {path.name}')

        isolate_copies = defaultdict(list)
        isolate_hits = set()
        for contig in contigs:
            if not contig['circular'] or len(contig['sequence']) < 2400:
                continue
            copies = contig['mean_depth'] / baseline
            if not 3.2 <= copies <= 7.5:
                continue
            tag = contig['family_tag']
            isolate_copies[tag].append(copies)
            for hit in _motif_hits(contig['sequence'], reference, anchors):
                isolate_hits.add((tag, *hit))

        for tag, copies in isolate_copies.items():
            cohort_copies[tag].append(median(copies))
        motif_prevalence.update(isolate_hits)

    # Overlapping windows that report the same substitution are not ambiguous.
    qualifying = {
        (tag, substitution)
        for (tag, _, _, substitution), count in motif_prevalence.items()
        if 10 * count >= 9 * len(paths)
    }
    if len(qualifying) != 1:
        raise ValueError(f'Expected one conserved family/substitution pair, found {len(qualifying)}')
    tag, substitution = qualifying.pop()
    return {
        'shared_plasmid_family_tag': tag,
        'median_plasmid_copy_number': float(median(cohort_copies[tag])),
        'amino_acid_substitution': substitution,
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort_dir')
    args = parser.parse_args()
    print(json.dumps(audit_outbreak_cohort(args.cohort_dir)))
