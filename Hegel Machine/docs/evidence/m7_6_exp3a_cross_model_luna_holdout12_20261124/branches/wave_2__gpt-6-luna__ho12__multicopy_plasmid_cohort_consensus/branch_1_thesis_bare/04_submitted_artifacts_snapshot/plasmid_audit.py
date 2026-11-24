"""Audit a multi-isolate cohort for its shared multicopy AMR plasmid."""
import glob
import json
import os
import statistics

_CODE = {
    'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S',
    'TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
    'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P',
    'CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
    'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T',
    'AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
    'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A',
    'GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G',
}


def _reverse_complement(sequence):
    return sequence.translate(str.maketrans('ACGT', 'TGCA'))[::-1]


def audit_outbreak_cohort(cohort_dir: str) -> dict:
    """Return the shared plasmid family, median copy number, and amino-acid change."""
    paths = glob.glob(os.path.join(cohort_dir, 'isolate_*.json'))
    isolates = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            isolates.append(json.load(handle))
    if len(isolates) < 12:
        raise ValueError('cohort must contain at least 12 isolate bundles')

    with open(os.path.join(cohort_dir, 'reference_panel.json'), encoding='utf-8') as handle:
        reference = json.load(handle)['reference_wildtype_cds'].upper()

    candidates_by_isolate = []
    family_counts = {}
    for isolate in isolates:
        contigs = isolate['contigs']
        chrom_depths = [float(c['mean_depth']) for c in contigs
                        if str(c.get('family_tag', '')).startswith('CHR_SEG')]
        if not chrom_depths:
            candidates_by_isolate.append([])
            continue
        chrom_median = statistics.median(chrom_depths)
        candidates = []
        for contig in contigs:
            seq = contig.get('sequence', '').upper()
            if contig.get('circular') is True and len(seq) >= 2400 and chrom_median > 0:
                copy = float(contig['mean_depth']) / chrom_median
                if 3.2 <= copy <= 7.5:
                    candidates.append((contig, copy))
                    tag = contig.get('family_tag')
                    family_counts[tag] = family_counts.get(tag, 0) + 1
        candidates_by_isolate.append(candidates)

    if not family_counts:
        raise ValueError('no multicopy circular plasmid candidates found')
    family = max(family_counts, key=family_counts.get)
    if family_counts[family] < 0.9 * len(isolates):
        raise ValueError('no plasmid family shared by at least 90% of isolates')

    copies, sequences = [], []
    for candidates in candidates_by_isolate:
        for contig, copy in candidates:
            if contig.get('family_tag') == family:
                copies.append(copy)
                sequences.append(contig['sequence'].upper())
                break

    # Align the conserved resistance-core motif; a codon may differ at multiple
    # nucleotides while encoding one amino-acid substitution.
    variants = {}
    n = len(reference)
    for seq in sequences:
        for target in (reference, _reverse_complement(reference)):
            for start in range(len(seq) - n + 1):
                window = seq[start:start + n]
                oriented = window if target == reference else _reverse_complement(window)
                diffs = [i for i, (a, b) in enumerate(zip(reference, oriented)) if a != b]
                if not diffs:
                    continue
                changed_codons = {i // 3 for i in diffs}
                if len(changed_codons) != 1:
                    continue
                codon_index = next(iter(changed_codons))
                ref_aa = _CODE.get(reference[codon_index*3:codon_index*3+3])
                alt_aa = _CODE.get(oriented[codon_index*3:codon_index*3+3])
                if ref_aa and alt_aa and ref_aa != alt_aa:
                    key = (codon_index + 1, ref_aa, alt_aa)
                    variants[key] = variants.get(key, 0) + 1
                    break
            else:
                continue
            break
    if not variants:
        raise ValueError('no non-synonymous single-substitution reference match found')
    position, ref_aa, alt_aa = max(variants, key=variants.get)
    return {
        'shared_plasmid_family_tag': family,
        'median_plasmid_copy_number': float(statistics.median(copies)),
        'amino_acid_substitution': f'{ref_aa}{position}{alt_aa}',
    }
