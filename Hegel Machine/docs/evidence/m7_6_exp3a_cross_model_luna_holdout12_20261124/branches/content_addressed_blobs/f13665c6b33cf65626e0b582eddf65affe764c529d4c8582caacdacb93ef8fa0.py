"""Cohort-wide audit of a conserved multicopy resistance plasmid."""
import glob, json, os, statistics
_CODE = {
'TTT':'F','TTC':'F','TTA':'L','TTG':'L','TCT':'S','TCC':'S','TCA':'S','TCG':'S','TAT':'Y','TAC':'Y','TAA':'*','TAG':'*','TGT':'C','TGC':'C','TGA':'*','TGG':'W',
'CTT':'L','CTC':'L','CTA':'L','CTG':'L','CCT':'P','CCC':'P','CCA':'P','CCG':'P','CAT':'H','CAC':'H','CAA':'Q','CAG':'Q','CGT':'R','CGC':'R','CGA':'R','CGG':'R',
'ATT':'I','ATC':'I','ATA':'I','ATG':'M','ACT':'T','ACC':'T','ACA':'T','ACG':'T','AAT':'N','AAC':'N','AAA':'K','AAG':'K','AGT':'S','AGC':'S','AGA':'R','AGG':'R',
'GTT':'V','GTC':'V','GTA':'V','GTG':'V','GCT':'A','GCC':'A','GCA':'A','GCG':'A','GAT':'D','GAC':'D','GAA':'E','GAG':'E','GGT':'G','GGC':'G','GGA':'G','GGG':'G'}
def _motif_variants(seq, ref):
    out=set(); n=len(ref)
    if not n or n%3: return out
    for strand in (seq, seq.translate(str.maketrans('ACGT','TGCA'))[::-1]):
        for i in range(len(strand)-n+1):
            w=strand[i:i+n]
            if any(b not in 'ACGT' for b in w): continue
            diffs=[k for k in range(0,n,3) if w[k:k+3]!=ref[k:k+3]]
            if len(diffs)==1:
                k=diffs[0]; a,b=ref[k:k+3],w[k:k+3]; aa0,aa1=_CODE.get(a),_CODE.get(b)
                if aa0 and aa1 and aa0!=aa1 and aa1!='*': out.add((k//3+1,aa0,aa1))
    return out
def audit_outbreak_cohort(cohort_dir: str) -> dict:
    files=sorted(glob.glob(os.path.join(cohort_dir,'isolate_*.json')))
    if len(files)<12: raise ValueError('expected at least 12 isolate JSON bundles')
    with open(os.path.join(cohort_dir,'reference_panel.json'),encoding='utf-8') as f: ref=json.load(f)['reference_wildtype_cds'].upper()
    rows=[]
    for path in files:
        with open(path,encoding='utf-8') as f: isolate=json.load(f)
        contigs=isolate.get('contigs',[])
        depths=[float(c['mean_depth']) for c in contigs if str(c.get('family_tag','')).startswith('CHR_') and float(c.get('mean_depth',0))>0]
        if not depths: continue
        chrom=statistics.median(depths)
        for c in contigs:
            seq=str(c.get('sequence','')).upper(); copy=float(c.get('mean_depth',0))/chrom if chrom else 0
            if c.get('circular') is True and len(seq)>=2400 and 3.2<=copy<=7.5:
                rows.append((c.get('family_tag'),copy,_motif_variants(seq,ref)))
    counts={}
    for tag,_,_ in rows: counts[tag]=counts.get(tag,0)+1
    qualified=[(tag,n) for tag,n in counts.items() if tag and n/len(files)>=.90]
    if not qualified: raise ValueError('no circular multicopy plasmid family shared by >=90% of cohort')
    tag=max(qualified,key=lambda x:x[1])[0]
    selected=[r for r in rows if r[0]==tag]
    variants=set().union(*(r[2] for r in selected))
    if len(variants)!=1: raise ValueError(f'expected one conserved nonsynonymous codon substitution; found {variants}')
    pos,wt,alt=next(iter(variants))
    return {'shared_plasmid_family_tag':str(tag),'median_plasmid_copy_number':float(statistics.median(r[1] for r in selected)),'amino_acid_substitution':f'{wt}{pos}{alt}'}
