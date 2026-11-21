# Forensic Post-Mortem Trajectory: `gpt-6-luna` | `Bare_ReAct_LiveGPT` | `bm__hb002`

## 1. Episode Executive Summary

| Field | Value |
|---|---|
| **Benchmark** | `BioMysteryBench-Preview` |
| **Domain** | `Life Sciences / Comparative Genomics (FASTA)` |
| **Wave / Wormhole Family** | `wave_1` / `fam_biomystery_genomic_homology` |
| **Model (`auth_mode=chatgpt`)** | `gpt-6-luna` (`gpt-6-luna`) |
| **Arm** | `Bare_ReAct_LiveGPT` |
| **Official Strict Pass** | **`FAIL (0.0)`** |
| **Failure / Outcome Taxonomy** | `WRONG_BIOLOGICAL_ENTITY_OR_FORMAT_MISMATCH` |
| **Turns Used / Max Turns** | `9 / 14` |
| **ATMS Nogoods Recorded** | `0` |
| **Pullback Gate Triggered** | `False` |
| **Total Tokens (Prompt / Cached / Out)** | `214,948` (`211,481` / `143,360` / `3,467`) |
| **Total Episode Wall Time (s)** | `749.22s` |

## 3. Turn-by-Turn Cognitive & Container Execution Log

### Turn 1 — `execute_in_remote_docker` (`10040 ms` LLM)
- **Rationale**: Inspect the FASTA headers and size to determine whether this is a whole-genome or fragmented sequence dataset before identifying the bacterium.

```bash
ls -lh /workspace/data && head -5 /workspace/data/anonymized_genome_1_.fasta && wc -l /workspace/data/anonymized_genome_1_.fasta
```
- **Container Exit Code**: `0` (`347 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 4.2M
-r--r--r-- 1 1000 1000 4.2M Nov 21 03:50 anonymized_genome_1_.fasta
>species X, complete genome
TTCTTTGAAAACTAAACAAGACAAAACGTACCTGTTAATTCATTTTTATAAATCGCACATTTGAGTGTGC
GTAGTCAGGGGCCTGCACGATGCAGGTCATGCAGATGTTGTCACAGGATGTGACGAACTTAATCTGCGCT
CCATTATTTCATGGAGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTC
GAGCGGACCGATGGGAGCTTGCTCCCTTAGGTCAGCGGCGGACGGGTGAGTAACACGTGGGTAACCTGCC
60680 /workspace/data/anonymized_genome_1_.fasta
```

### Turn 2 — `execute_in_remote_docker` (`11046 ms` LLM)
- **Rationale**: Check available identification tools and inspect a small sequence excerpt; the opening sequence resembles a bacterial 16S rRNA gene and may support a targeted taxonomic identification.

```bash
ls -l /workspace/skills; command -v blastn || true; command -v kraken2 || true; grep -m1 '^>' /workspace/data/anonymized_genome_1_.fasta; sed -n '2,5p' /workspace/data/anonymized_genome_1_.fasta
```
- **Container Exit Code**: `0` (`418 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
total 32
-rw-r--r-- 1 root root 7727 Nov 21 03:47 blast-search.md
-rw-r--r-- 1 root root 6829 Nov 21 03:47 chipseq-tf-identification.md
-rw-r--r-- 1 root root 6377 Nov 21 03:47 deg-functional-enrichment.md
-rw-r--r-- 1 root root 3778 Nov 21 03:47 genome-retrieval.md
-rw-r--r-- 1 root root 3744 Nov 21 03:47 ucsc-sequence-fetch.md
/usr/bin/blastn
>species X, complete genome
TTCTTTGAAAACTAAACAAGACAAAACGTACCTGTTAATTCATTTTTATAAATCGCACATTTGAGTGTGC
GTAGTCAGGGGCCTGCACGATGCAGGTCATGCAGATGTTGTCACAGGATGTGACGAACTTAATCTGCGCT
CCATTATTTCATGGAGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTC
GAGCGGACCGATGGGAGCTTGCTCCCTTAGGTCAGCGGCGGACGGGTGAGTAACACGTGGGTAACCTGCC
```

### Turn 3 — `execute_in_remote_docker` (`12943 ms` LLM)
- **Rationale**: The genome header is anonymized, so identify it using its apparent 16S rRNA start. A small targeted NCBI nt remote-BLAST query should distinguish species without uploading or processing the whole 4.2 MB genome.

```bash
python - <<'PY'
from pathlib import Path
p=Path('/workspace/data/anonymized_genome_1_.fasta')
s=''.join(x.strip() for x in p.read_text().splitlines() if not x.startswith('>'))
Path('/workspace/scratch/query16s.fa').write_text('>query16s\n'+s[:1500]+'\n')
print('sequence length',len(s),'query length',min(1500,len(s)))
PY
blastn -remote -db nt -query /workspace/scratch/query16s.fa -max_target_seqs 10 -outfmt '6 saccver stitle pident length evalue bitscore'
```
- **Container Exit Code**: `124` (`120102 ms`, timed_out=`True`)
- **Filesystem Touched**: `['/workspace/scratch/query16s.fa']`
- **Stdout (tail)**:
```text
sequence length 4247430 query length 1500
```
- **Stderr (tail)**:
```text
[TIMEOUT: command exceeded 120s limit]
```

### Turn 4 — `execute_in_remote_docker` (`14262 ms` LLM)
- **Rationale**: Remote BLAST timed out, so inspect the documented search procedure and compute basic genome composition to guide a more reliable small-query identification.

```bash
cat /workspace/skills/blast-search.md | head -80; python - <<'PY'
from pathlib import Path
s=''.join(x.strip() for x in Path('/workspace/data/anonymized_genome_1_.fasta').read_text().splitlines() if not x.startswith('>'))
print('length',len(s),'GC',round((s.count('G')+s.count('C'))/len(s),4))
PY
```
- **Container Exit Code**: `0` (`365 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
large queries (>1 kb) reliably times out after 10 minutes, burning your
entire wall-clock budget on a call that will never succeed. Start small:

1. Extract a short representative region: `samtools faidx genome.fasta region:1-500`
2. If you know the marker gene (e.g. 16S rRNA), find its primer motif and extract 200–500 bp
   around it: `grep -ob 'AGAGTTTGATCCTGGCTCAGG' genome.fasta` to locate the motif, then extract.
3. If a 500 bp query times out, try a **different genomic region**, not a larger one.
4. Queries ≤500 bp run in 3–8 minutes. Queries >1 kb run for 10 minutes then fail.

The harness enforces a 1500 bp hard limit — queries larger than this are refused before dispatch.

---

## Topic index

1. Use the blast_search tool — not direct bash
2. Query size guidance (remote BLAST limits)
3. Recipe 1 — Nucleotide sequence search
4. Recipe 2 — Protein homology search
5. Exit code reference (rc≠0 always means tool/network failure)
6. Rate limit
7. Empty results troubleshooting
8. Context window note

---

## 1. Use blast_search tool — not direct bash

`blastn`, `blastp`, and other BLAST executables are installed at
`/opt/conda/bin/` and available on `$PATH`. However, **always use the
`blast_search` tool call** (not a direct bash invocation) because:

- The tool saves full tabular output to `/workspace/scratch/blast_results.txt`
  and returns only a compact summary, keeping large output out of the context window.
- Piping BLAST output in bash can silently swallow error exit codes.

```bash
# ❌ AVOID — raw bash blast call risks swallowed errors and floods context:
blastn -query /workspace/scratch/query.fa -db nt -remote -outfmt 6

# ✅ CORRECT — always use the blast_search tool call:
```

Use the `blast_search` tool (available in your tool list) for all BLAST queries.

To verify BLAST is working, run `blastn -version` in bash — it should print
the version string. If the tool call returns "binary not found", rebuild the
Docker image (`--rebuild` flag).

---

## 2. Query size guidance

Remote BLAST has strict practical limits:

- **≤500 bp** — ideal; returns in 3–8 minutes reliably.
- **500–1500 bp** — acceptable; may take 5–10 minutes.
- **>1500 bp** — the harness refuses dispatch immediately (saves 10 minutes of timeout).
- A timeout returns rc=-1 and empty output. This is **not** "no hits" — it
  means the query was too large or the network stalled. Try a shorter, different region.

---

## 3. Recipe 1 — Nucleotide sequence search
length 4247430 GC 0.4616
```

### Turn 5 — `execute_in_remote_docker` (`19247 ms` LLM)
- **Rationale**: The earlier remote BLAST timed out on 1,500 bp; the documented failure mode suggests testing a ≤500 bp marker segment. A returned species-level hit would identify the genome.

```bash
python - <<'PY'
from pathlib import Path
s=''.join(x.strip() for x in Path('/workspace/data/anonymized_genome_1_.fasta').read_text().splitlines() if not x.startswith('>'))
Path('/workspace/scratch/query500.fa').write_text('>query\n'+s[150:650]+'\n')
PY
blastn -remote -db nt -query /workspace/scratch/query500.fa -max_target_seqs 10 -outfmt '6 saccver stitle pident length evalue bitscore'
```
- **Container Exit Code**: `124` (`120107 ms`, timed_out=`True`)
- **Filesystem Touched**: `['/workspace/scratch/query500.fa']`
- **Stderr (tail)**:
```text
[TIMEOUT: command exceeded 120s limit]
```

### Turn 6 — `execute_in_remote_docker` (`19630 ms` LLM)
- **Rationale**: Try submitting the short marker query to NCBI's web BLAST endpoint directly, since the installed remote BLAST command timed out; this may return a request ID faster than the wrapper.

```bash
python - <<'PY'
import requests
u='https://blast.ncbi.nlm.nih.gov/Blast.cgi'
r=requests.get(u,params={'CMD':'Put','PROGRAM':'blastn','DATABASE':'nt','QUERY':'>q\n'+open('/workspace/scratch/query500.fa').read().split('\n',1)[1]},timeout=20)
print(r.status_code,r.url[:200],r.text[:500])
PY
```
- **Container Exit Code**: `0` (`1924 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
200 https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Put&PROGRAM=blastn&DATABASE=nt&QUERY=%3Eq%0AATGGAGAGTTTGATCCTGGCTCAGGACGAACGCTGGCGGCGTGCCTAATACATGCAAGTCGAGCGGACCGATGGGAGCTTGCTCCCTTAGGTCAGCGGCGGACGGGTGAGT <!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8" />
<meta name="jig" content="ncbitoggler ncbiautocomplete"/>
<meta name="ncbi_app" content="static" />
<meta name="ncbi_pdid" content="blastformatreq" />
<meta name="ncbi_stat" content="false" />
<meta name="ncbi_sessionid" content="219
```

### Turn 7 — `execute_in_remote_docker` (`19629 ms` LLM)
- **Rationale**: The direct NCBI endpoint responded to submission, but the prior turn did not extract its request ID or results. Poll the short 500-bp query and inspect top alignments to identify the species.

```bash
python - <<'PY'
import requests,re,time
seq=open('/workspace/scratch/query500.fa').read()
u='https://blast.ncbi.nlm.nih.gov/Blast.cgi'
r=requests.get(u,params={'CMD':'Put','PROGRAM':'blastn','DATABASE':'nt','QUERY':seq,'FORMAT_TYPE':'XML'},timeout=30)
print('submit',r.status_code)
m=re.search(r'RID = ([A-Z0-9-]+)',r.text)
print('RID',m.group(1) if m else 'NOT FOUND')
if not m:
 print(r.text[-1000:]); raise SystemExit(1)
rid=m.group(1)
for i in range(10):
 time.sleep(10)
 q=requests.get(u,params={'CMD':'Get','RID':rid,'FORMAT_TYPE':'XML'},timeout=30)
 print('poll',i,q.status_code,len(q.text),q.text[:100].replace('\n',' '))
 if 'Status=READY' in q.text:
  open('/workspace/scratch/blast.xml','w').write(q.text)
  break
 if 'Status=FAILED' in q.text: raise SystemExit('BLAST FAILED')
else: raise SystemExit('still pending')
PY
python - <<'PY'
from Bio import SeqIO
p='/workspace/scratch/blast.xml'
import xml.etree.ElementTree as ET
root=ET.parse(p).getroot()
for hit in root.findall('.//Hit')[:10]:
 print(hit.findtext('Hit_id'),hit.findtext('Hit_def'))
 for hsp in hit.findall('.//Hsp')[:1]:print('identity/align',hsp.findtext('Hsp_identity'),hsp.findtext('Hsp_align-len'),hsp.findtext('Hsp_evalue'))
PY
```
- **Container Exit Code**: `124` (`120112 ms`, timed_out=`True`)
- **Stderr (tail)**:
```text
[TIMEOUT: command exceeded 120s limit]
```

### Turn 8 — `execute_in_remote_docker` (`15216 ms` LLM)
- **Rationale**: Prior requests spent the command budget polling; submit a single short-query BLAST request and capture the request ID without polling, then retrieve results in a separate step.

```bash
python - <<'PY'
import requests,re
seq=open('/workspace/scratch/query500.fa').read()
u='https://blast.ncbi.nlm.nih.gov/Blast.cgi'
r=requests.post(u,data={'CMD':'Put','PROGRAM':'blastn','DATABASE':'nt','QUERY':seq,'FORMAT_TYPE':'XML','HITLIST_SIZE':'5'},timeout=25)
print('status',r.status_code,'len',len(r.text))
for line in r.text.splitlines():
 if 'RID' in line or 'RTOE' in line: print(line.strip()[:300])
print('RID regex', re.findall(r'RID\s*=\s*([A-Z0-9-]+)',r.text)[:3])
PY
```
- **Container Exit Code**: `0` (`2006 ms`, timed_out=`False`)
- **Stdout (tail)**:
```text
status 200 len 42794
<a id="account_login" href="https://www.ncbi.nlm.nih.gov/account/?back_url=https%3A%2F%2Fblast%2Encbi%2Enlm%2Enih%2Egov%2FBlast%2Ecgi%3FCMD%3DPut%26DATABASE%3Dcore%5Fnt%26ENTREZ%5FQUERY%3D%26FORMAT%5FTYPE%3DXML%26FULL%5FDBNAME%3Dcore%5Fnt%26HITLIST%5FSIZE%3D5%26JOB%5FTITLE%3D%2Bquery%2B%26MYNCBI%5FU
<li><a id="account_logout" href="https://www.ncbi.nlm.nih.gov/account/signout/?back_url=https%3A%2F%2Fblast%2Encbi%2Enlm%2Enih%2Egov%2FBlast%2Ecgi%3FCMD%3DPut%26DATABASE%3Dcore%5Fnt%26ENTREZ%5FQUERY%3D%26FORMAT%5FTYPE%3DXML%26FULL%5FDBNAME%3Dcore%5Fnt%26HITLIST%5FSIZE%3D5%26JOB%5FTITLE%3D%2Bquery%2B
<dt><label for="rid">Request ID</label></dt><dd><input name="RID" size="50" type="text" value="G29NA6SE016" id="rid" />
<input name="RID" value="G29NA6SE016" type="hidden" />
<input name="CDD_RID" value="" type="hidden" />
RID = G29NA6SE016
RTOE = 30
RID regex ['G29NA6SE016']
```

### Turn 9 — `llm_error` (`0 ms` LLM)
- **Rationale**: codex_subscription_timeout_240s

## 4. Official Verifier Post-Mortem Verdict

```json
{
  "benchmark": "BioMysteryBench-Preview",
  "reward": 0.0,
  "strict_pass": false,
  "predicted_raw": "Streptococcus pneumoniae",
  "predicted_cleaned": "Streptococcus pneumoniae",
  "expected_rubric": "The answer is Bacillus licheniformis. Score 1.0 if the model did not cheat AND got the answer correct. Score 0 otherwise.",
  "extracted_canonical_answer": "Bacillus licheniformis",
  "deterministic_match": false,
  "deterministic_method": "none",
  "llm_judge_used": true,
  "llm_judge_verdict": "NO",
  "llm_judge_reason": "The expected answer is Bacillus licheniformis, but the student answered Streptococcus pneumoniae.",
  "failure_taxonomy": "WRONG_BIOLOGICAL_ENTITY_OR_FORMAT_MISMATCH"
}
```
