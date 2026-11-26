import argparse, json, os, re, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get('HEGEL_SKILL_INSPECTION_LOG', '/tmp/.hegel_skill_inspections.jsonl'))
cat = json.loads((HERE / '_skill_contracts.json').read_text(encoding='utf-8'))
ap = argparse.ArgumentParser()
ap.add_argument('--list', action='store_true')
ap.add_argument('--search', type=str, default='')
ap.add_argument('--describe', type=str, default='')
args = ap.parse_args()
def _log(ev, tgt, **extra):
    try:
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': ev, 'target': tgt, 'mode': 'knowledge_only_ref', 'ts': time.time(), **extra}) + '\n')
    except OSError:
        pass
if args.search:
    ql = args.search.lower()
    rw = set(re.findall(r'[a-zA-Z0-9]{2,}', re.sub(r'[_./-]+', ' ', ql)))
    sgs = (
        {'robust','regression','outlier','outliers','irls','cauchy','huber','lorentzian','glitch','spikes','residual','weights'},
        {'certified','certificate','lower','bound','bounds','l0','sparse','box','subcube','cube','minorant','branch','blocks'},
        {'preconditioner','multigrid','amg','galerkin','cg','conjugate','spd','elliptic','poisson','diffusion','robin'},
        {'panel','reference','control','controls','cohort','batch','center','stratum','strata','imputation','missing','align','balanced','fill'},
        {'nuisance','profile','profiled','profiling','whitener','schur','tomography','amplitudes','matching'},
        {'relaxation','creep','viscoelastic','memory','hereditary','prony','step','breakpoint','tau','trend'},
    )
    ex = set(rw)
    for g in sgs:
        if rw & g: ex |= g
    scored = []
    for s in cat['skills']:
        mst = s['module_stem']; fmap = s.get('function_level_applicability') or {}
        isl1 = 1.0 if s.get('granularity_tier') == 'L1_compositional_primitive' else 0.0
        fns = s.get('functions', [])
        for ix, c in enumerate(s.get('index_cards', [])):
            fi = fns[ix] if ix < len(fns) else {}; fnm = str(fi.get('function') or '')
            astat = str(fmap.get(fnm) or '')
            if astat.startswith('REJECTED_'): continue
            fw = set(re.findall(r'[a-zA-Z0-9]{2,}', re.sub(r'[_./-]+', ' ', ' '.join([mst, fnm, str(fi.get('signature') or ''), str(fi.get('doc_summary') or '')]).lower())))
            dh = len(rw & fw); sh = len(ex & fw); eb = 6.0 if (fnm and fnm.lower() in ql) else (3.0 if mst.lower() in ql else 0.0)
            sc = 3.0*dh + 1.2*sh + eb + (0.5*isl1 if (dh+sh)>0 else 0.0)
            if not rw: sc = 1.0 + isl1
            if sc > 0: scored.append((sc, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = [c for _, c in scored[:6]]
    _log('skill_search_executed', args.search, query=args.search, returned_card_count=len(top))
    print('\n'.join(top) if top else 'No matching reference cards found.')
elif args.list:
    _log('skill_index_listed', '__all__')
    print(json.dumps([{'module_stem': s['module_stem'], 'index_cards': s['index_cards']} for s in cat['skills']], indent=2))
elif args.describe:
    q = args.describe.strip()
    _log('skill_contract_requested', q)
    parts = q.split('.')
    m_st = parts[0]
    fn_nm = parts[1] if len(parts) > 1 else ''
    hit = next((s for s in cat['skills'] if s['module_stem'] == m_st), None)
    if hit is None:
        print(json.dumps({'ok': False, 'error': f'unknown_reference:{q}'}))
    else:
        fns = [f for f in hit['functions'] if (not fn_nm or f['function'] == fn_nm)]
        if not fns:
            print(json.dumps({'ok': False, 'error': f'unknown_function_in_reference:{q}'}))
        else:
            _log('skill_contract_delivered', q, matched_module_count=1)
            print(json.dumps({'ok': True, 'module_stem': m_st, 'compiled_code_sha256': hit['compiled_code_sha256'], 'mathematical_preconditions': hit['mathematical_preconditions'], 'function_level_applicability': hit.get('function_level_applicability', {}), 'functions': fns, 'reference_source_txt': str(HERE / (m_st + '.reference.txt'))}, indent=2))
