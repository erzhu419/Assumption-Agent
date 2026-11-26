import argparse, json, os, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG = Path(os.environ.get('HEGEL_SKILL_INSPECTION_LOG', '/tmp/.hegel_skill_inspections.jsonl'))
cat = json.loads((HERE / '_skill_contracts.json').read_text(encoding='utf-8'))
ap = argparse.ArgumentParser()
ap.add_argument('--list', action='store_true')
ap.add_argument('--describe', type=str, default='')
args = ap.parse_args()
def _log(ev, tgt):
    try:
        with LOG.open('a', encoding='utf-8') as f:
            f.write(json.dumps({'event': ev, 'target': tgt, 'mode': 'knowledge_only_ref', 'ts': time.time()}) + '\n')
    except OSError:
        pass
if args.list:
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
        _log('skill_contract_delivered', q)
        print(json.dumps({'ok': True, 'module_stem': m_st, 'compiled_code_sha256': hit['compiled_code_sha256'], 'mathematical_preconditions': hit['mathematical_preconditions'], 'function_level_applicability': hit.get('function_level_applicability', {}), 'functions': fns, 'reference_source_txt': str(HERE / (m_st + '.reference.txt'))}, indent=2))
