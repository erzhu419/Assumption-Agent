import importlib.util, json
import numpy as np
from scipy.optimize import least_squares

ARCHIVE = 'archive/source_001/_evaluated_model/model.py'
THRESHOLD = 1e-10

def rows_of(records):
    return [r for rec in records for r in rec.get('rows', []) if isinstance(r.get('target'), (int, float))]

def us(row):
    return [float(v['u']) for v in row.get('history', [])] + [float(row['features']['u'])]

def features(row):
    seq = us(row)
    x = np.asarray([0.] * max(0, 6-len(seq)) + seq[-6:])
    return np.r_[1., x, [x[i]*x[j] for i in range(6) for j in range(i, 6)]]

def residual(p, rows):
    a,b,c = p[:3]; q = p[3:]
    out=[]
    for row in rows:
        m=mp=up=0.
        for u in us(row):
            m=a*mp+b*u+c*up
            z=np.array([m,mp,u,up])
            f=np.array([1.,*z,z[0]**2,z[0]*z[1],z[0]*z[2],z[0]*z[3],z[1]**2,z[1]*z[2],z[1]*z[3],z[2]**2,z[2]*z[3],z[3]**2])
            y=float(f@q); mp,up=m,u
        out.append(y-float(row['target']))
    return np.asarray(out)

def fit_latent(rows):
    rng=np.random.default_rng(12); best=None
    for _ in range(5):
        p=np.r_[rng.uniform(-.5,.8),rng.uniform(.2,1.2),rng.uniform(-.6,.2),np.zeros(15)]
        p[3]=float(np.mean([r['target'] for r in rows]))
        opt=least_squares(residual,p,args=(rows,),max_nfev=1800,xtol=1e-13,ftol=1e-13,gtol=1e-13)
        err=float(np.mean(residual(opt.x,rows)**2))
        if best is None or err<best[0]: best=(err,opt.x)
        if err<1e-20: break
    return best

def model_text(coef,kind):
    if kind=='poly':
        return 'import numpy as np\nC=np.array('+repr(coef.tolist())+',dtype=float)\ndef predict(row):\n us=[float(v["u"]) for v in row.get("history",[])]+[float(row["features"]["u"])]\n x=np.asarray(([0.]*max(0,6-len(us))+us[-6:]))\n f=np.r_[1.,x,[x[i]*x[j] for i in range(6) for j in range(i,6)]]\n return float(f@C)\n'
    return 'import numpy as np\nP=np.array('+repr(coef.tolist())+',dtype=float)\ndef predict(row):\n us=[float(v["u"]) for v in row.get("history",[])]+[float(row["features"]["u"])]\n a,b,c=P[:3]; q=P[3:]; m=mp=up=0.\n for u in us:\n  m=a*mp+b*u+c*up; z=np.array([m,mp,u,up]); f=np.array([1.,*z,z[0]**2,z[0]*z[1],z[0]*z[2],z[0]*z[3],z[1]**2,z[1]*z[2],z[1]*z[3],z[2]**2,z[2]*z[3],z[3]**2]); y=float(f@q); mp,up=m,u\n return y\n'

def readout(result, public_spec, memory):
    forecasts = result.get('forecasts', [])
    scored = [f for f in forecasts if f.get('prospective_mse') is not None]
    valid = [f for f in scored if f.get('returncode') == 0]
    good = [f for f in valid if float(f['prospective_mse']) <= THRESHOLD]
    bad = [f for f in valid if float(f['prospective_mse']) > THRESHOLD]
    ids = []
    for rec in result.get('observations', []):
        ids.extend(rec.get('row_indices', []))
        if not rec.get('row_indices') and rec.get('observation_index') is not None:
            first = int(rec['observation_index'])
            count = int(rec.get('row_count', 0))
            ids.extend(range(first, first + count))
    if not ids:
        for forecast in scored:
            ids.extend(forecast.get('row_indices', []))
    ids = sorted(set(int(i) for i in ids))
    mse = {f['model']: float(f['prospective_mse']) for f in valid}
    answer = {
        'supported_models': [f['model'] for f in good],
        'refuted_models': [f['model'] for f in bad],
        'prospective_mse': mse,
        'interpretation': 'The acquired trajectories support models within the paid MSE threshold and refute those whose saved pre-acquisition forecasts exceed it.'
    }
    if good and bad:
        status = 'supported'
    elif good:
        status = 'supported'
    elif bad:
        status = 'refuted'
    else:
        status = 'inconclusive'
    return {'status': status, 'answer': answer, 'evidence': ids}

def read_state(memory):
    try:
        with open('research_state.json') as f: return json.load(f)
    except Exception: return memory.get('research_state',{})

def select_actions(public_spec, observations, memory):
    rows=rows_of(observations)
    prior_path=memory.get('research_roles',{}).get('predictor',ARCHIVE)
    prior_err=float('inf')
    try:
        spec=importlib.util.spec_from_file_location('prior',prior_path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        prior_err=float(np.mean([(float(mod.predict(r))-float(r['target']))**2 for r in rows]))
    except Exception: pass
    state=read_state(memory)
    # After the actual prospective acquisition/readout, submit the supported inherited model
    # if it still explains all paid observations; otherwise refit on those observations.
    if len(rows)>72:
        if prior_err<THRESHOLD:
            return {'model':prior_path,'lesson':'The inherited model fits the paid prospective outcomes; the authored readout rejected the paid-fit alternatives, so retain the unchanged explanation.'}
        y=np.asarray([float(r['target']) for r in rows]); X=np.asarray([features(r) for r in rows])
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        candidates=[(float(np.mean((X@coef-y)**2)),coef,'poly')]
        err,p=fit_latent(rows); candidates.append((err,p,'latent'))
        err,coef,kind=min(candidates,key=lambda z:z[0])
        if err<THRESHOLD:
            with open('model.py','w') as f:f.write(model_text(coef,kind))
            return {'model':'model.py','lesson':'The prospective evidence refuted the inherited model; fit the best exact control-history explanation to all paid rows.'}
        return {'stop':'The prospective evidence refuted the inherited model and current alternatives; retain the counterexamples for further revision.'}
    comparisons=state.get('experiment_comparisons',[])
    latest=state.get('latest_comparison') or (comparisons[-1] if comparisons else None)
    if latest and latest.get('selected_experiment'):
        selected=latest['selected_experiment']
        args=latest.get('query_arguments') or {'actions':selected['actions'],'revision':selected.get('revision',{})}
        args=dict(args); args['revision']=selected.get('revision',args.get('revision',{})); args['readout']='shared/consolidated_method.py'
        return {'action':'query','arguments':args,'research_question':'prospective'}
    y=np.asarray([float(r['target']) for r in rows]); X=np.asarray([features(r) for r in rows])
    poly=np.linalg.lstsq(X,y,rcond=None)[0]; _,latent=fit_latent(rows)
    with open('shared/prospective_poly.py','w') as f:f.write(model_text(poly,'poly'))
    with open('shared/prospective_latent.py','w') as f:f.write(model_text(latent,'latent'))
    n=min(24,int(public_spec.get('max_probe_length',24)))
    def action(seq): return {'kind':'trajectory','sequence':[{'u':float(v)} for v in seq],'channels':[]}
    experiments=[{'id':'alternating_and_hold','actions':[action([-.95 if t%2==0 else .95 for t in range(n)]),action([.65]*(n//2)+[-.65]*(n-n//2))]}, {'id':'switches_and_reversals','actions':[action([-.9 if (t//3)%2==0 else .9 for t in range(n)]),action([.8]*(n//3)+[-.7]*(n//3)+[.2]*(n-2*(n//3)))]}]
    return {'action':'run','arguments':{'operator':'compare_experiments','target':'response','models':[prior_path,'shared/prospective_poly.py','shared/prospective_latent.py'],'experiments':experiments},'research_question':{'id':'prospective','question':'Which authored bounded trajectory best separates the inherited predictor and paid-fit finite-lag and latent-memory alternatives?'}}

def load_predict(path):
    spec = importlib.util.spec_from_file_location('prior_predictor', path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.predict

def mse(predictor, rows):
    if predictor is None or not rows:
        return float('inf')
    return float(np.mean([(float(predictor(r)) - float(r['target'])) ** 2 for r in rows]))

# Scalar controls use prospective discrimination before submission: exact fit
# to the initial prefix alone does not establish the reset/history law.
def _scalar_select_actions(public_spec, observations, memory):
    rows = rows_of(observations)
    prior_path = memory.get('research_roles', {}).get('predictor', ARCHIVE)
    state = read_state(memory)
    if len(rows) > 72:
        prior = load_predict(prior_path)
        if prior is not None and mse(prior, rows) <= THRESHOLD:
            return {'model': prior_path, 'lesson': 'Prospective paid outcomes preserve the unchanged archived predictor; reuse it.'}
        y = np.asarray([float(r['target']) for r in rows])
        X = np.asarray([features(r) for r in rows])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        candidates = [(float(np.mean((X @ coef - y) ** 2)), coef, 'poly')]
        err, latent = fit_latent(rows)
        candidates.append((err, latent, 'latent'))
        err, coef, kind = min(candidates, key=lambda z: z[0])
        if err <= THRESHOLD:
            with open('model.py', 'w') as f:
                f.write(model_text(coef, kind))
            return {'model': 'model.py', 'lesson': 'Prospective evidence rejects the archived predictor; submit the best exact fit to all paid rows.'}
        return {'stop': 'Prospective evidence rejects the archived predictor and no fitted scalar alternative explains all paid rows.'}

    comparisons = state.get('experiment_comparisons', [])
    latest = state.get('latest_comparison') or (comparisons[-1] if comparisons else None)
    if latest and latest.get('selected_experiment'):
        selected = latest['selected_experiment']
        args = dict(latest.get('query_arguments') or {
            'actions': selected['actions'], 'revision': selected.get('revision', {})})
        args['revision'] = selected.get('revision', args.get('revision', {}))
        args['readout'] = 'shared/consolidated_method.py'
        return {'action': 'query', 'arguments': args, 'research_question': 'prospective'}

    y = np.asarray([float(r['target']) for r in rows])
    X = np.asarray([features(r) for r in rows])
    poly = np.linalg.lstsq(X, y, rcond=None)[0]
    _, latent = fit_latent(rows)
    with open('shared/prospective_poly.py', 'w') as f:
        f.write(model_text(poly, 'poly'))
    with open('shared/prospective_latent.py', 'w') as f:
        f.write(model_text(latent, 'latent'))
    n = min(24, int(public_spec.get('max_probe_length', 24)))
    def action(seq):
        return {'kind': 'trajectory', 'sequence': [{'u': float(v)} for v in seq], 'channels': []}
    experiments = [
        {'id': 'alternating_and_hold', 'actions': [
            action([-.95 if t % 2 == 0 else .95 for t in range(n)]),
            action([.65] * (n // 2) + [-.65] * (n - n // 2))]},
        {'id': 'switches_and_reversals', 'actions': [
            action([-.9 if (t // 3) % 2 == 0 else .9 for t in range(n)]),
            action([.8] * (n // 3) + [-.7] * (n // 3) + [.2] * (n - 2 * (n // 3)))]}]
    return {'action': 'run', 'arguments': {
        'operator': 'compare_experiments', 'target': 'response',
        'models': [prior_path, 'shared/prospective_poly.py', 'shared/prospective_latent.py'],
        'experiments': experiments}, 'research_question': {
        'id': 'prospective',
        'question': 'Which bounded trajectories distinguish the inherited predictor from paid-fit finite-lag and latent-memory alternatives?'}}

def _multivariate_window_fit(rows):
    cs = sorted({float(v.get('c', 0.0)) for r in rows
                 for v in list(r.get('history', [])) + [r.get('features', {})]})
    thresholds = [-1.500001] + [(a+b)/2 for a,b in zip(cs[:-1],cs[1:])] + [1.500001]
    y = np.asarray([float(r['target']) for r in rows])
    best = (float('inf'), None, None)
    for threshold in thresholds:
        X=[]
        for r in rows:
            window=(list(r.get('history', []))+[r.get('features', {})])[-5:]
            hi=[v for v in window if float(v.get('c',0.0)) >= threshold]
            lo=[v for v in window if float(v.get('c',0.0)) < threshold]
            X.append([sum(float(v.get('x',0.0)) for v in hi),
                      sum(float(v.get('z',0.0))**2 for v in hi),len(hi),
                      sum(float(v.get('x',0.0)) for v in lo),
                      sum(float(v.get('z',0.0))**2 for v in lo),len(lo)])
        X=np.asarray(X)
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        err=float(np.mean((X@coef-y)**2))
        if err < best[0]: best=(err,threshold,coef)
    err,threshold,coef=best
    source = ('import numpy as np\nC=np.asarray('+repr(coef.tolist())+')\nT='+repr(float(threshold))+'\n'
              'def predict(row):\n'
              ' s=list(row.get(\"history\",[]))+[row[\"features\"]]\n'
              ' w=s[-5:]\n'
              ' hi=[v for v in w if float(v.get(\"c\",0.))>=T]\n'
              ' lo=[v for v in w if float(v.get(\"c\",0.))<T]\n'
              ' q=[sum(float(v.get(\"x\",0.)) for v in hi),sum(float(v.get(\"z\",0.))**2 for v in hi),len(hi),sum(float(v.get(\"x\",0.)) for v in lo),sum(float(v.get(\"z\",0.))**2 for v in lo),len(lo)]\n'
              ' return float(C@np.asarray(q))\n')
    return err,source

def select_actions(public_spec, observations, memory):
    names = list(public_spec.get('controls', []))
    if len(names) == 1:
        return _scalar_select_actions(public_spec, observations, memory)

    rows = rows_of(observations)
    if not rows:
        return {'stop': 'No paid response rows.'}
    err, source = _multivariate_window_fit(rows)
    # A near-zero prefix fit from this direct window model failed case 009 confirmation.
    # Route multivariate cases through the finite native-condition search instead.
    if err <= -1:
        with open('model.py', 'w') as f:
            f.write(source)
        return {'model': 'model.py', 'lesson': 'A reset-aware threshold-conditioned window model fits paid evidence; submit for independent confirmation.'}

    state = read_state(memory)
    qid = 'multivariate_native_basis'
    native = [o for o in state.get('operator_results', [])
              if o.get('operator') == 'native_condition_research'
              and o.get('research_question_id') == qid
              and o.get('native_condition_research')]
    if not native:
        return {
            'action': 'run',
            'arguments': {'operator': 'native_condition_research',
                          'question': {'id': qid,
                                       'question': 'Does a reset-aware finite condition/window basis explain the multivariate paid response?',
                                       'parent_id': 'task', 'depends_on': []},
                          'max_candidates': 16, 'max_compute': 120000},
            'research_question': qid
        }

    nr = native[-1]['native_condition_research']
    # Use the native search's inexpensive boundary experiment before accepting
    # an exact paid fit: case 009 showed a gate-adjacent residual on fresh rows.
    prior_queries = [m.get('proposal', {}).get('arguments', {})
                     for m in state.get('method_results', [])
                     if m.get('proposal', {}).get('action') == 'query']
    boundary_already_bought = any(
        sum(len(a.get('sequence', [])) for a in q.get('actions', [])) == 1
        for q in prior_queries)
    if not boundary_already_bought:
        for experiment in nr.get('proposed_experiments', []):
            query_action = experiment.get('query_action', {})
            if experiment.get('purpose') == 'conditional_boundary' and query_action.get('arguments'):
                return {'action': 'query', 'arguments': query_action['arguments'],
                        'research_question': qid}
    if nr.get('status') == 'needs_matched_evidence':
        prior_queries = [m.get('proposal', {}).get('arguments', {})
                         for m in state.get('method_results', [])
                         if m.get('proposal', {}).get('action') == 'query']
        for experiment in nr.get('proposed_experiments', []):
            query_action = experiment.get('query_action')
            if query_action:
                args = dict(query_action.get('arguments', {}))
                if args.get('after_observation') and args not in prior_queries:
                    return {'action': 'query', 'arguments': args,
                            'research_question': qid}
        # A repeated proposal is not new evidence; fall through to evaluate any
        # returned candidates rather than purchasing the same experiment again.

    candidates = nr.get('candidates', [])
    evaluated = {}
    for step in state.get('method_results', []):
        prop = step.get('proposal', {})
        if prop.get('action') == 'evaluate':
            model = prop.get('arguments', {}).get('model')
            fb = step.get('feedback', {})
            if model:
                evaluated[model] = fb.get('mse')
                if (fb.get('returncode') == 0 and fb.get('mse') is not None
                        and float(fb['mse']) <= 1e-6):
                    return {'model': model,
                            'lesson': 'A native finite condition/window candidate fits paid evidence; submit for fresh confirmation.'}
    for candidate in candidates:
        model = candidate.get('model')
        if model and model not in evaluated:
            return {'action': 'evaluate', 'arguments': {'model': model},
                    'research_question': qid}
    return {'stop': 'The incumbent window and evaluated native candidates do not fit paid rows; retain counterevidence rather than repeat random queries.'}
