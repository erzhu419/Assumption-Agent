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
    seq=us(row); x=np.asarray([0.]*max(0,6-len(seq))+seq[-6:])
    return np.r_[1.,x,[x[i]*x[j] for i in range(6) for j in range(i,6)]]

def residual(p,rows):
    a,b,c=p[:3]; q=p[3:]; out=[]
    for row in rows:
        m=mp=up=0.
        for u in us(row):
            m=a*mp+b*u+c*up; z=np.array([m,mp,u,up])
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
        return 'import numpy as np\nC=np.array('+repr(coef.tolist())+',dtype=float)\ndef predict(row):\n us=[float(v["u"]) for v in row.get("history",[])]+[float(row["features"]["u"])]\n x=np.asarray([0.]*max(0,6-len(us))+us[-6:])\n f=np.r_[1.,x,[x[i]*x[j] for i in range(6) for j in range(i,6)]]\n return float(f@C)\n'
    return 'import numpy as np\nP=np.array('+repr(coef.tolist())+',dtype=float)\ndef predict(row):\n us=[float(v["u"]) for v in row.get("history",[])]+[float(row["features"]["u"])]\n a,b,c=P[:3]; q=P[3:]; m=mp=up=0.\n for u in us:\n  m=a*mp+b*u+c*up; z=np.array([m,mp,u,up]); f=np.array([1.,*z,z[0]**2,z[0]*z[1],z[0]*z[2],z[0]*z[3],z[1]**2,z[1]*z[2],z[1]*z[3],z[2]**2,z[2]*z[3],z[3]**2]); y=float(f@q); mp,up=m,u\n return y\n'

def readout(result,public_spec,memory):
    fs=[f for f in result.get('forecasts',[]) if f.get('prospective_mse') is not None and f.get('returncode')==0]
    good=[f for f in fs if float(f['prospective_mse'])<=THRESHOLD]
    bad=[f for f in fs if float(f['prospective_mse'])>THRESHOLD]
    ids=[]
    for rec in result.get('observations',[]):
        if rec.get('row_indices'): ids.extend(rec['row_indices'])
        elif rec.get('observation_index') is not None: ids.extend(range(int(rec['observation_index']),int(rec['observation_index'])+int(rec.get('row_count',0))))
    return {'status':'supported' if good else ('refuted' if bad else 'inconclusive'),'answer':{'supported_models':[f['model'] for f in good],'refuted_models':[f['model'] for f in bad],'prospective_mse':{f['model']:float(f['prospective_mse']) for f in fs},'interpretation':'Frozen pre-acquisition forecasts were compared with the purchased trajectories. Retain the inherited explanation when it fits; reject only alternatives whose prospective error exceeds the paid-fit threshold.'},'evidence':sorted(set(int(i) for i in ids))}

def read_state(memory):
    try:
        with open('research_state.json') as f:return json.load(f)
    except Exception:return memory.get('research_state',{})

def select_actions(public_spec,observations,memory):
    rows=rows_of(observations); state=read_state(memory)
    if len(rows)>72:
        results=state.get('method_results',[])
        analyzed=any(m.get('proposal',{}).get('action')=='run' and m.get('proposal',{}).get('arguments',{}).get('operator')=='analyze_revision' for m in results)
        if state.get('revisions') and not analyzed:
            return {'action':'run','arguments':{'operator':'analyze_revision','revision_index':len(state['revisions'])-1,'path':'shared/readout.py'},'research_question':'prospective'}
        prior_path=memory.get('research_roles',{}).get('predictor',ARCHIVE)
        try:
            spec=importlib.util.spec_from_file_location('prior',prior_path); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
            prior_err=float(np.mean([(float(mod.predict(r))-float(r['target']))**2 for r in rows]))
        except Exception: prior_err=float('inf')
        if prior_err<THRESHOLD:return {'model':prior_path,'lesson':'The authored prospective readout found the unchanged inherited predictor supported and the competing paid-fit alternatives refuted; retain the inherited explanation.'}
        y=np.asarray([float(r['target']) for r in rows]); X=np.asarray([features(r) for r in rows])
        coef=np.linalg.lstsq(X,y,rcond=None)[0]; candidates=[(float(np.mean((X@coef-y)**2)),coef,'poly')]
        err,p=fit_latent(rows); candidates.append((err,p,'latent'))
        err,coef,kind=min(candidates,key=lambda z:z[0])
        if err<THRESHOLD:
            with open('model.py','w') as f:f.write(model_text(coef,kind))
            return {'model':'model.py','lesson':'Prospective outcomes refuted the inherited predictor; fit and submit the best exact control-history explanation.'}
        return {'stop':'The inherited explanation and fitted alternatives do not explain all paid outcomes.'}
    # Ensure the independent readout file exists in every trial workspace before the revision is acquired.
    with open('shared/readout.py','w') as f:
        f.write('THRESHOLD=1e-10\ndef readout(result,public_spec,memory):\n fs=[x for x in result.get(\'forecasts\',[]) if x.get(\'prospective_mse\') is not None and x.get(\'returncode\')==0]\n good=[x for x in fs if float(x[\'prospective_mse\'])<=THRESHOLD]\n bad=[x for x in fs if float(x[\'prospective_mse\'])>THRESHOLD]\n ids=[]\n for r in result.get(\'observations\',[]):\n  if r.get(\'row_indices\'): ids.extend(r[\'row_indices\'])\n  elif r.get(\'observation_index\') is not None: ids.extend(range(int(r[\'observation_index\']),int(r[\'observation_index\'])+int(r.get(\'row_count\',0))))\n return {\'status\':\'supported\' if good else (\'refuted\' if bad else \'inconclusive\'),\'answer\':{\'supported_models\':[x[\'model\'] for x in good],\'refuted_models\':[x[\'model\'] for x in bad],\'prospective_mse\':{x[\'model\']:float(x[\'prospective_mse\']) for x in fs},\'interpretation\':\'Compare frozen pre-acquisition forecasts with purchased outcomes; retain supported explanations and reject only alternatives with prospective MSE above threshold.\'},\'evidence\':sorted(set(int(i) for i in ids))}\n')
    comparisons=state.get('experiment_comparisons',[]); latest=state.get('latest_comparison') or (comparisons[-1] if comparisons else None)
    if latest and latest.get('selected_experiment'):
        selected=latest['selected_experiment']; args=dict(latest.get('query_arguments') or {'actions':selected['actions'],'revision':selected.get('revision',{})})
        args['revision']=selected.get('revision',args.get('revision',{})); args['readout']='shared/readout.py'
        return {'action':'query','arguments':args,'research_question':'prospective'}
    y=np.asarray([float(r['target']) for r in rows]); X=np.asarray([features(r) for r in rows])
    poly=np.linalg.lstsq(X,y,rcond=None)[0]; _,latent=fit_latent(rows)
    with open('shared/prospective_poly.py','w') as f:f.write(model_text(poly,'poly'))
    with open('shared/prospective_latent.py','w') as f:f.write(model_text(latent,'latent'))
    n=min(24,int(public_spec.get('max_probe_length',24)))
    def action(seq):return {'kind':'trajectory','sequence':[{'u':float(v)} for v in seq],'channels':[]}
    experiments=[{'id':'alternating_and_hold','actions':[action([-.95 if t%2==0 else .95 for t in range(n)]),action([.65]*(n//2)+[-.65]*(n-n//2))]},{'id':'switches_and_reversals','actions':[action([-.9 if (t//3)%2==0 else .9 for t in range(n)]),action([.8]*(n//3)+[-.7]*(n//3)+[.2]*(n-2*(n//3)))]}]
    return {'action':'run','arguments':{'operator':'compare_experiments','target':'response','models':[memory.get('research_roles',{}).get('predictor',ARCHIVE),'shared/prospective_poly.py','shared/prospective_latent.py'],'experiments':experiments},'research_question':{'id':'prospective','question':'Which authored bounded trajectories most clearly separate the archived response predictor from paid-fit finite-lag and latent-state alternatives?'}}

_select_single_control = select_actions

def window_matrix(rows, width):
    return np.asarray([[len(s), sum(float(v['x']) for v in s),
                        sum(float(v['z'])**2 for v in s)]
                       for row in rows
                       for s in [(row.get('history', [])+[row['features']])[-width:]]])

def fit_gated_window(rows):
    y=np.asarray([r['target'] for r in rows],dtype=float)
    g=np.asarray([r['features']['c'] for r in rows],dtype=float)
    values=np.unique(g); cuts=(values[:-1]+values[1:])/2
    best=None
    for width in range(1,9):
        X=window_matrix(rows,width)
        for cut in cuts:
            mask=g<cut
            if min(mask.sum(),(~mask).sum())<3: continue
            if min(np.linalg.matrix_rank(X[mask]),np.linalg.matrix_rank(X[~mask]))<3: continue
            coefficients=[np.linalg.lstsq(X[~mask],y[~mask],rcond=None)[0],
                          np.linalg.lstsq(X[mask],y[mask],rcond=None)[0]]
            pred=np.where(mask,X@coefficients[1],X@coefficients[0])
            error=float(np.mean((pred-y)**2))
            if best is None or error<best['error']:
                best={'error':error,'width':width,'coefficients':[v.tolist() for v in coefficients],
                      'lower':float(g[mask].max()),'upper':float(g[~mask].min())}
    return best

def gated_source(fit):
    return ('W='+repr(fit['width'])+'\nC='+repr(fit['coefficients'])+
            '\nT='+repr((fit['lower']+fit['upper'])/2)+
            '\ndef represent(row):\n s=(row.get("history",[])+[row["features"]])[-W:]\n return [float(row["features"]["c"]),len(s),sum(float(v["x"]) for v in s),sum(float(v["z"])**2 for v in s)]\n'
            'def predict_features(f):\n c,n,x,z=f\n b,a,d=C[int(c<T)]\n return float(b*n+a*x+d*z)\n'
            'def predict(row):\n return predict_features(represent(row))\n')

def multivariate_actions(public_spec,records,memory):
    from pathlib import Path
    rows=rows_of(records)
    names=list(public_spec['controls'])
    def trajectory(sequence):
        return {'kind':'trajectory','sequence':sequence,'channels':[]}
    def random_actions(seed,count):
        rng=np.random.default_rng(seed)
        n=int(public_spec['max_probe_length'])
        return [trajectory([{k:float(rng.uniform(*public_spec['feature_bounds'][k])) for k in names}
                            for _ in range(min(n,count-start))]) for start in range(0,count,n)]
    if len(rows)==72:
        return {'actions':random_actions(2841,128),
                'research_question':{'id':'window_extent','question':'Does the short-prefix gated additive window explanation persist through full-horizon trajectories, or does older history change the response?'}}
    marker=Path('shared/window_validation.json')
    if marker.exists():
        saved=json.loads(marker.read_text())
        fresh=rows[saved['cutoff']:]
        if not fresh: return {'stop':'The requested prospective validation returned no new response rows.'}
        namespace={}; exec(Path('shared/gated_candidate.py').read_text(),namespace)
        err=float(np.mean([(namespace['predict'](r)-r['target'])**2 for r in fresh]))
        Path('shared/window_validation_result.json').write_text(json.dumps({'mse':err,'rows':len(fresh),'paid_cutoff':saved['cutoff']}))
        if err>1e-8:
            return {'stop':'The frozen gated-window prediction failed its full-horizon prospective challenge: '+str(err)}
        return {'model':'shared/gated_candidate.py','lesson':'Learned window length, branch coefficients and gate bounds from current paid evidence; narrowed the unresolved gate with reset contrasts, then validated frozen predictions on new full-horizon trajectories.'}
    fit=fit_gated_window(rows)
    Path('shared/window_fit.json').write_text(json.dumps(fit))
    if fit is None or fit['error']>1e-12:
        return {'stop':'The inherited gated additive window family does not explain the paid full-horizon outcomes: '+str(None if fit is None else fit['error'])}
    if fit['upper']-fit['lower']>1e-9:
        # Distinguish all remaining threshold fits where their branch forecasts
        # have the greatest contrast; reset removes uncertainty about memory.
        difference=np.asarray(fit['coefficients'][1])-np.asarray(fit['coefficients'][0])
        xname,zname,cname=fit['coordinates']
        xb=public_spec['feature_bounds'][xname]; zb=public_spec['feature_bounds'][zname]
        settings=[(x,z) for x in [xb[0],0.,xb[1]] for z in [zb[0],0.,zb[1]]]
        x,z=max(settings,key=lambda v:abs(float(np.asarray([1.,v[0],v[1]**2])@difference)))
        gates=np.linspace(fit['lower'],fit['upper'],17)[1:-1]
        return {'actions':[trajectory([{xname:float(x),zname:float(z),cname:float(c)}]) for c in gates],
                'research_question':{'id':'gate_boundary','question':'Which remaining gate thresholds survive reset probes inside the observed branch-transition interval?'}}
    Path('shared/gated_candidate.py').write_text(gated_source(fit))
    marker.write_text(json.dumps({'cutoff':len(rows),'fit':fit}))
    return {'actions':random_actions(95371,128),
            'research_question':{'id':'window_validation','question':'Can the frozen fitted gate and finite-window explanation predict independent full-horizon controls before submission?'}}

def select_actions(public_spec,observations,memory):
    if list(public_spec['controls'])==['u']:
        return _select_single_control(public_spec,observations,memory)
    return multivariate_actions(public_spec,observations,memory)


_fit_current_gate = fit_gated_window
_current_gate_source = gated_source

def fit_gated_window(rows):
    best = _fit_current_gate(rows)
    if best is not None and best['error'] < 1e-20:
        return best
    y=np.asarray([r['target'] for r in rows],dtype=float)
    sequences=[r.get('history',[])+[r['features']] for r in rows]
    gates=np.unique([float(v['c']) for seq in sequences for v in seq])
    cuts=(gates[:-1]+gates[1:])/2
    n=len(rows)
    values=np.zeros((n,8,3)); controls=np.zeros((n,8)); present=np.zeros((n,8),dtype=bool)
    for i,seq in enumerate(sequences):
        for lag,v in enumerate(reversed(seq[-8:])):
            values[i,lag]=[1.,float(v['x']),float(v['z'])**2]
            controls[i,lag]=float(v['c']); present[i,lag]=True
    for cut in cuts:
        low=(controls<cut)&present
        hi_sum=np.cumsum(values*(present&~low)[:,:,None],axis=1)
        lo_sum=np.cumsum(values*low[:,:,None],axis=1)
        for width in range(1,9):
            X=np.column_stack([hi_sum[:,width-1],lo_sum[:,width-1]])
            coefficients,_,rank,_=np.linalg.lstsq(X,y,rcond=None)
            if rank<6: continue
            error=float(np.mean((X@coefficients-y)**2))
            if best is None or error<best['error']:
                best={'error':error,'width':width,'coefficients':coefficients.reshape(2,3).tolist(),
                      'lower':float(gates[gates<cut].max()),
                      'upper':float(gates[gates>=cut].min()),'gate_mode':'per_step'}
    return best

def gated_source(fit):
    if fit.get('gate_mode')!='per_step':
        return _current_gate_source(fit)
    return ('W='+repr(fit['width'])+'\nC='+repr(fit['coefficients'])+
            '\nT='+repr((fit['lower']+fit['upper'])/2)+
            '\ndef represent(row):\n'
            ' s=(row.get("history",[])+[row["features"]])[-W:]\n'
            ' f=[[0.,0.,0.],[0.,0.,0.]]\n'
            ' for v in s:\n'
            '  a=f[int(float(v["c"])<T)]\n'
            '  a[0]+=1.; a[1]+=float(v["x"]); a[2]+=float(v["z"])**2\n'
            ' return f\n'
            'def predict_features(f):\n'
            ' return float(sum(a*b for group,coef in zip(f,C) for a,b in zip(group,coef)))\n'
            'def predict(row):\n return predict_features(represent(row))\n')


_fit_canonical_gate = fit_gated_window
_canonical_gated_source = gated_source

def fit_gated_window(rows):
    import itertools
    names=list(rows[0]['features'])
    preferred=('x','z','c') if set(names)=={'x','z','c'} else tuple(names)
    orders=[preferred]+[v for v in itertools.permutations(names) if v!=preferred]
    best=None
    for order in orders:
        def convert(v):
            return {canonical:float(v[name]) for canonical,name in zip(('x','z','c'),order)}
        mapped=[dict(row,features=convert(row['features']),
                     history=[convert(v) for v in row.get('history',[])]) for row in rows]
        fit=_fit_canonical_gate(mapped)
        if fit is not None and (best is None or fit['error']<best['error']):
            best=dict(fit,coordinates=list(order))
        if best is not None and best['error']<1e-20:
            break
    return best

def gated_source(fit):
    source=_canonical_gated_source(fit)
    # Simultaneous replacement also permits permutations of canonical names.
    for canonical in ('x','z','c'):
        source=source.replace('"'+canonical+'"','__CONTROL_'+canonical+'__')
    for canonical,name in zip(('x','z','c'),fit['coordinates']):
        source=source.replace('__CONTROL_'+canonical+'__',repr(name))
    return source
