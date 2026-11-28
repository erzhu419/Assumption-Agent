from pathlib import Path
src=Path('/tmp/hegel_continual_archive/dev_m102_c04_batch_00_solver.py').read_text()
base=src.split('def _select_interaction')[0]+src[src.index('def _ridge'):src.index('def solve_batch')]
entry='''def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    train = study['train']
    x, xt = train['x'], study['test']['x']
    offset, matrix = _calibration(study['panel_a'])
    q = np.linalg.solve(matrix.T, (np.concatenate((x, xt))-offset).T).T
    p_op, r_op = _panel_operators(study['panel_c'])
    common, offdiag = _features(p_op(q), r_op(q))
    n = len(x)
    y = _ridge(common[:n], train['y'], common[n:], 1e-6)
    return {'common':common, 'offdiag':offdiag, 'y_hat':y}
'''
extras='''def _fit_r(panel, mode):
    from scipy.optimize import minimize_scalar
    u, z = panel['u'], panel['v2']
    def design(x,k):
        a=_affine(x)
        i,j=np.triu_indices(x.shape[1])
        p=x[:,i]*x[:,j]
        n=np.sum(x*x,axis=1,keepdims=True)
        cross=p[:,i!=j]
        if mode=='poly': return np.c_[a,p]
        if mode=='radial': return np.c_[a,p]/(1+k*n)
        if mode=='cross': return np.c_[a,cross]
        if mode=='hybrid': return np.c_[a,x*x/(1+k*x*x),cross]
        if mode=='bounded_cross': return np.c_[a,x*x/(1+k*x*x),cross/(1+k*cross*cross)]
        if mode=='scale_mix': return np.c_[a,x*x/(1+.22*x*x),cross/(1+k*cross*cross)]
        if mode=='cross_scale': return np.c_[a,x*x/(1+k*x*x),cross/(1+.22*cross*cross)]
        if mode=='cycle_mix': return np.c_[a,x*x/(1+k*x*x),x*np.roll(x,-1,axis=1)]
    def loss(k):
        f=design(u,k); e=f@_ls(f,z)-z; return np.mean(e*e)
    k=minimize_scalar(loss,bounds=(0.,5.),method='bounded').x if mode not in ('poly','cross') else 0.
    w=_ls(design(u,k),z)
    return lambda x: design(x,k) @ w
'''
Path('/tmp/spatial_base.txt').write_text(base+extras+'\n'+entry)
import subprocess,os
p=Path('/holdout_task/node_00.py'); env=dict(os.environ,OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
best=(float('inf'),None,None)
for mode in ['archived','poly','radial','cross','hybrid','bounded_cross','scale_mix','cross_scale','cycle_mix']:
    code=base+extras+'\n'+entry
    if mode!='archived':code=code.replace('    common, offdiag =',f'    r_op = _fit_r(study["panel_c"], {mode!r})\n    common, offdiag =')
    p.write_text(code)
    r=subprocess.run(['python3','-I','-B','/holdout_task/test_dag_candidate.py'],capture_output=True,text=True,env=env)
    print(mode,r.stdout.strip(),r.stderr.strip(),flush=True)
    try:
        import json
        d=json.loads(r.stdout.strip().splitlines()[-1]); m=d['cv_metrics']; err=max(m.values())
        if err<best[0]: best=(err,mode,code)
    except (ValueError,IndexError,KeyError): pass
if best[2] is not None:p.write_text(best[2]); print('best',best[:2])
