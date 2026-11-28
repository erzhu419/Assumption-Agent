import pickle, runpy, numpy as np
from pathlib import Path
m=runpy.run_path('/tmp/batch_v01.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
tr=s['train']; x,y,T=[tr[k] for k in ('x','y','traj')]; h,hs=m['_designs'](s,x); z=m['_maps'](s,x)[0]

def _scalar_member(h, y, n, penalty, guard):
    h = h[:, 1:]
    mu, sd = h[:n].mean(0), np.maximum(h[:n].std(0), 1e-9)
    a, q = np.split((h-mu)/sd, [n])
    u, d, v = np.linalg.svd(a, full_matrices=False)
    w = q @ v.T
    center = y.mean()
    r = u.T @ (y-center)
    base = center + w @ (r*d/(d*d+penalty))
    if guard:
        stable = center + w @ (r*d/(d*d+max(.10, penalty)))
        leverage = (w*w/(d*d+penalty)).sum(axis=1)
        mix = np.clip((leverage-guard)/np.maximum(leverage, 1e-12), 0., 1.)
        base += mix*(stable-base)
    return base

rng=np.random.default_rng(1); N=len(x)
splits=[np.arange(52,72)]+[rng.choice(N,20,replace=False) for _ in range(30)]+[np.argsort(q)[-20:] for q in [np.linalg.norm(z,axis=1),*z.T,*(-z.T)]]
for j in [0,1,2,3]: splits.append(np.argsort(z[:,j])[:20]) if False else None
recipes=[(p,g,w) for p in [(a,a,b,b) for a in [.0001,.001,.003,.01,.03,.08] for b in [.001,.01,.03,.08]] for g in [0.,.4,.8,1.5] for w in [.35,.5,.65]]
err=[]; ts_err=[]
for va in splits:
    ix=np.setdiff1d(np.arange(N),va); order=np.r_[ix,va]; n=len(ix)
    pred={}
    for p,g,w in recipes:
        for j,a in enumerate(p):
            k=(j,a,g)
            if k not in pred: pred[k]=_scalar_member(hs[j][order],y[ix],n,a,g)
    err.append([np.mean(abs((w/2)*(pred[0,p[0],g]+pred[1,p[1],g])+((1-w)/2)*(pred[2,p[2],g]+pred[3,p[3],g])-y[va])) for p,g,w in recipes])
    ts_err.append(np.mean(abs(h[va]@m['_fit'](h[ix],T[ix])-T[va])))
e=np.array(err); q=np.array(ts_err); eligible=(e[0]<.035)&(e[1:31].mean(0)<.032)&(e[1:31].max(0)<.043)
score=e[31:].max(0)+.4*e[31:].mean(0)+.5*e[1:31].mean(0)
j=np.argmin(np.where(eligible,score,np.inf)) if eligible.any() else np.argmin(np.where(e[0]<.0375,score,np.inf))
p,g,w=recipes[j]; print('retained recipe',p,g,w,'eligible',bool(eligible[j])); print('scalar and trajectory public, random mean/max, directional mean/max:')
for a in (e[:,j],q): print(a[0],a[1:31].mean(),a[1:31].max(),a[31:].mean(),a[31:].max())
src=Path('/tmp/batch_v01.py').read_text().split('def solve_batch')[0]
# Remove unused time-profile and archived operator wrappers.
for first,last in [('def _operator(', 'def _trajectory('),('def _trajectory(', 'def _panel_map(')]:
    if first in src: src=src[:src.index(first)]+src[src.index(last):]
helper=Path(__file__).read_text().split('def _scalar_member')[1].split('\nrng=')[0]
Path('/holdout_task/batch_solver.py').write_text(src+'def _scalar_member'+helper+f'''\ndef solve_batch(study: dict) -> dict:
    """Fit panel-calibrated scalar readouts and an affine trajectory map."""
    tr, te = study['train'], study['test']
    n = len(tr['x'])
    h, designs = _designs(study, np.concatenate((tr['x'], te['x'])))
    members = [_scalar_member(q, tr['y'], n, p, {g!r})
               for q, p in zip(designs, {p!r})]
    y = ({w!r}/2)*(members[0]+members[1]) + ((1-{w!r})/2)*(members[2]+members[3])
    state = h[n:] @ _fit(h[:n], tr['traj'])
    terminal = state[:, -1]
    return {{'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}}
''')
