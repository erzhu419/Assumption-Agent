from pathlib import Path
import importlib.util, pickle, numpy as np
p=Path('/holdout_task/batch_solver.py'); base=p.read_text()
start=base.index('def _features'); end=base.index('def _joint',start)
clean=base[:start]+'''def _features(study, x, c, d):
    panel = study['panel_b']
    b = _bias(x) @ _fit(_bias(panel['u']), panel['v'])
    bc = (b[:, :, None] * c[:, None, :]).reshape(len(x), -1)
    def nonlinear(z):
        return np.column_stack((np.sin(2.4*z), np.cos(2.4*z),
                                np.tanh(z), z/(1+z*z)))
    return bc, nonlinear(c), nonlinear(c*d)


'''+base[end:]
clean=clean.replace('_features(study, raw,x,c,d)','_features(study, x, c, d)').replace('for a in (raw,x,np.c_[c,d]):','for a in (raw,):').replace('    selected = [0, 1, 2, 3, 4, 39, 44, 52, len(K)-1]\n    K=np.asarray(K)[selected]','    K=np.asarray(K)')
a=clean.index('    p=minimize'); b=clean.index('\n\n',a)
def multi(kind):
    replacement='''    starts = [prior.copy() for _ in range(4)]
    starts[1][5:8] = [0, 5, 5]
    starts[2][5:8] = [3, 3, 3]
    starts[3][5:8] = [-6, 1, 5]
    fits = [minimize(fit_cov, start, jac=True, method='L-BFGS-B',
                     bounds=[(-17,6)]*(len(K)-1)+[(-17,0)],
                     options={'maxiter':250,'ftol':1e-11}) for start in starts]
    fits.sort(key=lambda f: f.fun)
'''
    if kind=='best': replacement+='    return fit_cov(fits[0].x, True)\n'
    else: replacement+='''    best = fits[0].fun
    chosen = [f for f in fits if f.fun <= 1.10 * best]
    return np.mean([fit_cov(f.x, True) for f in chosen], axis=0)
'''
    return clean[:a]+replacement+clean[b:]
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; N=len(t['x']); rng=np.random.default_rng(541)
splits=[np.arange(52),np.arange(56),np.arange(20,72)]+[np.sort(rng.choice(N,52,False)) for _ in range(3)]
options=[]
for k,code in [('original',base)]+[(str(w)+k,multi(k).replace('2/np.sqrt',f'{w}/np.sqrt')) for w in (2,4,8) for k in ('best','mean')]:
    env={};exec(compile(code,k,'exec'),env); es=[]
    for tr in splits:
        te=np.setdiff1d(np.arange(N),tr); out=env['solve_batch'](dict(s,train={j:t[j][tr] for j in ('x','y','traj')},test={'x':t['x'][te]})); e=abs(out['state_pred']-t['traj'][te]); es.append([abs(out['y_hat']-t['y'][te]).mean(),e.mean(),e[:,-1].mean()])
    es=np.array(es); worst=es.max(0); print(k,'52/20',es[0].round(5),'56/16',es[1].round(5),'worst',worst.round(5),flush=True)
    if max(es[0,:2])<=.038 and worst[0]<=.033:
        options.append((worst[1],worst[2],k,code))
_,_,name,code=min(options,key=lambda r:r[:2]); print('selected',name,flush=True)
if code!=base: p.write_text(code)
