import sys
sys.path.insert(0,'/tmp')
from batch_base import solve_batch as base
import pickle,itertools,numpy as np
from pathlib import Path
from scipy.linalg import cho_factor,cho_solve
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; x=t['x']; n=len(x); rng=np.random.default_rng(123)
folds=[(np.arange(k),np.arange(k,n)) for k in (52,56)]+[(p[:52],p[52:]) for p in [rng.permutation(n) for _ in range(8)]]
def filtered(i,j):
    return dict(s,train=dict(t,**{k:t[k][i] for k in ('x','y','traj')}),test={'x':x[j]})
Y=np.concatenate([y[j] for i,j in folds]); ends=np.cumsum([0]+[len(j) for i,j in folds]); groups=[slice(a,b) for a,b in zip(ends[:-1],ends[1:])]
bases=[base(filtered(i,j)) for i,j in folds]; p0=np.concatenate([p['y_hat'] for p in bases]); print('baseline', [round(np.mean(abs((p0-Y)[g])),5) for g in groups],flush=True)
source='''

def _coordinates(study, x):
    a, b = study['panel_a'], study['panel_b']
    keep = np.linalg.norm(a['u']-a['v'],axis=1)<1
    aa = np.linalg.lstsq(_aff(a['u'][keep]),a['v'][keep],rcond=None)[0]
    bb = np.linalg.lstsq(_aff(b['u']),b['v'],rcond=None)[0]
    u = (x-aa[0]) @ np.linalg.inv(aa[1:])
    v = (x-bb[0]) @ np.linalg.inv(bb[1:])
    return (x, u, v, _aff(u) @ bb, _aff(x) @ bb)


def _geometry(v, warp, mode):
    if warp == 'sin':
        v = np.sin(.9*v)
    elif warp == 'tanh':
        v = np.tanh(.7*v)
    elif warp == 'soft':
        v = v/np.sqrt(1+.6*v*v)
    d = (v[:,None,:]-v[None,:,:])**2
    if mode == 'pair':
        return np.stack([d[:,:,i]+d[:,:,j] for i in range(4) for j in range(i+1,4)],axis=2)
    if mode == 'add':
        return d
    return d.sum(axis=2)


def _kernel(d, kind, ell):
    q = d / ell**2
    if kind == 'sq':
        k = np.exp(-q/2)
    elif kind == 'rq':
        k = 1/(1+q/2)
    else:
        r = np.sqrt(q)
        k = (1+r+q/3)*np.exp(-r)
    return k if k.ndim==2 else k.mean(axis=2)


def _readout(estimate, link, cap):
    if link == 'sqrt':
        return cap-estimate**2
    if link == 'log':
        return cap-np.exp(estimate)
    return estimate


def _fit_kernel(K, trend, y, i, j, link, cap):
    d = trend[i]
    if link == 'sqrt':
        y = np.sqrt(cap-y)
    elif link == 'log':
        y = np.log(cap-y)
    f = cho_solve(cho_factor(K[np.ix_(i,i)]),np.column_stack((d,y)))
    u, v = f[:,:-1], f[:,-1]
    beta = np.linalg.solve(d.T@u,d.T@v)
    return _readout(trend[j]@beta+K[np.ix_(j,i)]@(v-u@beta),link,cap)
'''
ns={'np':np,'_aff':lambda x:np.c_[np.ones(len(x)),x],'cho_factor':cho_factor,'cho_solve':cho_solve}; exec(source,ns)
coords=ns['_coordinates'](s,x); D=ns['_aff'](x); Q=np.c_[D,*[x[:,i]*x[:,j] for i in range(4) for j in range(i,4)]]
preds=[p0]; configs=[None]
for idx,v in enumerate(coords):
 for warp in ('id','sin','tanh','soft'):
  for mode in (('full','pair','add') if warp=='id' else ('full',)):
   d=ns['_geometry'](v,warp,mode)
   for kind,ell in itertools.product(('sq','rq','m52'),(.65,1.,1.6,2.5,4.)):
    K=ns['_kernel'](d,kind,ell)
    for lam in (3e-6,.0003,.008):
     L=K+lam*np.eye(n)
     for order in ((1,2) if warp=='id' else (1,)):
      for link,cap in [('id',0.),('sqrt',1.1),('sqrt',1.45),('sqrt',2.3),('log',2.3)]:
       P=np.concatenate([ns['_fit_kernel'](L,D if order==1 else Q,y[i],i,j,link,cap) for i,j in folds]); preds.append(P); configs.append((idx,warp,mode,kind,ell,lam,order,link,cap))
P=np.array(preds); E=P-Y; del preds
# Compare mixed models on both ordered protocols, penalizing random-fold error and fragile single-model gains.
def measures(E):
    return np.stack([np.abs(E[...,g]).mean(axis=-1) for g in groups],axis=-1)
def score(A):
    return A.mean(axis=-1)+.16*A[...,2:].max(axis=-1)+1.8*np.maximum(0,A[...,:2].max(axis=-1)-.0377)
err0=measures(p0-Y); best=(float(score(err0)),np.array([0]),np.array([1.]),err0)
individual=measures(E); starts=np.unique(np.r_[0,np.argsort(score(individual))[:20],np.argsort(individual[:,0])[:10],np.argsort(individual[:,1])[:10]])
for start in starts:
    r=E[start].copy(); ids=[int(start)]; w=np.array([1.]); prev=float(score(measures(r)))
    for step in range(7):
        trial=None
        for alpha in (.10,.20,.35,.50):
            m=measures((1-alpha)*r+alpha*E); k=int(np.argmin(score(m))); cost=float(score(m[k]));
            if trial is None or cost<trial[0]: trial=(cost,alpha,k,m[k])
        cost,alpha,k,err=trial
        if cost>=prev-1e-5: break
        w=np.r_[(1-alpha)*w,alpha]; ids.append(k); r=(1-alpha)*r+alpha*E[k]; prev=cost
        if cost<best[0]: best=(cost,np.array(ids),w.copy(),err)
_,ids,w,err=best
print('selected',err.round(6), 'baseline',err0.round(6),'models',[(configs[i],round(float(a),5)) for i,a in zip(ids,w)],flush=True)
if max(err[:2]) >= max(err0[:2]):
    raise SystemExit('No ordered-split improvement; baseline retained')
path=Path('/holdout_task/batch_solver.py')
old=Path('/tmp/batch_base.py').read_text().replace('def solve_batch(study):','def _joint_prediction(study):')
models={}
for i,a in zip(ids,w): models[configs[i]]=models.get(configs[i],0.)+float(a)
wrapper='''

def solve_batch(study):
    result = _joint_prediction(study)
    x, xq = study['train']['x'], study['test']['x']
    xx = np.vstack((x,xq))
    coordinates = _coordinates(study,xx)
    d = _aff(xx)
    q = np.column_stack([d]+[xx[:,i]*xx[:,j] for i in range(4) for j in range(i,4)])
    i, j = np.arange(len(x)), np.arange(len(x),len(xx))
    y = _BASE_WEIGHT * result['y_hat']
    for (idx,warp,mode,kind,ell,lam,order,link,cap), weight in _READOUT_MODELS:
        k = _kernel(_geometry(coordinates[idx],warp,mode),kind,ell)
        k.flat[::len(xx)+1] += lam
        y += weight*_fit_kernel(k,d if order==1 else q,study['train']['y'],i,j,link,cap)
    last = result['state_pred'][:,-1]
    result['y_hat'] = y
    result['diagnostics_vec'] = np.array([y.mean(),y.std(),last.mean(),last.std()])
    return result
'''
path.write_text(old+source+'\n\n_BASE_WEIGHT = '+repr(models.pop(None,0.))+'\n_READOUT_MODELS = '+repr(list(models.items()))+'\n'+wrapper)
