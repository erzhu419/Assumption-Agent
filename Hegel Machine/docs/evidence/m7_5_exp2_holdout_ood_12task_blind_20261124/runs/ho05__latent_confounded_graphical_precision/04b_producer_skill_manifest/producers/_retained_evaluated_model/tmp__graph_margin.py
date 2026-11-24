import numpy as np
from scipy.linalg import eigh
from graph_probe import generate
from deconfound_graph import _profile, _fit, recover_sparse_precision_support

def candidates(x):
    n,p=x.shape; z=x-x.mean(0); c=z.T@z/n
    d,u=eigh(c); w=u[:,:-2]**2
    scale=np.sqrt((w@(1/d[:-2]))/w.sum(1)); e=_profile(c*np.outer(scale,scale))
    ii,jj=np.triu_indices(p,1); pc=[]
    for alpha in (.70,.8775,1.10,1.35):
        s=_fit(e,np.eye(p),penalty=alpha/np.sqrt(n))
        pc.append(np.abs(s[ii,jj])/np.sqrt(s[ii,ii]*s[jj,jj])*np.sqrt(n))
    return np.array(pc)

def f1(y,z):
    return 2*np.count_nonzero(y&z)/(y.sum()+z.sum())

ii,jj=np.triu_indices(24,1)
pairs=[(.70,2.55),(.8775,2.5),(1.10,2.25),(1.35,2.05)]
public=np.load('/holdout_task/public_graph.npz'); raw=public['X']; a=public['gt_adj']; xx=raw-raw.mean(0)
base=recover_sparse_precision_support(raw); rng=np.random.default_rng(42); perm=rng.permutation(24); factors=np.exp(rng.uniform(-.75,.75,24)); alt=recover_sparse_precision_support(raw[:,perm]*factors+3)
scaled=alt['sparse_precision']*np.outer(factors,factors); print('public equivariance: adjacency differences, relative precision difference',np.count_nonzero(alt['adjacency']!=base['adjacency'][np.ix_(perm,perm)]),np.linalg.norm(scaled-base['sparse_precision'][np.ix_(perm,perm)])/np.linalg.norm(scaled))
for mode in ('public','coherent','weak','hubs','hetero','random','factor2_weak','hetero_coherent'):
    vals=[]; cases=range(50,86) if mode!='public' else range(1)
    for seed in cases:
        if mode=='public': x,y=raw,a
        elif mode in ('factor2_weak','hetero_coherent'):
            subset='hetero' if mode=='hetero_coherent' else 'sign'; _,y,s=generate(seed,subset)
            r=np.random.default_rng(seed+800); b=r.normal(size=(24,2))*.9
            if mode=='factor2_weak': b[:,1]*=.26
            else: b[:5]*=2.5
            x=r.normal(size=(900,24))@np.linalg.cholesky(np.linalg.inv(s)).T+r.normal(size=(900,2))@b.T
        else: x,y,_=generate(seed,mode)
        y=y[ii,jj]; q=candidates(x); votes=q>=np.array([k for _,k in pairs])[:,None]; zz=[*votes,votes.sum(0)>=3,votes[:3].sum(0)>=2]
        row=[f1(y,z) for z in zz]; vals.append(row)
    vals=np.array(vals); worst=np.argmin(vals[:,1]); print(mode,'F1 min',np.round(vals.min(0),3),'mean',np.round(vals.mean(0),3),'at baseline worst seed',cases[worst],np.round(vals[worst],3))
