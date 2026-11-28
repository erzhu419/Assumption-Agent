import importlib.util, pickle
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('candidate',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; x,y=(s['train'][k] for k in ('x','y')); n=len(x)
a,g=m._calibration(s['panel_a']); b=m._operator(s['panel_b'],(x-a)/g); F=m._quadratic(b)
def ls(f,y): return np.linalg.lstsq(f,y,rcond=None)[0]
def select(f,y):
    cols=list(range(5)); remaining=list(range(5,f.shape[1])); n=len(y)
    def bic(cols):
        a=f[:,cols]; r=y-a@ls(a,y)
        return n*np.log(max(np.mean(r*r),1e-30))+len(cols)*np.log(n)
    while remaining:
        j=min(remaining,key=lambda j:bic(cols+[j]))
        if bic(cols+[j])>=bic(cols): break
        cols.append(j); remaining.remove(j)
    return sorted(cols)
groups={'prescribed':[np.arange(52,72)],'blocked':[np.arange(i,i+20) for i in (0,13,26,39,52)],'random':[np.random.default_rng(i).permutation(n)[:20] for i in range(30)],'coordinate_tails':[np.argsort(sign*x[:,j])[-20:] for j in range(4) for sign in (-1,1)],'radial_tail':[np.argsort(np.linalg.norm(x,axis=1))[-20:]]}
for name,slices in groups.items():
    metrics=[]; picked={}
    for j in slices:
        i=np.setdiff1d(np.arange(n),j); cols=select(F[i],y[i]); key=tuple(cols[5:]); picked[key]=picked.get(key,0)+1
        dense=F[j]@ls(F[i],y[i]); q=F[:,cols]; sparse=q[j]@ls(q[i],y[i]); errors=np.abs(np.column_stack((dense,sparse))-y[j,None]); metrics.append(np.r_[errors.mean(0),errors.max(0)])
    print(name, 'dense/sparse MAE then dense/sparse max abs; worst',np.max(metrics,axis=0),'mean',np.mean(metrics,axis=0),'selected quadratic indices',picked)
