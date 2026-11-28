import importlib.util, pickle
from itertools import combinations, combinations_with_replacement
import numpy as np
p='/holdout_task/'
spec=importlib.util.spec_from_file_location('batch_solver',p+'batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(p+'public_batch.pkl','rb') as f: s=pickle.load(f)[0]['study']
tr=s['train']; x=m._coordinates(s,tr['x']); y=tr['y']; z=tr['traj']; n=len(x); ids=np.arange(n); base=m._scalar_design(x)
q=np.column_stack([x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)])
forms=[(f'x{i}',x[:,i]) for i in range(4)]+[(f'x{i}{sign}x{j}',(x[:,i]+c*x[:,j])/np.sqrt(2)) for i,j in combinations(range(4),2) for c,sign in [(1,'+'),(-1,'-')]]
rat=[('none',base)]+[(f'{name}:{a:g}',base/(1+a*f[:,None]**2)) for name,f in forms for a in (.002,.01,.04,.16,.64)]
poly=[('tied',base),('all-quadratic',np.c_[np.ones(n),x,q])]+[(f'tied+{i}',np.c_[base,q[:,i]]) for i in range(q.shape[1])]
def pred(d,it,va): return d[va]@m._fit(d[it],y[it])
def select(opts,it):
    folds=np.array_split(np.random.default_rng(37).permutation(it),4)
    return min(opts,key=lambda a: np.mean([np.mean(abs(y[v]-pred(a[1],np.setdiff1d(it,v),v))) for v in folds]))
slices={'last16':[ids[56:]],'blocked':[ids[i:i+16] for i in range(0,n,16)],'radius':[np.argsort(np.linalg.norm(x,axis=1))[-16:]],'tails':[np.argsort(x[:,j])[sl] for j in range(4) for sl in (slice(0,12),slice(-12,None))]}
for name,folds in slices.items():
    scores=[]; chosen=[]; horizon=[]; terminal=[]; diag=[]
    for va in folds:
        it=np.setdiff1d(ids,va); a=select(rat,it); b=select(poly,it); chosen.append((a[0],b[0]))
        h=dict(s,train={k:tr[k][it] for k in ('x','y','traj')},test={'x':tr['x'][va]}); r=m.solve_batch(h)
        scores.append([np.mean(abs(y[va]-v)) for v in (r['y_hat'],pred(a[1],it,va),pred(b[1],it,va))]); horizon.append(np.mean(abs(z[va]-r['state_pred']),axis=0)); terminal.append(horizon[-1][-1])
        e=z[va,-1]; diag.append(abs(r['diagnostics_vec']-[y[va].mean(),y[va].std(),e.mean(),e.std()]))
    print(name,'scalar mean/max [base,rational,quadratic]',np.mean(scores,axis=0),np.max(scores,axis=0),'selections',chosen,'state mean/worst/terminal worst',np.mean(horizon),np.max(horizon),np.max(terminal),'diagnostic worst',np.max(diag,axis=0))
# Quantify calibration support and whether a single panel point dominates the test features.
a=s['panel_a']; d=np.max(abs(a['v']-a['u']),axis=1); ref=m._coordinates(s,s['test']['x']); shifts={}
for k in ('panel_a','panel_b'):
    panel=s[k]; jx=np.arange(len(panel['u'])); shifts[k]=max(np.max(np.abs(ref-m._coordinates(dict(s,**{k:{j:v[jx!=i] for j,v in panel.items()}}),s['test']['x']))) for i in jx)
print('calibration threshold separation',np.max(d[d<1]),np.min(d[d>=1]),'panel jackknife worst test coordinate change',shifts)
print('study fields',sorted(s),'train fields',sorted(tr),'telemetry shape',s['telemetry_vec'].shape)
r=m.solve_batch(s); ntest=len(s['test']['x']); nt=len(s['eval_times']); assert set(r)=={'y_hat','state_pred','diagnostics_vec'}
for k,shape in zip(('y_hat','state_pred','diagnostics_vec'),((ntest,),(ntest,nt),(4,))): assert isinstance(r[k],np.ndarray) and r[k].shape==shape and np.all(np.isfinite(r[k]))
a=r['y_hat']; b=r['state_pred'][:,-1]; np.testing.assert_array_equal(r['diagnostics_vec'],[a.mean(),a.std(),b.mean(),b.std()]); print('Isolated solver contract and all diagnostics passed')
