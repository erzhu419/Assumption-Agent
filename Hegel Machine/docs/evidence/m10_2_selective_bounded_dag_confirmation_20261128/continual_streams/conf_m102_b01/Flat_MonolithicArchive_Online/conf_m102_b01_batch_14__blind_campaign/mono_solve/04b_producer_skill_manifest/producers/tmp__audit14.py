import importlib.util, pickle, numpy as np
from itertools import combinations
p='/holdout_task/batch_solver.py'
spec=importlib.util.spec_from_file_location('solver',p); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; obs=tr['traj']; t=s['eval_times']; n=len(x)
print('study and train keys',sorted(s),sorted(tr),'unique times',len(np.unique(t)))
a=s['panel_a']; u,v=a['u'],a['v']; affine=m._aff(u); basis=m._cal_basis(u)
keep=np.ones(len(u),bool)
for _ in range(8):
 c=np.stack([m._ls(basis[keep,j],v[keep,j]) for j in range(4)])
 e=np.linalg.norm((basis*c).sum(axis=-1)-v,axis=1); keep=e<=max(8*np.median(e),.002)
idx=np.flatnonzero(keep); out=[]
for k in idx:
 ii=idx[idx!=k]; ca=m._ls(affine[ii],v[ii]); cn=np.stack([m._ls(basis[ii,j],v[ii,j]) for j in range(4)])
 out.append([affine[k]@ca-v[k],(basis[k]*cn).sum(-1)-v[k]])
print('Panel A clean-row leave-one-out RMSE (affine, nonlinear)',np.sqrt(np.mean(np.array(out)**2,axis=(0,2))), 'gross residuals',e[~keep]); assert np.sqrt(np.mean(np.array(out)[:,1]**2)) < .001
q=m._latent(s,x); a=m._aff(q); d=np.c_[a,q[:,0]*q[:,2],q[:,1]*q[:,3]]
bases={'quadratic':d,'affine':a}
for power in [.05,.22,1.]:
 for name,g in [('coordinate',q),('sum',np.column_stack([q[:,i]+q[:,j] for i,j in combinations(range(4),2)])),('difference',np.column_stack([q[:,i]-q[:,j] for i,j in combinations(range(4),2)]))]:
  bases[f'plus_{name}_rational_{power}']=np.c_[d,g/(1+power*g*g)]
  for j in range(g.shape[1]):bases[f'quotient_{name}_{j}_{power}']=d/(1+power*g[:,j,None]**2)
bases['plus_distinct_triples']=np.c_[d,*[np.prod(q[:,j],axis=1) for j in combinations(range(4),3)]]
rng=np.random.default_rng(182); splits=[np.arange(n)] + [rng.permutation(n) for _ in range(40)]
for axis in [0,1,2,3]:
 i=np.argsort(x[:,axis]);splits.extend([i,i[::-1]])
splits.append(np.argsort(np.linalg.norm(x,axis=1)))
def i_mae(d,i,j):return np.mean(np.abs(d[j]@m._ls(d[i],y[i])-y[j]))
errs=[]; picks=[]
for split in splits:
 i,j=split[:52],split[52:]; inner=np.array_split(i,4)
 key=min(bases,key=lambda k:np.mean([i_mae(bases[k],np.setdiff1d(i,f),f) for f in inner])); picks.append(key)
 local={**s,'train':{**tr,**{k:tr[k][i] for k in ('x','y','traj')}},'test':{'x':x[j]}}; pred=m.solve_batch(local)
 d0=np.abs(pred['y_hat']-y[j]); dt=np.abs(pred['state_pred']-obs[j]); z=pred['state_pred'][:,-1]
 np.testing.assert_allclose(pred['diagnostics_vec'],[pred['y_hat'].mean(),pred['y_hat'].std(),z.mean(),z.std()]); errs.append([d0.mean(),dt.mean(),dt.mean(0).max(),i_mae(bases[key],i,j)])
E=np.array(errs);print('scalar / state / worst time / nested grammar scalar'); print('required',E[0]); print('means',E.mean(0),'max',E.max(0),'max ordered',E[-9:].max(0)); print('selected grammar counts',[(k,picks.count(k)) for k in sorted(set(picks))]); assert E[:,:3].max()<.038
