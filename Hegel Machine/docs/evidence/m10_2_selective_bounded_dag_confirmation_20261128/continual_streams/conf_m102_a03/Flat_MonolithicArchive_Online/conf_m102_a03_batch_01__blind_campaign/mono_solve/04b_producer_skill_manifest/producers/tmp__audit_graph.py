import importlib.util, pickle
import numpy as np
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
a=s['panel_a']; u,v=a['u'],a['v']; n=len(u); h=np.tile(np.eye(4),(n,1)); x=u.ravel(); j=np.roll(u,-1,axis=1).ravel(); D=np.column_stack((h,h*x[:,None],j,x*j,x**3)); clean=np.linalg.norm(v-u,axis=1)<3
for cols in (9,10,11):
 errs=[]
 for i in np.flatnonzero(clean):
  keep=np.repeat(clean & (np.arange(n)!=i),4); c=m._fit(D[keep,:cols],v.ravel()[keep]); errs.extend(D[4*i:4*i+4,:cols]@c-v[i])
 print('panel A clean leave-one-row-out columns',cols,'RMSE',np.sqrt(np.mean(np.square(errs))), 'max abs',max(map(abs,errs)))
tr=s['train']; y,T=tr['y'],tr['traj']; z=m._decode(tr['x'],m._calibration(a)); b=s['panel_b']; w=m._affine(z)@m._fit(m._affine(b['u']),b['v']); base=m._readout_features(w); q=m._quadratic(w) if hasattr(m,'_quadratic') else np.column_stack((m._affine(w),*[w[:,i]*w[:,j] for i in range(4) for j in range(i,4)]))
models={'bilinear':base,'intercept':np.c_[np.ones(len(w)),base],'quadratic':q}
for i in range(4):
 for j in range(i+1,4):
  for sign in (-1,1):
   for c in (.5,1.,2.):
    d=w[:,i]+sign*c*w[:,j]; d=d/np.sqrt(1+c*c)
    models[f'diff:{i}:{j}:{sign}:{c}']=np.c_[base,d*d]
    for alpha in (.005,.02,.1,.4): models[f'quotient:{i}:{j}:{sign}:{c}:{alpha}']=base/(1+alpha*d[:,None]**2)
idx=np.arange(len(y)); rng=np.random.default_rng(9823)
splits={'specified':idx[52:],'first':idx[:20],'radial':np.argsort(np.linalg.norm(z,axis=1))[-20:]}
for k in range(4):
 for sign in (-1,1): splits[f'tail{k}:{sign}']=np.argsort(sign*z[:,k])[-20:]
for k in range(25): splits[f'random{k}']=rng.permutation(idx)[:20]
record={k:[] for k in ('bilinear','intercept','quadratic','best_diff','best_quotient','trajectory','worst_time','terminal')}; selected={}
for name,te in splits.items():
 train=np.setdiff1d(idx,te); inner=np.array_split(rng.permutation(train),4)
 chosen={}
 for family in ('diff','quotient'):
  def score(D):
   return np.mean([np.mean(abs(D[v]@m._fit(D[np.setdiff1d(train,v)],y[np.setdiff1d(train,v)])-y[v])) for v in inner])
  k=min((k for k in models if k.startswith(family+':')),key=lambda k:score(models[k])); selected[k]=selected.get(k,0)+1; chosen['best_'+family]=models[k]
 for k,D in ([(k,models[k]) for k in ('bilinear','intercept','quadratic')]+list(chosen.items())): record[k].append(np.mean(abs(D[te]@m._fit(D[train],y[train])-y[te])))
 d=abs(w[te]@m._relaxation(w[train],T[train],s['eval_times'])-T[te]); record['trajectory'].append(d.mean()); record['worst_time'].append(d.mean(0).max()); record['terminal'].append(d[:,-1].mean())
for k, e in record.items(): print(k, 'specified',round(e[0],7),'mean',round(float(np.mean(e)),7),'worst',round(max(e),7),'slice',list(splits)[int(np.argmax(e))]); assert max(e)<.038 or k not in ('bilinear','trajectory','worst_time','terminal')
print('inner-selected alternatives, most frequent:',sorted(selected.items(),key=lambda item:-item[1])[:8]); print('public fields:',sorted(s),'training fields:',sorted(tr),'study telemetry shape:',np.shape(s['telemetry_vec']))
