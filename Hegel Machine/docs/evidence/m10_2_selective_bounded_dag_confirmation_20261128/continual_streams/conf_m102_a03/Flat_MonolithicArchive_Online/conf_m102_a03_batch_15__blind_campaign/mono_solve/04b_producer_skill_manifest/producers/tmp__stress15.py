import pickle, runpy, numpy as np
from scipy.optimize import minimize_scalar, least_squares
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=(tr[k] for k in ('x','y','traj')); t=s['eval_times']; fit=m['_fit']
z=m['_decode'](x,m['_calibration'](s['panel_a'])); b=m['_operator'](s['panel_b'],z); h,q=m['_readouts'](b)
N=len(x); rng=np.random.default_rng(624); ix=np.arange(N)
splits=[('fixed',ix[52:])]+[(f'random{i}',rng.permutation(N)[52:]) for i in range(30)]
for k,u in [('observed',x),('decoded',z),('operator',b)]:
 for j,v in enumerate(u.T):
  a=np.argsort(v); splits.extend([(f'{k}{j}_low',a[:20]),(f'{k}{j}_high',a[-20:])])
for k,u in [('decoded',z),('operator',b)]:
 a=np.argsort(np.linalg.norm(u-np.median(u,axis=0),axis=1)); splits.append((f'{k}_radial',a[-20:]))
compact=np.c_[b,b[:,0]*b[:,2]+b[:,1]*b[:,3]]
pairs=np.column_stack([b[:,i]+sgn*c*b[:,j] for i in range(4) for j in range(i+1,4) for sgn in (-1,1) for c in (.5,1.,2.)]); factors=np.c_[b,pairs]
# Every pairwise square is already in the full quadratic span; test for a rational denominator using only inner-fold losses.
assert np.max(abs(q@fit(q,pairs**2)-pairs**2))<1e-9
res={}
for label,te in splits:
 a=np.setdiff1d(ix,te); target=y[a]; traj=T[a]
 def linear(d): return d[te]@fit(d[a],target)
 inner=[(np.setdiff1d(np.arange(len(a)),f),f) for f in np.array_split(np.random.default_rng(1538).permutation(len(a)),4)]
 def cv(d):
  d=d[a]; return np.mean([np.mean(abs(d[v]@fit(d[u],target[u])-target[v])) for u,v in inner])
 best=(cv(compact),-1,0.)
 for j,w in enumerate(factors.T):
  w=w/max(np.sqrt(np.mean(w[a]**2)),1e-10)
  for alpha in (.001,.01,.05,.2,.8):
   d=compact/(1+alpha*w[:,None]**2); score=cv(d)
   if score<best[0]:best=(score,j,alpha)
 if best[1]<0: rat=linear(compact)
 else:
  w=factors[:,best[1]]; w=w/max(np.sqrt(np.mean(w[a]**2)),1e-10); rat=linear(compact/(1+best[2]*w[:,None]**2))
 def temporal(D):
  S=fit(D[a],traj)
  def basis(p):return -np.expm1(-np.outer(t,p))
  def err(p):
   B=basis(p);return (D[a]@fit(B,S.T).T@B.T-traj).ravel()
  p=least_squares(err,[.41,1.82],bounds=([.05,.85],[.83,5.])).x
  B=basis(p);return D[te]@fit(B,S.T).T@B.T,p
 state,p=temporal(b); base=m['_forecast_state'](h[te],h[a],t,traj)
 mets=np.array([*[np.mean(abs(f-y[te])) for f in (linear(q),linear(compact),rat)],np.mean(abs(base-T[te])),np.mean(abs(state-T[te])),np.mean(abs(base-T[te]),0).max(),np.mean(abs(state-T[te]),0).max()]); res[label]=(mets,best[2],p)
for label in ['fixed','decoded_radial','operator_radial']: print(label,res[label])
for filt in [lambda k:k.startswith('random'),lambda k:not k.startswith('random')]:
 sub=[(k,v) for k,v in res.items() if filt(k)]; J=np.array([v[0] for _,v in sub]); print('mean',J.mean(0),'worst',J.max(0),'worst names',[sub[i][0] for i in np.argmax(J,0)],'rational adopted',sum(v[1]>0 for _,v in sub),'of',len(sub))
print('training fields',list(tr),'study fields',list(s)); print('compact coefficients',fit(compact[:52],y[:52])); print('full-data compact residual RMSE',np.sqrt(np.mean((compact@fit(compact,y)-y)**2)))
