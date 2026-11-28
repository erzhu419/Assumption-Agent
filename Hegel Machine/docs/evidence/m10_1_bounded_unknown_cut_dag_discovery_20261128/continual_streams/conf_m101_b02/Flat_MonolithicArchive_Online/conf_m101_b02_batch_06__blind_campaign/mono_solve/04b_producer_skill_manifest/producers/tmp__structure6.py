from readout6 import *
x=s['train']['x']; ns=dict(zip('abcd',nodes(x))); Y=np.c_[s['train']['y'],s['train']['traj']]; t=s['eval_times']; n=52
np.set_printoptions(precision=5,suppress=True)
print('t',t,'telemetry',s['telemetry_vec'])
for r in ('a c ac','a c d ac','a b c d ac'):
 g,names=design(ns,r); c=fit(g[:n],Y[:n]); e=Y-g@c
 print(r,'RMSE',np.sqrt(np.mean(e[:n]**2,0)),np.sqrt(np.mean(e[n:]**2,0)),'sing residuals',np.linalg.svd(e[:n,1:],compute_uv=False))
 if r=='a b c d ac':
  print('coefs'); print('\n'.join(str((k,list(w))) for k,w in zip(names,c[:,[0,1,4,-1]].round(5))))
  ids=np.argsort(abs(e[n:,0]))[-5:]+n; print('worst',np.c_[x[ids],ns['a'][ids],Y[ids,0],e[ids,0],e[ids,-1]])
T=Y[:n,1:].T
for mode in ('two','three','intercept','damped'):
 def bas(q):
  if mode=='intercept': return np.column_stack((np.ones(len(t)),-np.expm1(-t[:,None]*q)))
  if mode=='damped': return np.column_stack((-np.expm1(-t[:,None]*q[:2]),np.exp(-q[2]*t)*np.sin(q[3]*t)))
  return -np.expm1(-t[:,None]*q)
 def res(q):
  g=bas(q); return (g@fit(g,T)-T).ravel()
 for q0 in {'two':[[.4,2]],'three':[[.4,1,2],[.05,.45,2],[.45,2,5]],'intercept':[[.4,2]],'damped':[[.4,2,.6,1.5],[.4,2,.6,3]]}[mode]:
  q=least_squares(res,q0,bounds=(.005,12),gtol=1e-11,max_nfev=400).x
  print(mode,q,np.std(res(q)), np.max(abs(res(q))))
print('summary observed',s['train']['summary'],[Y[:,0].mean(),Y[:,0].std(),Y[:,-1].mean(),Y[:,-1].std()])
