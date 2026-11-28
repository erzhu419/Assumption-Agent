import sys,pickle,numpy as np, importlib.util
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=140)
spec=importlib.util.spec_from_file_location('old','/tmp/hegel_continual_archive/conf_m102_b02_batch_05_solver.py'); old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
z,b=old.parts(s); f=old.times(s); t=s['train']; n=len(t['y']); T=np.c_[t['y'],old.fit(f,t['traj'].T).T]; sl=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(n),j),j) for j in np.array_split(np.random.default_rng(41).permutation(n),4)]
def D(x): return np.c_[np.ones(len(x)),x]
def poly(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,cs],axis=1) for k in range(1,d+1) for cs in combinations_with_replacement(range(x.shape[1]),k)])
def assess(a):
 out=[]
 for i,j in sl:
  e=a[j]@old.fit(a[i],T[i])-T[j];out.append([np.abs(e[:,0]).mean(),np.abs(e[:,1:]@f.T).mean()]);
 return np.round(out[0],5),np.round(np.mean(out[1:],axis=0),5)
for k in ['panel_a','panel_b','panel_c']:
 p=s[k];u=p['u']; print(k)
 for key in set(p)-{'u'}:
  v=p[key]; keep=np.ones(len(u),bool)
  if k=='panel_a':keep=old.cal(s,u)[2]<2
  for nam,A in [('aff',D(u)),('AF',old.AF(u)),('pow2',poly(u,2)),('sinh',D(np.sinh(u))),('asinh',D(np.arcsinh(u))),('tanh',D(np.tanh(u)))]:
   A=A[keep];y=v[keep];w=old.fit(A,y);e=y-A@w; lev=np.sum(A*np.linalg.pinv(A).T,axis=1);print(key,nam,'rmse',np.std(e),'LOO',np.std(e/np.maximum(1-lev,.0001)[:,None]))
  if k!='panel_a':
   H=lambda q: D(np.sign(u)*np.maximum(abs(u)-q[0],0)**q[1]); sol=least_squares(lambda q:(H(q)@old.fit(H(q),v)-v).ravel(),[.3,1.5],bounds=([0,.2],[.8,5]));print('threshold',sol.x,np.std(sol.fun));print('range',u.min(0),u.max(0))
print('T basis errors',np.abs(T[:,1:]@f.T-t['traj']).mean(),T[:8]); print('z',z.min(0),z.max(0))
for name,a in [('raw',np.r_[t['x'],s['test']['x']]),('z',z),('b',b),('zb',np.c_[z,b]),('asinhz',np.arcsinh(z)),('asinhzb',np.arcsinh(np.c_[z,b]))]:
 for d in range(1,4 if a.shape[1]<5 else 3): print(name,d,assess(poly(a,d)))
for i,j in sl:
 r=dict(s,train=dict(x=t['x'][i],y=t['y'][i],traj=t['traj'][i]),test=dict(x=t['x'][j])); yp,tp=old.predict(r); print('archive',np.abs(yp-t['y'][j]).mean(),np.abs(tp-t['traj'][j]).mean())
print('corr',np.corrcoef(np.c_[z[:n],b[:n],T].T)[-3:])
