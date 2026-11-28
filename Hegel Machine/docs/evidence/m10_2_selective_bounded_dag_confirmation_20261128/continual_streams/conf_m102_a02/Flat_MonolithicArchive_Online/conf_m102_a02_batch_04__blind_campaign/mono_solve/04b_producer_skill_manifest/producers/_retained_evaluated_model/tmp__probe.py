import numpy as np,pickle,runpy
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True,linewidth=150)
a=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a02_batch_03_solver.py'); w=a['_calibration'](s['panel_a']); z=a['_inverse'](s['train']['x'],w)
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x):return np.c_[np.ones(len(x)),x]
def nxt(x):return np.roll(x,-1,axis=1)
def poly(z,d):return PolynomialFeatures(d).fit_transform(z)
funs={'quad':lambda x:x*x,'cyc':lambda x:x*nxt(x),'cube':lambda x:x**3,'rat':lambda x:x/(1+x*x),'gauss':lambda x:x*np.exp(-x*x),'tanh':np.tanh,'sin':np.sin,'crossdiff':lambda x:(x-nxt(x))**2}
for p in ['panel_b','panel_c']:
 u=s[p]['u']
 for key,v in s[p].items():
  if key=='u':continue
  scores=[]
  from itertools import combinations
  for d in range(3):
   for keys in combinations(funs,d):
    F=np.column_stack([aff(u)]+[funs[k](u) for k in keys]); c=ls(F,v); h=np.sum(F*np.linalg.pinv(F).T,axis=1); scores.append((np.sqrt(np.mean(((F@c-v)/(1-h[:,None]))**2)),keys,c))
  scores.sort(key=lambda i:i[0]); print(p,key,[(q[0],q[1]) for q in scores[:6]],'coef',scores[0][2]);
t=s['eval_times']; S=s['train']['traj']; Y=s['train']['y'];
def B(r):return -np.expm1(-t[:,None]*r)
for k in (1,2,3):
 r=least_squares(lambda r:(B(r)@ls(B(r),S[:52].T)-S[:52].T).ravel(),[2,.4,.1][:k],bounds=(.001,20)).x; print('rates',r,'rms',np.sqrt(np.mean((B(r)@ls(B(r),S.T)-S.T)**2)))
 if k==2: amp=ls(B(r),S.T).T; BB=B(r)
print('amp range',amp.min(0),amp.max(0),'cal ranges',z.min(0),z.max(0))
for name,F in [('raw',aff(s['train']['x'])),('latent',aff(z)),('poly2',poly(z,2)),('poly3',poly(z,3))]+[(k,np.c_[aff(z),f(z)]) for k,f in funs.items()]:
 c=ls(F[:52],np.c_[Y,amp][:52]); pred=F[52:]@c; print(name,'y',np.mean(abs(pred[:,0]-Y[52:])),'state',np.mean(abs(pred[:,1:]@BB.T-S[52:])),'amp train',np.std(F[:52]@c-np.c_[Y,amp][:52],axis=0))
