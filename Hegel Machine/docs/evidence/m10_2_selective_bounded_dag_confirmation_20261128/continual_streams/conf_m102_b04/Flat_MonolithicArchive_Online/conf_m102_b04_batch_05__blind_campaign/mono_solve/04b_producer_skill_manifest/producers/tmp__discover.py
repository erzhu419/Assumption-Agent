import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True)
def a(x):return np.c_[np.ones(len(x)),x]
def ls(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def poly(x,d):return a(np.column_stack([np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),k)]))
pa=s['panel_a']; u=pa['u']; v=pa['v']; ok=np.linalg.norm(v-u,axis=1)<5
for name,u,v in [('a',u[ok],v[ok]),('c1',s['panel_c']['u'],s['panel_c']['v1']),('c2',s['panel_c']['u'],s['panel_c']['v2'])]:
 print(name, 'range',v.min(0),v.max(0));
 for d in [1,2,3]:
  g=poly(u,d); print('poly',d,np.sqrt(np.mean((g@ls(g,v)-v)**2)), 'co',np.round(ls(g,v),3) if d==2 and name=='a' else '')
 U=a(u)
 for shift in [False,True]:
  for fname,f in [('tanh',np.tanh),('sin',np.sin),('rational',lambda w:w/np.sqrt(1+w*w)),('atan',np.arctan),('square',lambda w:w*w)]:
   rms=[]; pars=[]
   for j in range(4):
    y=v[:,j];
    def res(p):
     F=f(U@p[:5]); G=np.c_[U,F] if shift else np.c_[U[:,0],F]; return G@ls(G,y)-y
    p0=ls(U,y); sol=min((least_squares(res,p0*k,max_nfev=200) for k in [.5,1,2]),key=lambda p:np.linalg.norm(p.fun)); rms.append(np.std(sol.fun)); pars.append(sol.x)
   print(fname,shift,rms, np.round(pars,2) if max(rms)<.025 else '')
tr=s['train']; t=s['eval_times']; z=tr['traj'];
for name,b in [('exp',lambda p:-np.expm1(-t[:,None]*np.exp(p))),('log',lambda p:np.log1p(t[:,None]*np.exp(p))),('rat',lambda p:t[:,None]/(1+t[:,None]*np.exp(p))),('power',lambda p:t[:,None]**p)]:
 def rr(p):
  g=b(p);return (g@ls(g,z.T)-z.T).ravel()
 fit=least_squares(rr,[0.3,1.5],max_nfev=400); print(name,fit.x,np.std(fit.fun));
