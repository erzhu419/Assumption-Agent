import sys,pickle,numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from itertools import combinations,combinations_with_replacement
sys.path.insert(0,'/holdout_task')
import node_00 as a,node_01 as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ids=np.arange(n); g=[x[:n] for x in b._features(s,a.run_node({'__input__':s}))[0]]; z=g[0]; Y=np.c_[tr['y'],tr['traj']]; H=b._time(s['eval_times'],tr['traj']); y=np.c_[tr['y'],b._fit(H,tr['traj'].T).T]; print('label summaries',np.quantile(y,[0,.1,.5,.9,1],axis=0),np.corrcoef(y.T),flush=True)
folds=[ids[-k:] for k in [18,20,24]]+list(np.array_split(np.random.default_rng(81).permutation(n),4))
D=lambda x:np.c_[np.ones(len(x)),x]; L=b._fit
# Compare coordinatewise B warp with tied even/odd and rational alternatives.
p=s['panel_b']; u,v=p['u'],p['v']; specs={
 'cubic':(lambda x,q:x+q[0]*x**3,[-1.]), 'rat':(lambda x,q:x+q[0]*x/(1+q[1]*x*x),[-1.4,.7]),
 'rat-even':(lambda x,q:x+q[0]*x/(1+q[1]*x*x)+q[2]*x*x,[-1.4,.7,0]),
 'exp-even':(lambda x,q:x+q[0]*x*(-np.expm1(-(q[1]*x)**2))+q[2]*x*x,[-11,1,0]),
 'sin-even':(lambda x,q:x+q[0]*np.sin(q[1]*x)+q[2]*x*x,[-.4,2.85,0]),
 'cubic5':(lambda x,q:x+q[0]*x**3+q[1]*x**5,[-2,.6]),
 'cubic-even':(lambda x,q:x+q[0]*x**3+q[1]*x*x,[-2,0]),
 'tanh-cross-axis':(lambda x,q:x+q[0]*np.tanh(q[1]*x)+q[2]*x*np.roll(x,1,axis=1),[-1,1,0]),
 'exp-cross-axis':(lambda x,q:x+q[0]*x*(-np.expm1(-(q[1]*x)**2))+q[2]*x*np.roll(x,1,axis=1),[-11,1,0]),
 'sine_cross':(lambda x,q:x+q[0]*np.sin(q[1]*x)+q[2]*np.sin(q[1]*x)*np.roll(x,1,axis=1),[-.4,2.85,0])}
for name,(f,q) in specs.items():
 def res(q):
  X=D(f(u,q)); return (X@L(X,v)-v).ravel()
 r=least_squares(res,q,max_nfev=150); print('B',name,r.x,np.sqrt(np.mean(r.fun**2)),flush=True)
# Feature-weighted linear/quadratic kernels and nonlinear links on measured amplitudes.
paths={'r':[0,1,3,5],'j':list(range(6)),'r2':[0,1,3,5,2],'c':[0,2,4],'r3':[0,1,3,5,4]}
for name,which in paths.items():
 best=[]
 for bz in [.15,.5,1,2,5]:
  for bc in [.25,1,4]:
   for mode in ('q','q3','rbf','addq','cross'):
    Q=[]
    for va in folds:
     ti=np.setdiff1d(ids,va); G=[(g[i]-g[i][ti].mean(0))/np.maximum(g[i][ti].std(0),1e-9) for i in which]; ps=[x@x.T/4 for x in G]; P=sum(p*w for p,w in zip(ps,[bz,1]+[bc]*4))/len(ps); d=np.maximum(P.diagonal()[:,None]+P.diagonal()-2*P,0); K=1+P
     if mode=='q': K=K+P**2
     if mode=='q3': K=K+P**2+.05*P**3
     if mode=='rbf': K=K+np.exp(-d/32)
     if mode=='addq': K=K+sum(p*p for p in ps)/len(ps)+.1*P**2
     if mode=='cross': K=K+ps[0]*sum(ps[1:])/len(ps)+.3*P**2
     Q.append((va,ti,K))
    for lam in [1e-7,1e-5,.0003,.003,.03]:
     for link in [0,1,2]:
      es=[]
      for va,ti,K in Q:
       mu=y[ti].mean(0); sc=np.max(abs(y[ti]-mu),axis=0)*(1.04 if link==1 else 1.33); u=(y[ti]-mu)/sc
       target=y[ti] if link==0 else np.arctanh(u); pred=K[np.ix_(va,ti)]@solve(K[np.ix_(ti,ti)]+lam*np.eye(len(ti)),target,assume_a='pos'); pred=pred if link==0 else mu+sc*np.tanh(pred)
       e=abs(np.c_[pred[:,0],pred[:,1:]@H.T]-Y[va]); es.append([e[:,0].mean(),e[:,1:].mean()]);
      best.append((max(np.max(es),1.5*np.max(es[0])),bz,bc,mode,lam,link,np.round(es[:3],5).tolist(),np.round(np.max(es,axis=0),5).tolist()))
 print('K',name,sorted(best)[:4],flush=True)
