import numpy as np, pickle, importlib.util
from scipy.linalg import svd
spec=importlib.util.spec_from_file_location('a','/tmp/hegel_continual_archive/conf_m102_b03_batch_01_solver.py'); a=importlib.util.module_from_spec(spec); spec.loader.exec_module(a)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; T=s['train']['traj']; n=len(y); z,l,q,r,p,c=a.features(s,s['train']['x']); dot=(q*r).sum(1); cross=(p[:,:,None]*c[:,None,:]).reshape(n,-1); mix=(q[:,:,None]*r[:,None,:]).reshape(n,-1); zp=(z[:,:,None]*p[:,None,:]).reshape(n,-1); zc=(z[:,:,None]*c[:,None,:]).reshape(n,-1)
from scipy.optimize import least_squares
from itertools import combinations_with_replacement,product
print('times',s['eval_times']); print('z ranges',np.round(np.array([z.min(0),z.max(0)]),3)); print('test z ranges',np.round(np.array([a.features(s,s['test']['x'])[0].min(0),a.features(s,s['test']['x'])[0].max(0)]),3)); print('split z ranges',np.round(np.array([z[:52].min(0),z[:52].max(0),z[52:].min(0),z[52:].max(0)]),3))
t=s['eval_times']; W=np.c_[y,T]; fs={}; fs['base']=np.c_[p,c,cross]; fs['zbase']=np.c_[z,fs['base']];
for name,F in list(fs.items()):
 A=np.c_[np.ones(n),F]; E=W-A@np.linalg.lstsq(A,W,rcond=None)[0]; _,ss,V=np.linalg.svd(E,full_matrices=False); print(name,'singular',ss,'ratio',np.round(V[0,1:]/V[0,0],5)); print('orthogonal spatial MAE',np.mean(abs(E[:,1:]-E[:,0,None]*(V[0,1:]/V[0,0]))))
def poly(X,k): return np.stack([np.prod(X[:,j],axis=1) for m in range(1,k+1) for j in combinations_with_replacement(range(X.shape[1]),m)],1)
def xprod(A,B):return (A[:,:,None]*B[:,None,:]).reshape(n,-1)
fs.update({'z2':poly(z,2),'z3':poly(z,3),'z4':poly(z,4),'z5':poly(z,5),'z2base':np.c_[poly(z,2),fs['base']],'z3base':np.c_[poly(z,3),fs['base']],'z4base':np.c_[poly(z,4),fs['base']],'trig':np.c_[fs['zbase'],np.sin(2.4*z),xprod(p,np.sin(2.4*z))],'pair':np.c_[fs['zbase'],np.sin(z),np.sin(2*z),np.sin(2.4*z),np.cos(z)],'pq2':np.c_[z,poly(np.c_[p,c],2)]})
F=[np.arange(52,72),*np.array_split(np.random.default_rng(325).permutation(n),4)]
for name,B in fs.items():
 B=B/np.maximum(B.std(0),.02); B=np.c_[np.ones(n),B]; K=B@B.T
 for lam in [.00001,.003,.1,1.]:
  scores=[]
  for te in F:
   tr=np.setdiff1d(np.arange(n),te); k=K[np.ix_(tr,tr)]; w=np.linalg.solve(k+lam*np.eye(len(tr)),W[tr]); e=abs(K[np.ix_(te,tr)]@w-W[te]); scores.append([e[:,0].mean(),e[:,1:].mean()]);
  print(name,lam,np.round(scores[0],5),np.round(np.max(scores,axis=0),5))
for withz in [False,True]:
 B=np.c_[np.ones(n),p,c,dot] if not withz else np.c_[np.ones(n),z,p,c,dot];
 for fun in ['sin','tanh','erf','rat','exp','softplus']:
  from scipy.special import erf
  f={'sin':np.sin,'tanh':np.tanh,'erf':erf,'rat':lambda x:x/(1+x*x),'exp':lambda x:np.exp(-x*x),'softplus':lambda x:np.logaddexp(0,x)}[fun]; S=[]; ps=[]
  for te in F:
   tr=np.setdiff1d(np.arange(n),te); Q=np.linalg.qr(B[tr])[0]; yy=y[tr]-Q@(Q.T@y[tr]);
   def pred(g):
    v=f(z[tr]@g[:4]+g[4]); v-=Q@(Q.T@v); return v*(v@yy)/(v@v+1e-14)
   opts=[least_squares(lambda g:pred(g)-yy,g,max_nfev=100) for g in np.random.default_rng(467).normal(0,.9,(12,5))]; op=min(opts,key=lambda o:o.cost); G=np.c_[B,f(z@op.x[:4]+op.x[4])]; out=G[te]@np.linalg.lstsq(G[tr],y[tr],rcond=None)[0]; S.append(np.mean(abs(out-y[te]))); ps.append(np.round(op.x,3));
  print('index',withz,fun,np.round(S,5),'params',ps[0])
