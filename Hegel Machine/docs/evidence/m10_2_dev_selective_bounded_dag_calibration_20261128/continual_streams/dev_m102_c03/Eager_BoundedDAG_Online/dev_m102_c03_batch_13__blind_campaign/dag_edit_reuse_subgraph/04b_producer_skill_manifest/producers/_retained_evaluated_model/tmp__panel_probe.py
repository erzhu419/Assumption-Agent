import numpy as np, pickle, runpy
from scipy.optimize import minimize_scalar,least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=len(s['train']['x']); x=s['train']['x']; yy=np.c_[s['train']['y'],s['train']['traj']];
def bias(x): return np.c_[np.ones(len(x)),x]
def ls(f,y): return np.linalg.lstsq(f,y,rcond=1e-10)[0]
a=s['panel_a']; u,v=a['u'],a['v']; keep=np.ones(len(u),bool)
for i in range(5):
 b=ls(bias(u)[keep],v[keep]); e=np.linalg.norm(bias(u)@b-v,axis=1); keep=e<max(0.001,4*np.median(e)); print('a',i,keep.sum(),np.quantile(e,[0,.5,.85,1]))
print('a params',b); print('b params',ls(bias(s['panel_b']['u']),s['panel_b']['v'])); z=(x-b[0])@np.linalg.inv(b[1:]); print('z range',z.min(0),z.max(0)); print('times',s['eval_times']);
lib={'linear': lambda u,p:u,'sin':lambda u,p:np.sin(p*u),'cos':lambda u,p:np.cos(p*u),'tanh':lambda u,p:np.tanh(p*u),'atan':lambda u,p:np.arctan(p*u),'soft':lambda u,p:u/(1+p*abs(u)),'rational':lambda u,p:u/(1+p*u*u),'cubic':lambda u,p:u+p*u**3,'sinh':lambda u,p:np.sinh(p*u),'gauss':lambda u,p:np.exp(-p*u*u),'logistic':lambda u,p:1/(1+np.exp(-p*u))}
from itertools import combinations
p=s['panel_c']; u=p['u']; ops=[]
for name in ['v1','v2']:
 v=p[name]; result=[]
 for key,g in lib.items():
  for add in [False,True]:
   def make(u,k): return bias(np.c_[u,g(u,k)] if add else g(u,k))
   def fun(k):
    f=make(u,k); return np.mean((f@ls(f,v)-v)**2)
   grid=np.linspace(.02,4,50); j=np.argmin([fun(k) for k in grid]); r=minimize_scalar(fun,bounds=(grid[max(0,j-1)],grid[min(len(grid)-1,j+1)]),method='bounded'); result.append((r.fun,key,add,r.x,make(z,r.x)@ls(make(u,r.x),v)))
 result.sort(key=lambda o:o[0]); print(name,[(a,b,c,d) for a,b,c,d,_ in result[:9]]); ops.append(result)
t=s['eval_times']; tr=yy[:,1:]
for nm,make in [('relax',lambda p:-np.expm1(-t[:,None]*p)),('decay',lambda p:np.exp(-t[:,None]*p)),('one each',lambda p:np.c_[-np.expm1(-t*p[0]),np.exp(-t*p[1])])]:
 def obj(p):
  f=make(p); return (f@ls(f,tr.T)-tr.T).ravel()
 r=least_squares(obj,[.4,1.86],bounds=(.0001,8)); print(nm,r.x,np.std(r.fun))
cv=[(np.arange(52),np.arange(52,72))]; cv.extend((np.setdiff1d(np.arange(n),j),j) for seed in [182,991] for j in np.array_split(np.random.default_rng(seed).permutation(n),4));
def scores(f,reg=0):
 es=[]
 for i,j in cv:
  fi=f[i]; fj=f[j]; fm=fi.mean(0); fs=np.maximum(fi.std(0),1e-9); fi=(fi-fm)/fs; fj=(fj-fm)/fs; y=yy[i]; ym=y.mean(0); b=ls(fi,y-ym) if reg==0 else np.linalg.solve(fi.T@fi+reg*np.eye(fi.shape[1]),fi.T@(y-ym)); e=abs(fj@b+ym-yy[j]); es.append([e[:,0].mean(),e[:,1:].mean()]);
 return np.round(np.r_[es[0],np.mean(es[1:],0),np.max(es[1:],0)],5)
def quad(w): return np.c_[w,*[w[:,i]*w[:,j] for i in range(w.shape[1]) for j in range(i,w.shape[1])]]
for i,j in [(0,0),(0,1),(1,0),(1,1),(2,2)]:
 a=ops[0][i][-1]; b=ops[1][j][-1]; w=np.c_[a,b];
 for q in [0,1,2]:
  f=np.c_[w,(a*b).sum(1),a[:,0]*b[:,1]] if q==0 else quad(w) if q==1 else np.c_[quad(a),quad(b),a*b]; print('read',i,j,q,[scores(f,r).tolist() for r in [0,.03,.3,1]])
for key,g in lib.items():
 for k in [.22,.7,1.25,2.15]:
  f=quad(g(z,k)); print('direct',key,k,scores(f,.03))
