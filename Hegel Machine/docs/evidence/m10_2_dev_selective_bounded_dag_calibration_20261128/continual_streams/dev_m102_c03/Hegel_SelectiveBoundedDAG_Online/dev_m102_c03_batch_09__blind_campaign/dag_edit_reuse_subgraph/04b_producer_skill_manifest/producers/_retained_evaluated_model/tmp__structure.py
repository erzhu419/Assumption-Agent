import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; a=s['panel_a']; u,v=a['u'],a['v']; h=np.c_[np.ones(len(u)),u]; w=np.ones(len(u))
for _ in range(15):
 b=np.linalg.lstsq(h*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(v-h@b,axis=1); w=1/np.sqrt(1+(r/max(.0002,np.median(r)))**2)
z=(tr['x']-b[0])/np.diag(b[1:]); y=np.c_[tr['y'],tr['traj']]; n=len(z); print('n',n)
p=s['panel_c']; u=p['u']; t=s['eval_times']; print('times',t)
def one(x):return np.c_[np.ones(len(x)),x]
def fit(h,y):return np.linalg.lstsq(h,y,rcond=None)[0]
def fun(a,k):
 return {'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'exp':lambda v:np.expm1(-v*v),'softsign':lambda v:v/(1+abs(v))}[k](a)
branches={}
for key in ('v1','v2'):
 v=p[key]; rows=[]
 for kind in ('tanh','sin','atan','exp','softsign'):
  for mode in (0,1,2,3):
   def h(a,g):
    if mode==0: return one(fun(a*g[0],kind))
    if mode==1: return one(np.c_[a,fun(a*g[0],kind)])
    if mode==2: return one(fun(a*g,kind))
    return one(np.c_[a,fun(a*g[:4]+g[4:],kind)])
   g0={0:[1.],1:[1.],2:[1.]*4,3:[1.]*4+[0.]*4}[mode]
   def resid(g):
    H=h(u,g); return (H@fit(H,v)-v).ravel()
   lo=[.05]*min(len(g0),4)+[-1.]*max(0,len(g0)-4);hi=[4.]*min(len(g0),4)+[1.]*max(0,len(g0)-4)
   o=least_squares(resid,g0,bounds=(lo,hi),max_nfev=100); e=np.std(o.fun); rows.append((e,kind,mode,o.x,h(z,o.x)@fit(h(u,o.x),v)))
 rows.sort(key=lambda row:row[0]); print(key,[(round(e,7),k,m,g.round(4).tolist()) for e,k,m,g,_ in rows[:7]]); branches[key]=rows[0][-1]
c,d=branches.values(); q=-np.expm1(-t[:,None]*np.array([.392,1.857])); yy=np.c_[y[:,0],fit(q,y[:,1:].T).T];
from itertools import combinations
pairs=lambda a:np.column_stack([a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(a.shape[1]),2)])
edge=lambda a:a*np.roll(a,-1,axis=1)
g=np.expm1(-.2962*z*z); h=np.tanh(1.492*z)
F={'base':np.c_[z,c,d,np.sum(c*d,axis=1)],'latent':np.c_[z,h,g],'pairs':np.c_[z,c,d,pairs(c),pairs(d),np.einsum('ni,nj->nij',c,d).reshape(n,-1)],'primitives':np.c_[z,h,g,pairs(z),pairs(h),pairs(g)],'hybrid':np.c_[z,c,d,pairs(z),pairs(c),pairs(d),c*d,edge(z),edge(h),edge(g)]}
for name,A in F.items():
 A=one(A); print(name,A.shape[1], 'full',np.std(A@fit(A,yy)-yy,0).round(4)); out=[]
 for pp in [np.arange(n)]+[np.random.default_rng(i+100).permutation(n) for i in range(10)]:
  ia,it=pp[:-20],pp[-20:]; scale=np.maximum(1e-10,np.linalg.norm(A[ia],axis=0)); B=A/scale; K=B@B.T; out.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@q.T))] if False else [np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@q.T))] ) if False else None
  e=K[np.ix_(it,ia)]@np.linalg.solve(K[np.ix_(ia,ia)]+.0003*np.eye(len(ia)),yy[ia])-yy[it]; out.append([np.mean(abs(e[:,0])),np.mean(abs(e[:,1:]@q.T))]);
 print(np.round(out[0],4),np.mean(out,0).round(4),np.max(out,0).round(4))
