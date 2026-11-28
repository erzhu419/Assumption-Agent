import numpy as np,pickle,importlib.util
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; N=52
one=lambda x:np.column_stack((np.ones(len(x)),x)); ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
A=s['panel_a']; u,v=A['u'],A['v']; U=one(u); rng=np.random.default_rng(45)
sol=[]
for _ in range(200):
 i=rng.choice(len(u),8,False); q=ls(U[i],v[i]); sol.append((np.median(np.linalg.norm(U@q-v,axis=1)),q))
q=min(sol,key=lambda p:p[0])[1]; r=np.linalg.norm(U@q-v,axis=1); good=r<max(.005,5*np.median(r)); q=ls(U[good],v[good]); print('full cal',q,'error',np.sort(np.linalg.norm(U@q-v,axis=1))); z=np.linalg.solve(q[1:].T,(tr['x']-q[0]).T).T
B=s['panel_b']; qb=ls(one(B['u']),B['v']); bfn=lambda z:one(z)@qb
C=s['panel_c']; fs=[]
for k,f,guess in [('v1',lambda w:w/(1+abs(w)),.84),('v2',lambda w:w/(1+np.exp(-w)),1.45)]:
 def fit(k,f,g):
  u,v=C['u'],C[k]; D=lambda z,h:one(np.c_[z,f(h[0]*z)]); res=lambda h:(D(u,h)@ls(D(u,h),v)-v).ravel(); o=least_squares(res,[g]); print(k,o.x,np.sqrt(np.mean(o.fun**2))); coef=ls(D(u,o.x),v); return lambda z:D(z,o.x)@coef
 fs.append(fit(k,f,guess))

def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def score(F):
 F=one(F);return ' '.join(str(tuple(round(w,5) for w in (np.mean(abs(F[N:]@ls(F[:N],tr[k][:N])-tr[k][N:])),np.mean(abs(F[:N]@ls(F[:N],tr[k][:N])-tr[k][:N]))))) for k in ['y','traj'])
for root,zz in [('inv',z),('direct',one(tr['x'])@q),('raw',tr['x']),('invT',(tr['x']-q[0])@np.linalg.inv(q[1:].T))]:
 b=bfn(zz); bi=(zz-qb[0])@np.linalg.inv(qb[1:]);
 for src,h in [('z',zz),('b',b),('bi',bi)]:
  c,d=[f(h) for f in fs]; c1,d1=bfn(c),bfn(d); print(root,src)
  for tag,F in [('cd',np.c_[c,d]),('cd+dot',np.c_[c,d,(c*d).sum(1)]),('cd+prod',np.c_[c,d,c*d]),('cd+cross',np.c_[c,d,cross(c,d)]),('cd+dotB',np.c_[c,d,(c1*d1).sum(1)]),('cd+prodB',np.c_[c,d,c1*d1]),('all+cross',np.c_[h,c,d,cross(c,d)])]:print(tag,score(F))
  if (root,src)==('inv','z'):np.savez('/tmp/fullops.npz',z=zz,b=b,c=c,d=d,cb=c1,db=d1)
