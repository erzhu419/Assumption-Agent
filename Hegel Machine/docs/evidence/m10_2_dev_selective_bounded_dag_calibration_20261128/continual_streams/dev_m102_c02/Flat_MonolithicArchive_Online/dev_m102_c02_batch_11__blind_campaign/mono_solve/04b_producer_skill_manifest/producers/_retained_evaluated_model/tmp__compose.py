import numpy as np,pickle, itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; A=np.load('/tmp/cached.npz'); z=A['z']; b=A['panel_bvlinear']; y=s['train']['y']; n=52
ls=lambda h,v:np.linalg.lstsq(h,v,rcond=1e-9)[0]
a=lambda x:np.c_[np.ones(len(x)),x]
u=s['panel_c']['u']; C=[]
for key,fn,ps,bd in [('v1',lambda x,p:x/(1+p[0]*abs(x)**p[1])**p[2],[.62,1.07,1.57],([0,.1,.1],[5,5,5])),('v2',lambda x,p:np.logaddexp(0,p[0]*x+p[1]),[2.2,0],([.1,-3],[8,3]))]:
 v=s['panel_c'][key]
 def er(p):
  h=a(fn(u,p)); return (h@ls(h,v)-v).ravel()
 p=least_squares(er,ps,bounds=bd,gtol=1e-13).x; h=a(fn(u,p)); B=ls(h,v); print(key,p,np.std(er(p))); print(B)
 C.append(lambda x,p=p,B=B,fn=fn:a(fn(x,p))@B)
P={'z':z,'b':b}; F={}; i,j=np.triu_indices(4)
for k,x in [('z',z),('b',b)]:
 for r,f in enumerate(C,1):P[f'{k}{r}']=f(x)
for k,x in P.items():
 F[k]=x; F[k+'q']=x[:,i]*x[:,j]; F[k+'p']=(x[:,0]*x[:,2]+x[:,1]*x[:,3])[:,None]; F[k+'d']=(x[:,0]*x[:,2]-x[:,1]*x[:,3])[:,None]; F[k+'c']=x**3
for (k,x),(l,w) in itertools.combinations(P.items(),2):
 for l1,w1,k1,x1 in [(l,w,k,x),(k,x,l,w)]:
  F[k1+'r'+l1]=x1/(1+.22*w1*w1); F[k1+'m'+l1]=(x1*w1)
 F[k+'dot'+l]=(x*w).sum(1)[:,None]
def score(h):
 H=h[:n]; U,S,V=np.linalg.svd(H,full_matrices=False); r=S>S[0]*1e-9
 if r.sum()<len(r):return (100,100,100)
 Q=U[:,r]; er=y[:n]-Q@(Q.T@y[:n]); return np.mean(abs(er/np.maximum(1-(Q*Q).sum(1),1e-7))),np.mean(abs(h[n:]@ls(H,y[:n])-y[n:])),np.std(er)
D=[]; beam=[()]; seen=set(beam)
for depth in range(4):
 R=[]
 for seq in beam:
  for k in F:
   q=tuple(sorted(set(seq+(k,))));
   if q in seen:continue
   seen.add(q); h=a(np.column_stack([F[k] for k in q])); R.append((score(h),q))
 R.sort(); D+=R[:25]; beam=[q for _,q in R[:22]]
print('best'); print(*sorted(D)[:30],sep='\n'); np.savez('/tmp/composed.npz',**P)
