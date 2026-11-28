from probe import *
from itertools import product
from scipy.linalg import qr
W=np.c_[u,uz]; Q=np.c_[x,z]; V1=C['v1']; V2=C['v2'];

def audit(F,v):
 F=np.c_[np.ones(len(F)),F]; inv=np.linalg.pinv(F); E=v-F@inv@v; h=np.sum(F*inv.T,axis=1); return np.sqrt(np.mean(E**2)), np.mean(abs(E/np.maximum(1-h,1e-9)[:,None])), inv@v

a={ 'xz':(W,Q),'x':(u,x),'z':(uz,z) }
extras={
 'x²':lambda a,b:a*a,'z²':lambda a,b:b*b,'xzdiag':lambda a,b:a*b,
 'xcube':lambda a,b:a**3,'zcube':lambda a,b:b**3,
 'xcyclic':lambda a,b:a*np.roll(a,-1,axis=1),'zcyclic':lambda a,b:b*np.roll(b,-1,axis=1),
 'xtanh':lambda a,b:np.tanh(a),'ztanh':lambda a,b:np.tanh(b),
 'zsat':lambda a,b:b/(1+b*b),'xsat':lambda a,b:a/(1+a*a),
 'xznext':lambda a,b:a*np.roll(b,-1,axis=1),'zxnext':lambda a,b:b*np.roll(a,-1,axis=1),
 'zsin':lambda a,b:np.sin(b),'xsin':lambda a,b:np.sin(a), 'zcos':lambda a,b:np.cos(b),'xcos':lambda a,b:np.cos(a)}
for k,v in [('v1',V1),('v2',V2)]:
 results=[]
 for name,(wp,wq) in a.items():
  for m in range(3):
   for ks in itertools.combinations(extras,m):
    F=np.column_stack([wp]+[extras[i](u,uz) for i in ks]); G=np.column_stack([wq]+[extras[i](x,z) for i in ks]); tr,te,co=audit(F,v); results.append((te,tr,name,ks,co,G))
 results.sort(key=lambda r:r[:2]); print(k,[(round(e,6),round(r,6),n,f) for e,r,n,f,c,g in results[:17]])
 for e,r,n,f,c,g in results[:3]: print(n,f,c.round(3))
 F=results[0][-1]; p=np.c_[np.ones(len(F)),F]@results[0][-2]
 if k=='v1': h=p
 else: j=p
print('panel v1 xz residual singulars', np.linalg.svd(V1-np.c_[np.ones(len(W)),W]@audit(W,V1)[2],compute_uv=False))
for label,a,b in [('hj',h,j),('hz',h,z),('xh',x,h),('xj',x,j)]:
 f,keys=feats(a,b); R=np.c_[y,T]; R=R[:52] if False else R
 for label2,fs in [('linear',f[:,:8]),('diag',np.c_[a,b,a*a,b*b,a*b]),('full',f),('cub',np.c_[f,a**3,b**3]),('sat', np.c_[a,b,a*a,b*b,a*b,np.tanh(a),np.tanh(b)])]:
  g=np.c_[np.ones(len(f)),fs]; sc=np.maximum(1e-8,g[:52].std(0)); sc[0]=1; g=g/sc
  for lam in ([0,.003,.1] if label2 in ('full','cub') else [0]):
   pred=g[52:]@np.linalg.solve(g[:52].T@g[:52]+lam*np.eye(g.shape[1]),g[:52].T@R[:52]) if lam else g[52:]@np.linalg.lstsq(g[:52],R[:52],rcond=None)[0]; es=np.mean(abs(pred-R[52:]),axis=0); print(label,label2,lam,np.round([es[0],es[1:].mean(),es[-1]],5))
