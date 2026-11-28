exec(open('/tmp/probe05.py').read().split('p=temp(z[:56])')[0])
from itertools import combinations
from scipy.linalg import qr

def poly(q,k): return np.column_stack([np.ones(len(q))]+[np.prod(q[:,ii],axis=1) for j in range(1,k+1) for ii in combinations_with_replacement(range(q.shape[1]),j)])
def loo(F,Y):
 U,s,V=np.linalg.svd(F,full_matrices=False); U=U[:,s>s[0]*1e-10]; E=Y-U@(U.T@Y); return np.mean(abs(E/np.maximum(1e-8,1-np.sum(U*U,axis=1))[:,None])), np.sqrt(np.mean(E**2))
u=s['panel_c']['u']; Y=np.c_[s['panel_c']['v1'],s['panel_c']['v2']]; print('panel baseline',loo(poly(u,1),Y[:,:4]),loo(poly(u,2),Y[:,:4]),loo(poly(u,2),Y[:,4:])); print('quad coefficient',np.linalg.lstsq(poly(u,2),Y,rcond=None)[0]);
qs={'raw':u}
for name,op in [('A',a),('B',b),('BA',lambda u:b(a(u)))]: qs[name]=op(u)
for name,q in list(qs.items()):
 qs[name+' pair']=np.column_stack([q[:,i]+sg*q[:,j] for i,j in combinations(range(4),2) for sg in (-1,1)])
for g in range(8):
 arr=[]
 for name,q in qs.items():
  for i in range(q.shape[1]):
   v=q[:,i];
   for nm,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('sqrat',lambda h:1/(1+h*h)),('cube',lambda h:h**3)]:
    for k in (.5,1,1.5,2,2.5,3,4):
     for deg in (1,2): arr.append((loo(np.c_[poly(u,deg),fn(k*v)],Y[:,g:g+1]),(name,i,nm,k,deg)))
 print('single panel feature',g,sorted(arr)[:3])
p=temp(z[:56]); v=amp(z,p); F=poly(a(x),2); lst=[(np.arange(56),np.arange(56,72))]+[(r[:56],r[56:]) for r in (np.random.default_rng(k).permutation(72) for k in range(4))]
for j in range(3):
 D=np.c_[y,v]; print('coefficient',j,[(n,np.linalg.lstsq(np.c_[F,D[:,n]],D[:,j],rcond=None)[0][-1]) for n in range(3) if n!=j]);
for mode in ('base','a1','a2','both','prod'):
 ans=[]
 for i,j in lst:
  v=amp(z,temp(z[i])); G={'base':F,'a1':np.c_[F,v[:,0]],'a2':np.c_[F,v[:,1]],'both':np.c_[F,v],'prod':np.c_[F,v,v[:,0]*v[:,1]]}[mode]; co=np.linalg.lstsq(G[i],y[i],rcond=None)[0]; ans.append(np.mean(abs(G[j]@co-y[j])))
 print('oracle',mode,ans)
