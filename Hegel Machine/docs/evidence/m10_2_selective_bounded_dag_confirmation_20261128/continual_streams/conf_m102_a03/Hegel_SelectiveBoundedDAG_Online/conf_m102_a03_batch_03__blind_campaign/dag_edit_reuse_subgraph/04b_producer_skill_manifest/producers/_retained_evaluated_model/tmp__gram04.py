exec(open('/tmp/probe04.py').read().split('for pn,kn')[0])
from itertools import combinations_with_replacement as comb

def test(z,y):
 for kind in ('sin','tanh','rat','sqrt','cubic'):
  yp,yt,w=ridge(z[:52],y[:52],z[52:],kind,True,6)
  print(kind,round(abs(yp-y[:52]).mean(),5),round(abs(yt-y[52:]).mean(),5),np.round(w,4))
T=tr['traj']; _,_,V=np.linalg.svd(T[:52],full_matrices=False); scores=T@V[:2].T
for y in [tr['y'],*scores.T]:
 print('target');test(x,y)
p=s['panel_b'];u=p['u'];v=p['v'];print('panel shape',u.shape,'first pairs',np.round(np.c_[u,v][:6],3));
for p0 in ('panel_b','panel_c'):
 p=s[p0];u=p['u'];
 for key in sorted(p.keys()-{'u'}):
  v=p[key]; print(p0,key)
  for order in (1,2,3):
   idx=[ix for d in range(1,order+1) for ix in comb(range(4),d)]; A=np.column_stack([np.ones(len(u))]+[np.prod(u[:,ix],axis=1) for ix in idx]);C=fit(A,v); h=np.sum(A*np.linalg.pinv(A).T,axis=1);e=(v-A@C); print(order,'train',abs(e).mean(),'loo',abs(e/np.maximum(1-h[:,None],1e-8)).mean(),'sv',np.round(np.linalg.svd(e,compute_uv=False),4));
   if order in (1,3): print('strongest',[(idx[i],np.round(C[i+1],3)) for i in np.argsort(np.linalg.norm(C[1:],axis=1))[-9:][::-1]])
# Coordinate-frequency search for each target; first-stage coordinate terms and products.
for name,y in [('y',tr['y']),('s0',scores[:,0]),('s1',scores[:,1])]:
 A=a(x); idx=list(comb(range(4),2)); B=np.c_[A,*[x[:,j]*x[:,k] for j,k in idx]]
 R=y[:52]-B[:52]@fit(B[:52],y[:52]); features=[]
 for j in range(4):
  for freq in np.geomspace(.3,8,75):
   for typ in ('sin','cos','tanh','rat'):
    f=(np.cos if typ=='cos' else fs[typ])(freq*x[:,j]); F=np.c_[B,f]; c=fit(F[:52],y[:52]); features.append((abs(F[:52]@c-y[:52]).mean(),abs(F[52:]@c-y[52:]).mean(),j,freq,typ))
 print(name,sorted(features)[:3])
