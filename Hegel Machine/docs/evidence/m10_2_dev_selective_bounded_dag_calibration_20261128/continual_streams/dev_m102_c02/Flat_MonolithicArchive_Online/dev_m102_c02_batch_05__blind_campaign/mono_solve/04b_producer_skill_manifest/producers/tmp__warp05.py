import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from graph05 import *
from scipy.linalg import subspace_angles
U,V=s['panel_b']['u'],s['panel_b']['v']
def run(name,fun,start=[],lo=[],hi=[]):
 def fit(p):
  B=fun(U,p); H=A(B.reshape(len(U),-1)); return H,F(H,V)
 def loss(p):
  h,c=fit(p);return (h@c-V).ravel()
 p=least_squares(loss,start,bounds=(lo,hi),gtol=1e-12,max_nfev=500).x if len(start) else []
 h,c=fit(p); wz=A(fun(z,p).reshape(len(z),-1))@c
 a=A(wz); pair=wz[:,0]*wz[:,2]+wz[:,1]*wz[:,3]; h=np.c_[a,pair]; r=np.std(loss(p)); print(name,p,round(r,7),score(h), 'traj',np.mean(abs(a[52:72]@F(a[:52],yt[:52])-yt[52:])))
 return wz,r
G={}
for ex in [1,1.25,2,2.5,3,4,5,6,8]:
 name='fixed cubic '+str(ex);G[name]=run(name,lambda u,p: u**3/(1+.22*abs(u)**ex))
for k in [2,3,4,5]:
 name='odd poly '+str(k);G[name]=run(name,lambda u,p: sum(co*u**(2*i+1) for i,co in enumerate([*p,1])),[-1]+[0]*(k-2),[-1e4]*(k-1),[1e4]*(k-1))
for kk in [2,3,4,5,6]:
 for den in [1,2,3,4,6]:
  name='shared rational '+str((kk,den));G[name]=run(name,lambda u,p: (u**3+p[0]*u+p[1]*np.sign(u)*abs(u)**kk)/(1+p[2]*abs(u)**den+p[3]*abs(u)**(2*den)),[0,.01,.22,.01],[-10,-10,0,0],[10,10,30,30])
for power in [2,3,4,5,6,8]:
 name='sum damped '+str(power);G[name]=run(name,lambda u,p:u**3/(1+p[0]*abs(u)**power)+p[1]*u+p[2]*u**3,[.22,0,0],[0,-10,-10],[50,10,10])
print('same-coordinate affine+nonlinear vector weights')
for name,fun in [('cube',lambda x:x**3),('ratcube',lambda x:x**3/(1+.22*x*x)),('ratquart',lambda x:x**3/(1+.22*x**4)),('power25',lambda x:np.sign(x)*abs(x)**2.5)]:
 H=A(np.c_[U,fun(U)]);C=F(H,V);print(name,np.std(H@C-V),C[1:5],C[5:]);
for name,(d,r) in sorted(G.items(),key=lambda kv:kv[1][1])[:6]:
 a=A(d); pair=d[:,0]*d[:,2]+d[:,1]*d[:,3]; h=np.c_[a,pair]; q=F(h[:72],y); err=y-h[:72]@q
 print('candidate',name,'readout',q,'errs',np.std(err))
 feats=[('linCube',d**3),('cubeZ',z**3),('saturation',d/(1+.22*d*d)),('linZ',z),('C1',w['c1powered']),('C2',w['c2even']),('P*lin',pair[:,None]*d),('T',np.c_[d[:,0]*d[:,1]*d[:,2],d[:,0]*d[:,1]*d[:,3],d[:,0]*d[:,2]*d[:,3],d[:,1]*d[:,2]*d[:,3]])]
 print([(tag,score(np.c_[h,f])) for tag,f in feats])
