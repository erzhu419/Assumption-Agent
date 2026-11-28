import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from branch6 import *
F=lambda u,a,b:u+a*(np.logaddexp(b*u,-b*u)-np.log(2)); pp=least_squares(residual,[.9,1.1],args=(F,)).x
C=lambda u:A(F(u,*pp))@L(A(F(cu,*pp)),V)
r=C(w); R=C(z)
_,_,vt=np.linalg.svd(Y[:52,1:],full_matrices=False)
D0=A(np.c_[w,q]); Z=z[:,[1,2,3,0]]*q
print('dead',fit.x,'C2',pp)
base=[('wq',D0),('rotated',np.c_[D0,Z]),('simple',np.c_[D0,Z,w[:,0]*q[:,0],q[:,1]*R[:,0],q[:,0]*r[:,2],(q[:,2]-r[:,2])*R[:,2]]),('sparse8',np.c_[D0,Z,w[:,0]*expit(1.25*q[:,0]),q[:,1]*R[:,0],q[:,0]*expit(1.25*r[:,2]),(q[:,2]-r[:,2])*R[:,2]])]
for name,D in base:
 print(name,score(D)); Q=L(D[:52],Y[:52]); print(np.c_[Q[:,0],Q[:,1:]@vt[:3].T]); print('resid svd',np.linalg.svd(D[:52]@Q[:,1:]-Y[:52,1:],compute_uv=False))
D=base[-1][1]; times=s['eval_times']; print('times',times)
def inspect(D,target):
 P=L(D[:52],target[:52]); E=target[:52]-D[:52]@P; Q0=np.linalg.qr(D[:52])[0]; vals=[]
 d0={'z':z,'w':w,'q':q,'r':r,'R':R,'u':dead(z,*fit.x),'x':x}
 for n,b0 in d0.items():
  for i in range(4):
   a=b0[:,i]; key=n+str(i)
   for nm,v in [('linear',a),('abs',abs(a)),('square',a*a),('cube',a**3),('relu',np.maximum(a,0)),('dead',dead(a,.22,1)),('tanh',np.tanh(1.25*a)),('cosh',np.logaddexp(a,-a))]:vals.append((nm+key,v))
   for n1,b1 in d0.items():
    for j in range(4):
     b=b1[:,j]; k=key+n1+str(j)
     for nm,v in [('mul',a*b),('sig',a*expit(1.25*b)),('rat',a/(1+1.25*b*b)),('rat2',a/(1+.22*b*b)),('abs',a*abs(b)),('min',np.minimum(a,b)),('max',np.maximum(a,b)),('mul2',a*b*b),('mulrat',a*b/(1+1.25*b*b)),('sigplus',a*expit(1.25*(a+b))),('ratplus',a/(1+1.25*(a+b)**2)),('ratdiff',a/(1+1.25*(a-b)**2))]: vals.append((nm+k,v))
 M=np.column_stack([v for _,v in vals]); Mr=M[:52]-Q0@(Q0.T@M[:52]); c=np.linalg.norm(Mr.T@E,axis=1)/np.maximum(np.linalg.norm(Mr,axis=0),1e-6); print('resid features',[(vals[i][0],round(c[i],5),score(np.c_[D,M[:,i]])) for i in np.argsort(-c)[:14]])
for jj in [[0],list(range(1,9))]: inspect(D,Y[:,jj])
# Test if the readout uses an exact cyclic sum, or a gated version of it.
for n,M in [('z',z),('w',w),('r',r),('R',R),('x',x)]:
 for k in range(4):
  G=q*np.roll(M,-k,axis=1)
  for H in [G,G.sum(1),q*expit(1.25*np.roll(M,-k,axis=1)),q*np.tanh(1.25*np.roll(M,-k,axis=1))]:
   S=score(np.c_[D0,H]);
   if min(S[2:])<.045: print('cyclic',n,k,H.shape,S)
