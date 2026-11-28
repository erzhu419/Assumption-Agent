from spatial06 import *
from itertools import combinations_with_replacement as cwr
z,w,((a,b),_)=geom(s);y=s['train']['y']; S=s['train']['traj']; t=s['eval_times'];n=len(y);i=np.arange(56);j=np.arange(56,n)
B=s['panel_b'];ab=ls(aff(B['u']),B['v']);wc=w-ab[0];dot=np.sum(a*b,1);cross=a[:,0]*b[:,1];H=aff(np.c_[b,dot,cross]);
def polys(x,d):return np.stack([np.prod(x[:,k],1) for k in cwr(range(4),d)],1)
K=polys(z,2); Z=polys(z,3); C=polys(wc,3); P=np.c_[H,z,K];
F=np.c_[P,Z]; Y=np.c_[y,S]; A=ls(F[i],Y[i]); poly=A[-20:];print('cubic singular values',np.linalg.svd(poly,compute_uv=False)); print('cubic z/w coeffs',np.round(np.c_[poly[:,0],ls(C,Z)@poly[:,0]],4));
print('residual MAE by time',np.mean(abs((F[j]@A-Y[j])),0));
xs={'z':z,'w':w,'wc':wc}; xs.update({f'z{u}':z[:,u,None] for u in range(4)}); xs.update({f'w{u}':w[:,u,None] for u in range(4)});
L=np.c_[z,w,wc]; labels=[f'{x}{u}' for x in ['z','w','wc'] for u in range(4)]; names=[];V=[]
for a1,a2,a3 in cwr(range(L.shape[1]),3):
 names.append('*'.join(labels[k] for k in [a1,a2,a3]));V.append(L[:,a1]*L[:,a2]*L[:,a3])
D=np.stack(V,1);R=D-P@ls(P[i],D[i]);U=Y[:,:1]-P[:n]@ls(P[i],Y[i,:1]);
for it in range(10):
 Q=R[i]; norms=np.linalg.norm(Q,axis=0); score=np.divide(Q.T@U[i,0],norms,out=np.zeros(len(norms)),where=norms>1e-7);k=np.argmax(abs(score));P=np.c_[P,D[:,k]];R=D-P@ls(P[i],D[i]);U=Y[:,:1]-P[:n]@ls(P[i],Y[i,:1]);print(it,names[k],np.std(U[i]),np.mean(abs(U[j])))
