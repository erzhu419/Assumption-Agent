from dag_probe import *
from scipy.spatial.distance import cdist
z=mods[0]._align(s['panel_a'],s['train']['x']); za=np.abs(z); tr=np.arange(52); va=np.arange(52,n)
q=2.4; C=z*np.roll(z,-1,axis=1); D=np.cos(q*z); E=np.c_[C,D]; Z=np.c_[z, E];
H=np.column_stack([np.prod(z[:,v],axis=1) for v in combinations_with_replacement(range(4),2)]); J=np.column_stack([np.prod(z[:,v],axis=1) for v in combinations_with_replacement(range(4),3)])
for name,F in [('even',E),('even+fullquad',np.c_[E,H]),('signed+fullquad',np.c_[Z,H]),('signed+cubic',np.c_[Z,H,J]),('signed nonlinear',np.c_[Z,H,np.sin(2.4*z),np.tanh(z),z/(1+.25*z*z)]),('signed+cos/cyclic interactions',np.c_[Z,H,np.sin(2.4*z),C*D,C*np.roll(D,-1,axis=1)])]:
 F=np.c_[np.ones(n),F]; yh=F@lst(F[tr],T[tr]); e=np.abs(yh-T); print(name, np.mean(e[tr,0]),np.mean(e[tr,1:]),np.mean(e[va,0]),np.mean(e[va,1:]));
 d=lst(F[tr],T[tr]); print('coeff scalar and terminal', np.round(d[:,[0,-1]].T,3))
print('nearest antipodes and parallel analogs'); M=cdist(z,-z); L=cdist(z,z); np.fill_diagonal(L,np.inf); np.fill_diagonal(M,np.inf)
for name,A in [('opposite',M),('parallel',L)]:
 ids=np.argsort(A.min(1))[:10]; js=A.argmin(1)[ids]; print(name, np.c_[ids,js,A[ids,js],T[ids,0],T[js,0],T[ids,-1],T[js,-1]].round(3))
