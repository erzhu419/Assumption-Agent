from studyfit import *
f=np.load('/tmp/refined.npz');z,L,C,D,T,Y,H=(f[k] for k in ('z','L','C','D','T','Y','H')); N=52
P=(C[:,:,None]*D[:,None,:]).reshape(len(C),16);dot=(C*D).sum(1);dotL=(C*L).sum(1)
F=one(np.c_[C,D]); np.set_printoptions(precision=4,suppress=True)
for n,M in [('base',F),('dot',np.c_[F,dot]),('dot_cross',np.c_[F,dot,C[:,0]*D[:,1]]),('full',np.c_[F,P]),('slow',one(np.c_[z,pair(z),C[:,0]*D[:,1]]))]:
 b=ls(M[:N],Y[:N]);e=abs(M@b-Y);print(n,e[:N].mean(0),e[N:].mean(0),abs((M@b-Y)[N:,1:]@H.T).mean(),b.T)
for k in range(3):
 y=Y[:,k]; A=np.c_[P,dot,dotL, z,C/(1+.22*D*D),D/(1+.22*C*C),C**2,D**2]; labels=[f'c{i}d{j}' for i in range(4) for j in range(4)]+['dot','dotL']+[f'{n}{i}' for n in ['z','Cr','Dr','C2','D2'] for i in range(4)]
 for M in [F,one(np.c_[z,pair(z)])]:
  for step in range(6):
   R=A[:N]-M[:N]@ls(M[:N],A[:N]); r=y[:N]-M[:N]@ls(M[:N],y[:N]); j=np.argmax(abs(r@R)/np.maximum(np.linalg.norm(R,axis=0),1e-7));M=np.c_[M,A[:,j]]; b=ls(M[:N],y[:N]);e=abs(M@b-y);print(k,labels[j],b[-1],e[:N].mean(),e[N:].mean())
