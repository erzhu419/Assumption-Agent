from graph08 import *
from scipy.linalg import svd
from scipy.optimize import least_squares
D=setup(s)(s['train']['x']); z,b,p,r=[D[k] for k in ('z','b','p','r')]; t=s['eval_times']; T=s['train']['traj']; y=s['train']['y']; N=52
mul=lambda a,b:np.einsum('ni,nj->nij',a,b).reshape(len(a),-1)
dot=lambda a,b:np.sum(a*b,axis=1)
q=np.c_[z[:,0]*z[:,1],z[:,0]*z[:,3],z[:,1]*z[:,2],z[:,2]*z[:,3]]
idx=sorted({tuple(sorted((i,j,k))) for i in range(4) for j,k in [(0,1),(0,3),(1,2),(2,3)]})
C=np.column_stack([np.prod(z[:,i],axis=1) for i in idx]); U=mul(p,r); dp=dot(p,r)
B=aff(z,p,r); Y=np.c_[y,T]; F=np.c_[B,dp,C,U[:,[i for i in range(16) if i!=15]]]; W=ls(F[:N],Y[:N]); E=F@W-Y
print('full rank',np.linalg.matrix_rank(F),F.shape,'errors',np.mean(abs(E[N:]),axis=0))
for k,a in zip(['const']+[f'{k}{i}' for k in 'zpr' for i in range(4)]+['dot']+['c'+''.join(map(str,j)) for j in idx]+[f'pr{i//4}{i%4}' for i in range(15)],W): print(k,np.round(a,4))
Z={}
for j,a in [('z',z),('b',b),('p',p),('q',q),('r',r)]:
 for k,c in [('z',z),('b',b),('p',p),('q',q),('r',r)]:
  Z[j+'.'+k]=dot(a,c)
  for h,col in enumerate(mul(a,c).T): Z[f'{j}{h//4}*{k}{h%4}']=col
for i,col in zip(idx,C.T):Z['mono'+str(i)]=col
L=list(Z); Lmat=np.column_stack(list(Z.values()))
for base_name,base,goal in [('y',np.c_[B,dp],y[:,None]),('T',np.c_[B,dp],T),('T after cubic',np.c_[B,dp,C],T)]:
 print('selection',base_name)
 for j in range(7):
  res=goal[:N]-base[:N]@ls(base[:N],goal[:N]); F=Lmat[:N]-base[:N]@ls(base[:N],Lmat[:N]); den=np.linalg.norm(F,axis=0); score=np.linalg.norm(F.T@res,axis=1)/np.maximum(den,1e-12); score[den<1e-7]=0; k=np.argmax(score);base=np.c_[base,Lmat[:,k]]; e=goal-base@ls(base[:N],goal[:N]);print(L[k],np.round(np.sqrt(np.mean(e[:N]**2)),6),np.round(np.mean(abs(e[N:])),6))
for pair in [('b','p'),('z','p'),('b','q'),('z','q')]:
 name='.'.join(pair); A=np.c_[B,dp,Z[name],p[:,0]*r[:,1]]; E=Y-A@ls(A[:N],Y[:N]);print(name,'cv',np.mean(abs(E[N:]),axis=0),'train',np.std(E[:N],axis=0))
