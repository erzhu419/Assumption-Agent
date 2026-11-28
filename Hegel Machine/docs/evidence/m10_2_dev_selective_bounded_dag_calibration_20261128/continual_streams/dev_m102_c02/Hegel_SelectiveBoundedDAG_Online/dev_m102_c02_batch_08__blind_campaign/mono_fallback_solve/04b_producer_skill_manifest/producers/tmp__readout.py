from graph import *
from scipy.optimize import least_squares
from scipy.linalg import solve
X=np.r_[s['train']['x'],s['test']['x']]; Z=calib(s['panel_a'],X); Y=np.c_[s['train']['y'],s['train']['traj']]; n=len(Y)
pb,pc=s['panel_b'],s['panel_c']; B=configs(pb,pb['v'])[0][-2]; C=configs(pc,pc['v1'])[0][-2]; V=configs(pc,pc['v2'])[0][-2];
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def calc(F,ii,jj,reg=0):
 a=F[ii]; scale=np.maximum(a.std(0),1e-4); scale[0]=1; A=a/scale; pen=np.eye(A.shape[1]); pen[0,0]=0; beta=ls(np.r_[A,reg*pen],np.r_[Y[ii],np.zeros((A.shape[1],9))])/scale[:,None];return F[jj]@beta,beta
rng=np.random.default_rng(550); splits=[(np.arange(52),np.arange(52,72))];
for i in range(8):
 p=rng.permutation(n);splits.append((p[:52],p[52:]));
def score(F,reg=0):
 e=np.array([abs(calc(F,ii,jj,reg)[0]-Y[jj]).mean(0) for ii,jj in splits]); return np.array([e[0,0],e[0,1:].mean(),e[:,0].mean(),e[:,0].max(),e[:,1:].mean()])
b,c,d=B(Z),C(Z),V(Z); F=lin(np.c_[b,c,cross(b,c)]); E=Y[:52]-calc(F,np.arange(52),np.arange(52))[0];print('baseline',score(F),'res train',abs(E).mean(0),'corr y state',np.corrcoef(E.T)[0]); print('coeff y/state',calc(F,np.arange(52),np.arange(52))[1][:,[0,-1]].T)
D={'b':b,'c':c,'d':d,'z':Z,'rat':Z/(1+Z*Z),'tanh':np.tanh(1.25*Z),'tanh2':np.tanh(2.5*Z),'atan':np.arctan(1.25*Z),'cubic':Z**3,'sq':Z**2,'cyc':cyc(Z),'b2':b*b,'c2':c*c,'d2':d*d,'thb':np.tanh(b),'thc':np.tanh(c),'ab':abs(b),'ac':abs(c)}
L=[]
for na,a in D.items():
 for nb,h in D.items():
  if na==nb:continue
  for reg in [.08,.4,1.]:
   for nm,G in [('cross',lin(np.c_[a,h,cross(a,h)])),('add',np.c_[F,a]),('crossadd',np.c_[F,cross(a,h)]),('crossdiag',np.c_[F,a*h])]:
    r=score(G,reg);L.append((na,nb,nm,reg,r))
for j in [0,2,3,4]:print('best',j,*sorted(L,key=lambda t:t[-1][j])[:7],sep='\n')
# Kernel corrections around the structured readout; profile an unpenalized baseline.
for space,w in [('z',Z),('b',b),('c',c),('bc',np.c_[b,c]),('resid',np.c_[Z,b,c]),('d',d)]:
 w=w/w[:n].std(0);dist=((w[:,None]-w[None,:])**2).mean(2)
 for ell in [.25,.5,1.,2.]:
  K=np.exp(-dist/(2*ell**2));
  for ridge in [.001,.03,.3,1.,5.]:
   r=[]
   for ii,jj in splits:
    M=K[np.ix_(ii,ii)]+ridge*np.eye(len(ii)); A=F[ii]; G=solve(M,np.c_[A,Y[ii]],assume_a='pos'); p=ls(A.T@G[:,:A.shape[1]], A.T@G[:,A.shape[1]:]); y=F[jj]@p+K[np.ix_(jj,ii)]@(G[:,A.shape[1]:]-G[:,:A.shape[1]]@p);r.append(abs(y-Y[jj]).mean(0))
   e=np.array(r);L.append((space,ell,'kernel',ridge,np.array([e[0,0],e[0,1:].mean(),e[:,0].mean(),e[:,0].max(),e[:,1:].mean()])))
for j in [0,2,3,4]:print('best with kernel',j,*sorted(L,key=lambda t:t[-1][j])[:6],sep='\n')
