import pickle,numpy as np
from itertools import combinations_with_replacement
from scipy.optimize import least_squares
np.set_printoptions(precision=5,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; n=len(x)
def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,k],axis=1) for j in range(1,d+1) for k in combinations_with_replacement(range(x.shape[1]),j)])
def ls(X,Y):return np.linalg.lstsq(X,Y,rcond=1e-10)[0]
a=s['panel_a']; A=poly(a['u'],1); V=a['v']; err=np.linalg.norm(V-a['u'],axis=1); print('a shift norms',np.sort(err)); keep=err<2; C=ls(A[keep],V[keep]); print('trimmed affine',C,'errors',np.sort(np.linalg.norm(V-A@C,axis=1))); print('trimmed quadratic rms',np.std((V[keep]-poly(a['u'][keep],2)@ls(poly(a['u'][keep],2),V[keep])),axis=0)); ax=poly(x,1)@C
b=s['panel_b']; B=ls(poly(b['u'],1),b['v']); bx=poly(ax,1)@B
T=lambda r: (1-np.exp(-t[:,None]*r)).T
for rates in [[1.6],[1.6,.2],[1.6,.2,.1]]:
 Q=T(rates); Z=ls(Q.T,z.T).T; print('rates',rates,'traj residual',np.std(z-Z@Q),'amp sd',Z.std(0)); print('amp coefficients',ls(poly(ax,1),Z).T,'amp linear residual',np.std(Z-poly(ax,1)@ls(poly(ax,1),Z),axis=0)); print('amp correlations in bx',np.corrcoef(np.c_[Z,bx].T)[:len(rates),len(rates):]);
f=least_squares(lambda r:(z-ls(T(r).T,z.T).T@T(r)).ravel(),[1.6,.2],bounds=([.5,.01],[3,.8])); print('param profile',f.x,np.std(f.fun)); Z=ls(T([1.6,.2]).T,z.T).T
for d in range(1,5):
 for nm,F in [('x',x),('ax',ax),('Z',np.c_[ax,Z])]:
  Q=poly(F,d); m=56; pred=Q@ls(Q[:m],np.c_[y,Z][:m]); E=pred-np.c_[y,Z]; print('poly',nm,d,'in',np.mean(abs(E[:m]),0),'cv',np.mean(abs(E[m:]),0))
