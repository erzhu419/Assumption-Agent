from graph08 import *
from scipy.optimize import least_squares
D=setup(s)(s['train']['x']); z,p,r=[D[k] for k in ('z','p','r')]; t=s['eval_times']; y=s['train']['y']; T=s['train']['traj']; N=52
pairs=[(0,1),(0,3),(1,2),(2,3)]; idx=sorted({tuple(sorted((i,j,k))) for i in range(4) for j,k in pairs}); C=np.column_stack([np.prod(z[:,i],axis=1) for i in idx]);groups=[np.c_[p,np.sum(p*r,axis=1)],np.c_[r,p[:,0]*r[:,1]],np.c_[z,C]]
F=aff(*groups); W=ls(F[:N],np.c_[y,T][:N]);
for k,ww in enumerate([W[1:6,1:],W[6:11,1:],W[11:,1:]]):
 u,ss,h=np.linalg.svd(ww,full_matrices=False); print(k, ss[:4],h[0]/h[0,-1], 'fit coeff', u[:,0]*ss[0]*h[0,-1]);
print('y third weights',W[11:,0]);print('direct scalar interactions P/z',np.linalg.lstsq(np.column_stack([np.prod(z[:,i],axis=1) for i in idx]),C@W[15:,0],rcond=1e-10)[0]);
for key,fun,start in [('sqrt',lambda a:np.sqrt(t),[]),('power',lambda a:t**a[0],[.5]),('log',lambda a:np.log1p(a[0]*t),[1]),('shifted power',lambda a:(t+a[1])**a[0]-a[1]**a[0],[.5,.05]),('rat-sqrt',lambda a:np.sqrt(t/(1+a[0]*t)),[.1]),('sqrt+linear',lambda a:np.sqrt(t)+a[0]*t,[.1]),('free',lambda a:np.r_[a,1.],list(np.sqrt(t[:-1]/t[-1])))]:
 def mat(a,i,group=groups):
  h=[-np.expm1(-a[0]*t),-np.expm1(-a[1]*t),fun(a[2:])];return np.column_stack([(g[i,None,:]*hh[None,:,None]).reshape(len(i)*len(t),-1) for g,hh in zip(group,h)])
 i=np.arange(N);j=np.arange(N,len(y));
 def err(a):
  M=mat(a,i); tar=T[i].ravel();return M@ls(M,tar)-tar
 opt=least_squares(err,[1.85,.55,*start],bounds=(1e-6,15),gtol=1e-12,ftol=1e-12); M=mat(opt.x,i); coef=ls(M,T[i].ravel()); E=(mat(opt.x,j)@coef).reshape(-1,len(t))-T[j]; print(key,opt.x,np.std(opt.fun),np.mean(abs(E)),np.mean(abs(E),axis=0),'cubic:y ratios',np.round(coef[10:]/W[11:,0],3));
