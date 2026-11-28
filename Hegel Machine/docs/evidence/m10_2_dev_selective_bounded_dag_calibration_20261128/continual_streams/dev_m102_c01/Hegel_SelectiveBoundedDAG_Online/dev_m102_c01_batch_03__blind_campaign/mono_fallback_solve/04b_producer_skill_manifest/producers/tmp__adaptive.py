import numpy as np
import pickle
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def calibrate(p):
 u,v=p['u'],p['v']; i,j=np.triu_indices(len(u),1); d=u[i]-u[j]; scale=np.median((v[i]-v[j])/np.where(abs(d)>1e-10,d,np.inf),axis=0); offset=np.median(v-u*scale,axis=0)
 for _ in range(3):
  r=np.linalg.norm(v-u*scale-offset,axis=1); keep=r<max(6*np.median(r),.0005)
  if keep.sum()<6: break
  for j in range(u.shape[1]): scale[j],offset[j]=np.linalg.lstsq(np.c_[u[keep,j],np.ones(keep.sum())],v[keep,j],rcond=None)[0]
 return scale,offset

def coords(s,x):
 a,b=calibrate(s['panel_a']); z=(x-b)/a; p=s['panel_b']; t=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; return [x,z,z@t[1:]+t[0],x@t[1:]+t[0]]

def fitgp(z,q,y,kind,objective='ml',sig=.009):
 n,d=z.shape; N=n+len(q); Z=np.vstack([z,q]); D=Z[:,None,:]-Z[None,:,:]; D2=D**2; I=np.eye(n); mu=y.mean(axis=0); y=(y-mu).reshape(n,-1); dot=Z@Z.T; B0=np.ones((N,N)); B1=dot; B2=dot**2
 full=kind in ('full','fulladd','fullm52'); add=kind in ('add','fulladd','diagadd'); isot=kind in ('iso','add'); m52=kind=='fullm52'; ij=np.tril_indices(d,-1); nf=len(ij[0]) if full else 0; nd=1 if isot else d; nk=5 if add else 4
 def cov(p,allrows=False):
  R=D if allrows else D[:n,:n]; T=D2 if allrows else D2[:n,:n]; sl=np.s_[:,:] if allrows else np.s_[:n,:n]; v=np.exp(p[:nd]); W=R*v; G=W
  if full:
   L=np.diag(v); L[ij]=p[nd:nd+nf]; G=R@L
  S=np.sum(G**2,axis=-1)
  if m52:
   A=np.sqrt(5*S); K=(1+A+5*S/3)*np.exp(-A); gradk=-(5/6)*(1+A)*np.exp(-A)
  else: K=np.exp(-S); gradk=-K
  bases=[B0[sl],B1[sl],B2[sl],K]; parts=None
  if add:
   parts=np.exp(-T*v**2); bases.append(parts.mean(axis=-1))
  amps=np.exp(p[nd+nf:nd+nf+nk]); C=sum(a*b for a,b in zip(amps,bases));
  if allrows:return C
  noise=np.exp(p[-1]); C=C+noise*I; J=2*amps[3]*gradk[:,:,None]*G*W
  if add:J-=2*amps[4]/d*parts*T*v**2
  deriv=[J.sum(axis=-1)] if isot else list(J.transpose(2,0,1));
  if full: deriv += [2*amps[3]*gradk*G[:,:,j]*R[:,:,i] for i,j in zip(*ij)]
  return C,np.stack(deriv+[a*b for a,b in zip(amps,bases)]+[noise*I])
 def loss(p):
  C,g=cov(p); CF=cho_factor(C); alpha=cho_solve(CF,y); inv=cho_solve(CF,I)
  if objective=='ml':
   f=.5*np.sum(alpha*y)+y.shape[1]*np.log(CF[0].diagonal()).sum(); H=.5*(y.shape[1]*inv-alpha@alpha.T)
  else:
   a=inv.diagonal()[:,None]; e=alpha/a; J=(inv@((e/a)@alpha.T)); V=(e*e/a).sum(axis=1); f=.5*np.sum(e*e)*n; H=((inv*V)@inv-.5*(J+J.T))*n
  diff=p-p0; return f+.09*np.sum(diff*diff), np.einsum('ijk,jk->i',g,H)+.18*diff
 p0=np.r_[np.full(nd,-.55),np.zeros(nf),np.log([1,.5,.025,1]+([.25] if add else [])),2*np.log(sig)]; bounds=[(-4,1)]*nd+[(-2,2)]*nf+[(-11,4)]*nk+[(np.log(sig**2)-2,np.log(sig**2)+3)]; opt=minimize(loss,p0,jac=True,bounds=bounds,method='L-BFGS-B',options={'maxiter':115,'ftol':1e-8}); p=opt.x
 C=cov(p,True); pred=C[n:,:n]@cho_solve(cho_factor(C[:n,:n]+np.exp(p[-1])*I),y)+mu
 return pred,p,opt.fun

if __name__=='__main__':
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; T=s['train']['traj']; U,S,V=np.linalg.svd(T,full_matrices=False); A=T@V[:2].T
 folds=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(821)
 for _ in range(6):
  i=rng.permutation(len(y)); folds.append((i[:52],i[52:]))
 for zz,z in enumerate(coords(s,x)):
  for kind in ['iso','diag','diagadd','full','fulladd','fullm52']:
   if zz in [0,3] and kind not in ['diag','fulladd']: continue
   for ob in ['ml','loo']:
    errs=[]
    for i,j in folds:
     yp,p,f=fitgp(z[i],z[j],y[i],kind,ob); errs.append(np.abs(yp.ravel()-y[j]).mean())
    print(zz,kind,ob,'gate, mean, max',np.round([errs[0],np.mean(errs),max(errs)],5),np.round(p,2),flush=True)
 for zz,z in enumerate(coords(s,x)):
  for kind in ['iso','diag','diagadd','full']:
   errs=[]
   for i,j in folds:
    _,_,v=np.linalg.svd(T[i],full_matrices=False); scale=np.std(T[i]@v[:2].T,axis=0); pp,p,f=fitgp(z[i],z[j],T[i]@v[:2].T/scale,kind,'ml',.016); errs.append(np.abs((pp*scale)@v[:2]-T[j]).mean())
   print('traj',zz,kind,np.round([errs[0],np.mean(errs),max(errs)],5),flush=True)
