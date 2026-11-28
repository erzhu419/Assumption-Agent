import numpy as np
import pickle
from scipy.linalg import cho_factor,cho_solve
from itertools import product
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


def _baseline_scalar(z, q, y):
    """Average smooth, soft-sign, and pairwise spatial response models."""
    Z=np.concatenate((z,q)); n=len(z)
    D=(Z[:,None,:]-Z[None,:,:])**2
    L=Z@Z.T
    F=Z/(1+.83*np.abs(Z)); G=F@F.T
    A=np.exp(-.44*D)
    pairs=sum(A[:,:,i]*A[:,:,j] for i in range(Z.shape[1]) for j in range(i))
    smooth=1+.35*L+np.exp(-.44*D.sum(axis=-1))
    def predict(C,r):
        return C[n:,:n]@cho_solve(cho_factor(C[:n,:n]+r*np.eye(n)),y)
    return (.66*predict(smooth+.2*(1+G)*(1+L),.0001)
            +.26*predict(smooth+.2*pairs*(1+L),.0001)
            +.08*predict(1+.4*L+np.exp(-.25*D.sum(axis=-1)),.000003))

def _chart(z, mode, h):
    if mode==1:return np.arcsinh(h*z)/h
    if mode==2:return z/(1+h*abs(z))
    if mode==3:return np.sign(z)*np.log1p(h*abs(z))/h
    if mode==4:return z/(1+h*z*z)
    return z

def _gram(z, spec):
    mode,h,kind,a,degree,b,c=spec
    q=_chart(z,mode,h)
    L=z@z.T; M=q@q.T; D=cdist(q,q,'sqeuclidean'); D0=cdist(z,z,'sqeuclidean')
    U=(1+.38*L)**degree
    if kind==0: K=np.exp(-a*D)
    elif kind==1: K=(1+a*D)**(-1)
    elif kind==2:
        R=np.sqrt(5*a*D); K=(1+R+R**2/3)*np.exp(-R)
    elif kind==3: K=np.exp(-a*D)*(1+b*L); b=0
    elif kind==4:
        A=np.exp(-a*(q[:,None,:]-q[None,:,:])**2); S=A.sum(2); K=(S*S-(A*A).sum(2))/12*(1+b*L); b=0
    elif kind==5: K=np.exp(-a*D0)+b*(1+M)*(1+L); b=0
    elif kind==6: K=np.exp(-a*D0)+b*((1+M)**2+.2*M*L); b=0
    elif kind==7: K=np.exp(-a*D0)+b*np.exp(-a*D); b=0
    elif kind==8:
        V=1/(1+c*np.sum(q*q,axis=1)); K=np.exp(-a*D0)+b*(1+.55*M)**3*V[:,None]*V[None,:]; b=0
    elif kind==9:
        V=np.array([v for v in product([-1,0,1],repeat=z.shape[1]) if any(v) and next(t for t in v if t)!=-1]); V=V/np.linalg.norm(V,axis=1)[:,None]; P=q@V.T
        S=np.sin(a*P); C=np.cos(a*P); K=np.exp(-c*D0)+b*(S@S.T+C@C.T)*(1+.4*L)/len(V); b=0
    elif kind==10:
        G=np.sqrt(np.maximum(D0,0)); K=np.exp(-a*D0)+b*np.exp(-c*G)*(1+.25*L); b=0
    else:
        P=np.tanh(h*z); J=P@P.T; K=np.exp(-a*D0)+b*(1+J)*(1+L); b=0
    return U+K+b*M

def _prediction(C,i,j,y,r):
    return C[np.ix_(j,i)]@cho_solve(cho_factor(C[np.ix_(i,i)]+r*np.eye(len(i))),y)

def solve_batch(study: dict) -> dict:
    train=study['train']; x=np.asarray(train['x']); q=np.asarray(study['test']['x']); T=np.asarray(train['traj']); n=len(x)
    if not len(q): return {'y_hat':np.empty(0),'state_pred':np.empty((0,len(study['eval_times']))),'diagnostics_vec':np.zeros(4)}
    Z=coords(study,np.concatenate((x,q)))[2]
    charts=coords(study,np.concatenate((x,q)))
    y=np.asarray(train['y'])
    yp=np.zeros(len(q))
    for model, weight in [((2, (2, 0.22, 0, 0.46, 0, 0.18, 0.22), 4e-05), 0.4219488), ((2, (2, 1.4, 11, 0.23, 2, 0.8, 0.22), 4e-05), 0.2272032), ((1, (2, 0.22, 9, 2.25, 1, 0.8, 0.11), 4e-05), 0.123648), ((1, (0, 1, 9, 2.25, 0, 0.035, 0.11), 2e-06), 0.1472), ((3, (2, 1.4, 11, 0.23, 2, 0.8, 0.22), 0.00023), 0.08)]:
        if model is None:
            pred=_baseline_scalar(Z[:n],Z[n:],y)
        else:
            index,spec,r=model
            pred=_prediction(_gram(charts[index],spec),np.arange(n),np.arange(n,n+len(q)),y,r)
        yp+=weight*pred
    _,_,V=np.linalg.svd(T,full_matrices=False); V=V[:2]; A=T@V.T
    scales=np.maximum(A.std(axis=0),1e-8)
    P=fitgp(Z[:n],Z[n:],A/scales,'full','ml',.016)[0]
    state=(P*scales)@V
    return {'y_hat':yp,'state_pred':state,'diagnostics_vec':np.array([yp.mean(),yp.std(),state[:,-1].mean(),state[:,-1].std()])}
