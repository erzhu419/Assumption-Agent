import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement

def aff(u): return np.c_[np.ones(len(u)),u]
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def quad(u): return np.column_stack([u[:,i]*u[:,j] for i,j in combinations_with_replacement(range(4),2)])

def coords(s,x):
 p=s['panel_a']; u=p['u'];v=p['v']; ii=np.linalg.norm(v-u,axis=1)<2;u=u[ii];v=v[ii]
 def ades(a,w): return aff(np.c_[a,np.sin(w*a)])
 def ar(w):
  a=ades(u,w[0]);return (a@ls(a,v)-v).ravel()
 w=least_squares(ar,[2.45],bounds=(1.5,3.3)).x[0]; c=ls(ades(u,w),v)
 z=np.linalg.solve(c[1:5].T,(x-c[0]).T).T
 for _ in range(25):
  j=c[1:5].T[None]+c[5:].T[None]*(w*np.cos(w*z))[:,None,:]
  z+=np.clip(np.linalg.solve(j,(x-ades(z,w)@c)[...,None])[...,0],-.3,.3)
 p=s['panel_b']; u=p['u'];v=p['v']
 def bd(a,p):return aff(a+p[0]*np.sin(p[1]*a))
 def br(p):
  a=bd(u,p);return (a@ls(a,v)-v).ravel()
 p=least_squares(br,[-.4,2.9],bounds=([-1,2],[0,4])).x
 return z,bd(z,p)@ls(bd(u,p),v)

def ridge(a,v,b,l):
 m=a.mean(0); sd=np.maximum(a.std(0),1e-10); a=(a-m)/sd;b=(b-m)/sd
 return (v.mean(0)+b@np.linalg.solve(a.T@a+l*np.eye(a.shape[1]), a.T@(v-v.mean(0))))

def solve_batch(study):
 t=study['train']; n=len(t['x']); z,q=coords(study,np.r_[t['x'],study['test']['x']]);
 a=np.c_[z,q,quad(z),quad(q),(z[:,:,None]*q[:,None,:]).reshape(len(z),-1)]
 yh=ridge(a[:n],t['y'],a[n:],.25); tr=ridge(a[:n],t['traj'],a[n:],.003)
 return dict(y_hat=yh,state_pred=tr,diagnostics_vec=np.array([yh.mean(),yh.std(),tr[:,-1].mean(),tr[:,-1].std()]))

def transforms(s,x):
 z,q=coords(s,x)
 def op(k,g,init):
  pan=s['panel_c'];u=pan['u'];v=pan[k]
  def d(u,p):return aff(u+p[0]*g(p[1]*u))
  def res(p):
   a=d(u,p);return (a@ls(a,v)-v).ravel()
  p=least_squares(res,init,max_nfev=50).x;c=ls(d(u,p),v)
  return lambda w:d(w,p)@c
 c=op('v1',np.tanh,[2,2.27]);d=op('v2',lambda w:np.cos(w)-1,[-.65,1.29]);return z,q,[z,q,c(z),d(z),c(q),d(q)]

def candidate(study, verbose=False):
 t=study['train'];n=len(t['x']);z,q,F=transforms(study,np.r_[t['x'],study['test']['x']]); y=t['y'];T=t['traj'];ts=study['eval_times'];basis=np.c_[-np.expm1(-ts),ts];amp=ls(basis,T.T).T
 def prod(u,v):return (u[:,:,None]*v[:,None,:]).reshape(len(u),-1)
 def standard(u):return (u-u[:n].mean(0))/np.maximum(u[:n].std(0),1e-9)
 def out(k,yy,alpha):
  m=k[:n,:n].mean(0);mm=k[:,:n].mean(1);k=k[:,:n]-m-mm[:,None]+m.mean(); yc=yy-yy.mean(0);g=np.linalg.solve(k[:n]+alpha*np.eye(n),np.c_[yc,np.eye(n)]);p=g[:,:yc.shape[1]];h=g[:,yc.shape[1]:].diagonal()-1/(n*alpha);return np.mean(abs(p/h[:,None]),0),yy.mean(0)+k[n:]@p
 Y=np.c_[y,T]; results=[]
 def record(a,yy=Y,tag='',penalties=(.003,.015,.06,.25,1.)):
  a=standard(a);k=a@a.T
  for pen in penalties:
   e,p=out(k,yy,pen);results.append((e,p,tag,pen))
 record(np.c_[z,q,quad(z),quad(q),prod(z,q)],tag='mix')
 for i,u in enumerate(F):
  for j,v in enumerate(F[:i+1]):
   record(np.c_[q,u,v,quad(q),prod(u,v)],tag=f'cross-{i}-{j}')
  u=standard(u);dist=np.maximum(0,((u*u).sum(1)[:,None]+(u*u).sum(1)-2*u@u.T));base=standard(np.c_[q,z,quad(q),quad(z)]); k0=base@base.T
  for ell in [1.5,2.5,4.5,7.]:
   for c in [.2,1.,5.]:
    k=k0+c*20*np.exp(-.5*dist/ell**2)
    for p in [.002,.025,.15]:
     e,w=out(k,Y,p);results.append((e,w,f'gp-{i}',p))
 # A shared low-rank bilinear residual, with independent affine readouts.
 L=aff(standard(q)); W=np.c_[y,amp[:,0]];R=[]
 for i,u0 in enumerate(F):
  for j,v0 in enumerate(F[:i+1]):
   u=standard(u0);v=standard(v0);B=prod(u,v)
   # SVD of unconstrained multi-response quadratic gives stable directions for the rank-one optimization.
   l=L[:n]; bm=B[:n]-l@ls(l,B[:n]); wm=W-l@ls(l,W); left,sv,right=np.linalg.svd(wm,full_matrices=False)
   b=ls(np.r_[bm,.3*np.eye(16)],np.r_[left[:,0]*sv[0],np.zeros(16)]).reshape(4,4);g,h,vt=np.linalg.svd(b);rng=np.random.default_rng(618)
   for form in ['bilin','tanh','rational','sin']:
    for reg in [.035,.16]:
     def design(p):
      a=u@p[:4];b=v@p[4:];f=a*b
      if form=='tanh':f=np.tanh(a)*np.tanh(b)
      if form=='rational':f=f/(1+a*a+b*b)
      if form=='sin':f=np.sin(a)*np.sin(b)
      return np.c_[L,f]
     def er(p):
      a=design(p)[:n];return np.r_[(a@ls(a,W)-W).ravel(),reg*p]
     starts=[np.r_[g[:,k],vt[k]]*sc for k in range(2) for sc in ([.5,1.2] if form!='bilin' else [1.])];
     rr=min((least_squares(er,p,max_nfev=55) for p in starts),key=lambda a:a.cost);a=design(rr.x); c=ls(a[:n],Y);e=a[:n]@c-Y
     # Penalize sensitivity to the fitted directions, not just to linear coefficients.
     jac=np.c_[a[:n],rr.jac[:n*2].reshape(n,2,8)[:,0,:]]; h0=np.sum(np.linalg.svd(jac,full_matrices=False)[0][:,:min(12,n-1)]**2,1)
     score=np.mean(abs(e)/np.maximum(.20,1-h0[:,None]),0);R.append((score,a[n:]@c,f'rank-{i}-{j}-{form}',reg))
 results+=R
 # Average distinct, highly ranked feature families to limit sensitivity to one grammar choice.
 def ensemble(which):
  key=lambda a:float(a[0][0] if which==0 else np.mean(a[0][1:]));ordered=sorted(results,key=key);chosen=[];seen=set()
  for a in ordered:
   if a[2] in seen:continue
   seen.add(a[2]);chosen.append(a)
   if len(chosen)>=10:break
  best=key(chosen[0]); wt=np.exp(-(np.array([key(a) for a in chosen])-best)/max(.16*best,.002)); wt/=sum(wt)
  if verbose:print('chosen',which,[(a[2],round(key(a),5)) for a in chosen])
  pred=np.tensordot(wt,np.array([a[1] for a in chosen]),axes=1)
  return pred
 pred=ensemble(0); pt=ensemble(1)
 yy=pred[:,0];tr=pt[:,1:];return dict(y_hat=yy,state_pred=tr,diagnostics_vec=np.array([yy.mean(),yy.std(),tr[:,-1].mean(),tr[:,-1].std()]))

def multiscale(study):
 from scipy.optimize import minimize
 from scipy.linalg import cho_factor, cho_solve
 t=study['train']; n=len(t['x']); z,q,F=transforms(study,np.r_[t['x'],study['test']['x']]); y=t['y']; T=t['traj']; Y=np.c_[y,T]; eps=1e-9
 def std(a):
  d=a-a[:n].mean(0); return d/np.maximum(a[:n].std(0),eps)
 def gram(a):
  a=std(a); return a@a.T/a.shape[1]
 def prod(a,b): return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
 def rbf(a,l):
  a=std(a); d=(a*a).sum(1); return np.exp(-np.maximum(0,d[:,None]+d-2*a@a.T)/l)
 zz=std(z); qq=std(q)
 groups=[gram(q),gram(quad(q)),gram(z),gram(quad(z)),gram(prod(z,q)),gram(qq**3),gram(zz**3)]
 for a in [z,q]:
  for w in [.65,1.25,2.15,2.9,4.3]:
   k1=gram(np.sin(w*a)); k2=gram(np.cos(w*a)); groups.extend([k1,k2,k1*k1,k1*k2,k2*k2,k1*groups[0],k2*groups[0]])
  a=std(a); k=gram(a); groups += [k*k*k,k**4,gram(prod(a,a*a))]
 for a in F:
  k=gram(a); groups.extend([k,gram(quad(a)),k*groups[0],k*groups[1]])
  for l in [4,12,36]: groups.extend([rbf(a,l),rbf(a,l)*groups[0]])
 for a in F[2:]:
  groups.extend([gram(prod(a,z)),gram(prod(a,q)),gram(prod(a,F[2])),gram(prod(a,F[3]))])
 K=np.stack(groups); m=K[:,:n,:n].mean(1); mm=K[:,:,:n].mean(2); K=K[:,:,:n]-m[:,None,:]-mm[:,:,None]+m.mean(1)[:,None,None]; K/=np.maximum(np.trace(K[:,:n],axis1=1,axis2=2)/n,.05)[:,None,None]
 U=K[:,:n].copy().reshape(len(K),-1); E=np.eye(n); yc=Y-Y.mean(0); prior=np.full(len(K),.0015); prior[:7]=[.48,.13,.015,.022,.04,.01,.01]; prior/=prior.sum(); pred=[]
 for trade in [0.,.35]:
  for reg in [.025,.085]:
   def loss(g,ret=False):
    ww=np.exp(g[:-1])*prior; noise=np.exp(g[-1]); a=(ww@U).reshape(n,n)+noise*E; inv=cho_solve(cho_factor(a,check_finite=False),E,check_finite=False); v=inv@yc; h=np.maximum(inv.diagonal()-1/(n*noise),eps); e=v/h[:,None]; sc=np.maximum(Y.std(0),eps); weights=np.r_[1-trade,np.full(T.shape[1],trade/T.shape[1])]; ed=e/sc; rr=np.sqrt(ed*ed+.025**2); val=(rr@weights).mean();
    if ret: return Y.mean(0)+np.tensordot(ww,K[:,n:],axes=1)@v
    d=ed/rr*weights/sc/n; dv=d/h[:,None]; dh=-np.sum(d*e,axis=1)/h; ag=-(inv@dv)@v.T-(inv*dh)@inv; grad=ww*(U@ag.ravel()); gn=noise*np.trace(ag)+dh.sum()/(n*noise); penalty=reg*(np.dot(prior,g[:-1]**2)+.02*(g[-1]+5)**2)
    return val+penalty,np.r_[grad+2*reg*prior*g[:-1],gn+.04*reg*(g[-1]+5)]
   r=min((minimize(loss,np.r_[np.zeros(len(K)),r],jac=True,method='L-BFGS-B',bounds=[(-5,6)]*len(K)+[(-11,-.4)],options={'maxiter':110}) for r in [-4.,-7.]),key=lambda r:r.fun); pred.append(loss(r.x,True))
 p=np.mean(pred,axis=0); yy=p[:,0]; tr=p[:,1:]; return dict(y_hat=yy,state_pred=tr,diagnostics_vec=np.array([yy.mean(),yy.std(),tr[:,-1].mean(),tr[:,-1].std()]))
