import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(x): return np.c_[np.ones(len(x)),x]
def rms(x): return np.mean(x*x)**.5
p=s['panel_a']; u,v=p['u'],p['v']; cal=[]
for x,y in zip(u.T,v.T):
 i,j=np.triu_indices(len(x),1); m=np.median((y[i]-y[j])/(x[i]-x[j])); r=abs(y-m*x-np.median(y-m*x)); k=r<6*np.median(r); cal.append(ls(aff(x[k,None]),y[k]))
o,g=np.array(cal).T
z=(s['train']['x']-o)/g
p=s['panel_c']; u=p['u']; t=s['eval_times']; Y=s['train']['traj']; y=s['train']['y']
def branch(fn,v,q,bounds):
 def err(q):
  f=aff(fn(u,*q)); return (f@ls(f,v)-v).ravel()
 fit=least_squares(err,q,bounds=bounds,gtol=1e-13,ftol=1e-13,xtol=1e-13); F=aff(fn(u,*fit.x)); M=ls(F,v); print('branch',fit.x,rms(fit.fun),'matrix',M,'singular',np.linalg.svd(M[1:],compute_uv=False)); return aff(fn(z,*fit.x))@M
v=p['v1']
def profile(f, v, orth=True):
 if not orth: return aff(f)@ls(aff(f),v)-v,None,None
 fm=f.mean(0); vm=v.mean(0); a=f-fm; b=v-vm
 h,d,j=np.linalg.svd(a.T@b); m=(h@j)*d.sum()/np.sum(a*a); off=vm-fm@m
 return f@m+off-v,m,off
base=lambda u,a,k:a*u+np.tanh(k*u)
funcs=[('base',base,[.5,2.3]),('coordinate slopes',lambda u,a,k:np.array(a)*u+np.tanh(k*u),None),('coordinate rates',lambda u,a,k:a*u+np.tanh(np.array(k)*u),None),('shift',lambda u,a,k,c:a*u+np.tanh(k*u+c),[.5,2.3,0]),('cubic',lambda u,a,k,c:base(u,a,k)+c*u**3,[.5,2.3,0]),('quintic',lambda u,a,k,c,d:base(u,a,k)+c*u**3+d*u**5,[.5,2.3,0,0]),('softsign',lambda u,a,k:a*u+k*u/np.sqrt(1+(k*u)**2),[.5,2.3]),('atan',lambda u,a,k:a*u+np.arctan(k*u),[.5,2.3]),('cubic inside',lambda u,a,k,c:a*u+np.tanh(k*u+c*u**3),[.5,2.3,0]),('rational',lambda u,a,k:a*u+u/(1+k*u**2),[.5,1]),('sin plus tanh',lambda u,a,k,c,l:base(u,a,k)+c*np.sin(l*u),[.5,2.3,.01,3]),('tanh plus scaled rational',lambda u,a,k,c,l:base(u,a,k)+c*u/(1+l*u*u),[.5,2.3,.01,1]),('input norm coupling',lambda u,a,k,c:base(u,a,k)+c*u*np.sum(u*u,axis=1)[:,None],[.5,2.3,0])]
for name,fn,q in funcs:
 if name=='coordinate slopes': fn=lambda u,*p:base(u,np.array(p[:4]),p[4]); q=[.5]*4+[2.3]
 if name=='coordinate rates': fn=lambda u,*p:base(u,p[0],np.array(p[1:])); q=[.5]+[2.3]*4
 with np.errstate(invalid='ignore',divide='ignore'):
  try:
   opt=least_squares(lambda p:profile(fn(u,*p),v)[0].ravel(),q,max_nfev=400); print(name,np.round(opt.x,7),rms(opt.fun))
  except (ValueError,np.linalg.LinAlgError): pass
opt=least_squares(lambda p:profile(base(u,*p),v)[0].ravel(),[.5,2.3]); e,m,_=profile(base(u,*opt.x),v); e=e@np.linalg.inv(m); print('unrotated residual against input (each column sorted by u)'); print(np.round(np.column_stack([np.c_[u[:,i],e[:,i]][np.argsort(u[:,i])] for i in range(4)]),5)); print('resid-product correlations',np.round(np.corrcoef(np.c_[e,u,u**2,u[:,[0,0,0,1,1,2]]*u[:,[1,2,3,2,3,3]]].T)[:4,4:],2))
A=branch(base,p['v1'],[.5,2.3],([0,.4],[3,5])); B=branch(lambda u,k:np.cos(k*u),p['v2'],[2.4],([1.9],[2.9])); L=aff(np.c_[A,B]); D=aff(np.c_[A,B,np.sum(A*B,1)]); S=aff(np.c_[A,B,A[:,0]*B[:,1]]);
def modes(i,j,fast,slow):
 q=least_squares(lambda q: proj(q,i,fast,slow)[0].ravel(),[.4,1.8],bounds=([.08,1],[.9,3])).x
 e,c=proj(q,i,fast,slow); f=-np.expm1(-t[:,None]*q)
 st=(slow[j]@c[:slow.shape[1]])[:,None]*f[:,0]+(fast[j]@c[slow.shape[1]:])[:,None]*f[:,1]
 ey=D[j]@ls(D[i],y[i])-y[j]; es=st-Y[j]; return [np.mean(abs(ey)),np.mean(abs(es)),max(np.mean(abs(es),axis=0))],q,c

def proj(q,i,fast,slow):
 f=-np.expm1(-t[:,None]*q); x=np.c_[(slow[i,None,:]*f[None,:,0,None]).reshape(len(i)*len(t),-1),(fast[i,None,:]*f[None,:,1,None]).reshape(len(i)*len(t),-1)]; c=ls(x,Y[i].ravel()); return x@c-Y[i].ravel(),c
for fast,slow in [(D,S),(D,np.c_[S,D[:,-1]]),(np.c_[D,S[:,-1]],S),(np.c_[D,S[:,-1]],np.c_[S,D[:,-1]])]:
 out=[]
 for i,j in [(np.arange(56),np.arange(56,72))]+[(a[:56],a[56:]) for a in [np.random.default_rng(k).permutation(72) for k in range(8)]]:
  ee,q,c=modes(i,j,fast,slow); out.append(ee)
 print('mode widths',fast.shape[1],slow.shape[1],'ordered',out[0],'worst',np.max(out,axis=0),'rate/coefs',q,np.round(c,4))
