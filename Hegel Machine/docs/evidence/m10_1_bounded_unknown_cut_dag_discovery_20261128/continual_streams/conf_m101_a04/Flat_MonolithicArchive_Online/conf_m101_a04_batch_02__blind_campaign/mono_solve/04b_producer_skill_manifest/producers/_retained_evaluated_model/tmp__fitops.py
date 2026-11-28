from probe import *
from scipy.optimize import least_squares

def bas(u,w,k):
 z=u*w
 if k=='tanh': h=np.tanh(z)
 if k=='rational': h=u/(1+w*u*u)
 if k=='atan': h=np.arctan(z)
 if k=='sin': h=np.sin(z)
 if k=='sin2': h=np.c_[np.sin(z),np.sin(2*z)]
 if k=='cos': h=np.cos(z)
 if k=='erf':
  from scipy.special import erf
  h=erf(z)
 return np.c_[aff(u),h]

def op(u,v,x,k,w0,mask=None,shared=False):
 if mask is None: mask=np.ones(len(u),bool)
 a,b=u[mask],v[mask]
 def residual(w):
  f=bas(a,w,k); return (f@ls(f,b)-b).ravel()
 fit=least_squares(residual,np.full(1 if shared else 4,w0),bounds=(.15,8),gtol=1e-12)
 f=bas(a,fit.x,k)
 return bas(x,fit.x,k)@ls(f,b),fit.x,np.std(fit.fun)

def features(s):
 m,o,_=cal(s['panel_a']); X=np.r_[s['train']['x'],s['test']['x']]; x=(X-o)/np.diag(m)
 p=s['panel_b']; b=aff(x)@ls(aff(p['u']),p['v']); pb=ls(aff(p['u']),p['v']); bx=(x-pb[0])@np.linalg.inv(pb[1:]); p=s['panel_c']; u=p['u']; c=op(u,p['v1'],x,'tanh',2.2)[0]; d=op(u,p['v2'],x,'cos',2.4,shared=True)[0]
 return X,x,b,bx,c,d

if __name__=='__main__':
 p=s['panel_c']; u=p['u']; v=p['v1']; mask=np.arange(len(u))%4!=0
 for k,w0 in [('tanh',2.2),('rational',1.9),('atan',2.2),('sin',3.4),('sin2',3),('erf',1.5)]:
  _,w,r=op(u,v,u,k,w0); yh,wcv,_=op(u,v,u,k,w0,mask); print(k,w,r,'cv',np.std(yh[~mask]-v[~mask]),wcv)
 X,x,b,bx,c,d=features(s); n=len(s['train']['y']); T=np.c_[s['train']['y'],s['train']['traj']]; tr=np.arange(52); te=np.arange(52,n)
 def score(f):
  w=ls(f[tr],T[tr]); e=abs(f[:n]@w-T); return np.round([e[tr,0].mean(),e[te,0].mean(),e[tr,1:].mean(),e[te,1:].mean()],5)
 def quad(x):return np.c_[aff(x),*[x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(x.shape[1]),2)]]
 for name,f in [('c',aff(c)),('d',aff(d)),('x',aff(x)),('cd',np.c_[aff(c),d]),('cdmatch',np.c_[aff(c),d,c*d]),('cdmatchsum',np.c_[aff(c),d,np.sum(c*d,axis=1)]),('quadcd',quad(np.c_[c,d])),('quadxc',quad(np.c_[x,c])),('quadxd',quad(np.c_[x,d])),('quadx',quad(x)),('crosscd',np.c_[aff(c),d,(c[:,:,None]*d[:,None,:]).reshape(len(c),16)]),('xcdb',np.c_[aff(c),d,x,c*d,b*d,b*c])]:print(name,score(f))
