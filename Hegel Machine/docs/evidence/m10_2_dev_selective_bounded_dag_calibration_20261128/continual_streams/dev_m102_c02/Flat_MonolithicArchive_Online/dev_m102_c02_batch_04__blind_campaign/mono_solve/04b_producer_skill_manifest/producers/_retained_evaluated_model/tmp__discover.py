from probe import *
from scipy.optimize import least_squares

def calibrate():
 u,v=s['panel_a']['u'],s['panel_a']['v']; r=np.roll(u,-1,axis=1); d=u.shape[1]; j=np.arange(d)
 terms=[u,r,u*r,u**3,r**3,u*u*r,u*r*r,u*u,r*r,np.roll(u,-2,axis=1),np.roll(u,1,axis=1)]
 H=np.zeros((len(u),d,d*2+len(terms)-1)); H[:,j,j]=1;H[:,j,d+j]=u;H[:,:,2*d:]=np.moveaxis(np.array(terms[1:]),0,-1)
 keep=np.ones(len(u),bool)
 for _ in range(8):
  w=L(H[keep].reshape(-1,H.shape[-1]),v[keep].ravel(),rcond=None)[0];e=np.linalg.norm(H@w-v,axis=1); keep=e<5*np.median(e)
 print('calibration coefficients',w,'inlier quantiles',np.quantile(e[keep],[0,.5,1]))
 def cal(z):
  r=np.roll(z,-1); ts=[r,z*r,z**3,r**3,z*z*r,z*r*r,z*z,r*r,np.roll(z,-2),np.roll(z,1)]
  return w[:d]+w[d:2*d]*z+w[2*d:]@np.array(ts)
 return lambda xs:np.array([least_squares(lambda z:cal(z)-x,(x-w[:d])/w[d:2*d],gtol=1e-11).x for x in xs])

def panel(u,v):
 def loo(H,Y):
  g=np.linalg.pinv(H); E=(H@(g@Y)-Y)/(1-np.einsum('ij,ji->i',H,g))[:,None];return np.sqrt(np.mean(E**2))
 m=mon(u,2); base=aff(u); r=np.roll(u,-1,axis=1)
 D={'linear':base,'sq':np.c_[base,u*u],'cyclicp':np.c_[base,u*r],'quadratic':m,'cubic':np.c_[base,u**3],'tanh(x)':aff(np.tanh(u)),'sin(x)':aff(np.sin(u)),'quad+cub':np.c_[m,u**3]}
 for name,h in D.items(): print(name,round(loo(h,v),6),end='; ')
 print()
 for name in ('sine','tanh','powered','rational'):
  for hname in ('linear','sq','cyclicp','quadratic'):
   H=D[hname];
   def model(p):
    W=p[2:].reshape(H.shape[1],-1); g=H@W; a=np.exp(p[0]);b=p[1]
    return b+a*({'sine':np.sin,'tanh':np.tanh,'powered':lambda z:z+0.25*z**3,'rational':lambda z:z/(1+z*z)}[name])(g)
   init=np.r_[np.log(1.5),0,L(H,v/1.5,rcond=None)[0].ravel()];q=least_squares(lambda p:(model(p)-v).ravel(),init,max_nfev=100);print(name,hname,round(np.sqrt(np.mean((model(q.x)-v)**2)),5),np.exp(q.x[0]))
 print('quadratic coefficients',L(m,v,rcond=None)[0]);print('linear row errors',np.linalg.norm(base@L(base,v,rcond=None)[0]-v,axis=1))

def cv_baselines(z):
 from scipy.spatial.distance import cdist
 x=s['train']['x']; y=s['train']['y']; t=s['train']['traj']; V=np.c_[y,t]
 def cv(h,lam=0):
  B=h[:52]; pred=h[52:]@L(np.r_[B,np.sqrt(lam)*np.eye(B.shape[1])],np.r_[V[:52],np.zeros((B.shape[1],V.shape[1]))],rcond=None)[0];return np.mean(np.abs(pred-V[52:]),axis=0)
 for n,f in [('raw',x),('cal',z),('tanhcal',np.tanh(z))]:
  for d in [1,2,3]:print(n,d,[np.round(cv(mon(f,d),lam)[[0,1,-1]],5).tolist() for lam in [0,.001,.03,.3]])
 for f in (x,z):
  for scale in (.2,.5,1,2):
   K=np.exp(-scale*cdist(f,f,'sqeuclidean'))
   for nug in (.001,.02,.1):
    pred=K[52:,:52]@np.linalg.solve(K[:52,:52]+nug*np.eye(52),V[:52]);print('kernel',scale,nug,np.mean(abs(pred-V[52:]),0)[[0,1,-1]])
if __name__=='__main__':
 z=calibrate()(s['train']['x']);cv_baselines(z)
 for k,j in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:print(k,j);panel(s[k]['u'],s[k][j])
