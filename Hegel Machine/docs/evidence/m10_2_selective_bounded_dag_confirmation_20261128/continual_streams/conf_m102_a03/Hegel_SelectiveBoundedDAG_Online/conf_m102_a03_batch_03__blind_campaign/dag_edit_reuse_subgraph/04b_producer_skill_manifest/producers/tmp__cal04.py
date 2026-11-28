import numpy as np
from scipy.optimize import least_squares

def aff(x): return np.c_[np.ones(len(x)), x]
def fit(a,b): return np.linalg.lstsq(a,b,rcond=None)[0]
def cal(s, verbose=False):
 p=s['panel_a']; u,v=p['u'],p['v']; keep=np.linalg.norm(v-u,axis=1)<2; u=u[keep];v=v[keep]
 def evaluate(w, mask):
  z=np.empty_like(w)
  for j in range(w.shape[1]):
   x,y=u[mask,j],v[mask,j]; t=w[:,j]; best=[]
   for kind in ('linear','cubic','tanh','asinh','sin','rational','sqrt'):
    if kind=='linear':
     b,a=fit(aff(x[:,None]),y); pred=b+a*x; inv=(t-b)/a
    elif kind=='cubic':
     b,a,c=fit(np.c_[np.ones(len(x)),x,x**3],y);pred=b+a*x+c*x**3; inv=(t-b)/a
     for k in range(10): inv-=(b+a*inv+c*inv**3-t)/(a+3*c*inv**2)
    else:
     fs={'tanh':(np.tanh,np.arctanh,.999999), 'asinh':(np.arcsinh,np.sinh,np.inf),'sin':(np.sin,np.arcsin,.999999),'rational':(lambda x:x/(1+abs(x)),lambda x:x/(1-abs(x)),.999999),'sqrt':(lambda x:x/np.sqrt(1+x*x),lambda x:x/np.sqrt(1-x*x),.999999)}
     f,g,clip=fs[kind]; c=least_squares(lambda c:c[0]+c[1]*f(c[2]*x)-y,[np.mean(y-x),1.,1.],bounds=([-2,.01,.001],[2,100,1.2])).x;pred=c[0]+c[1]*f(c[2]*x);inv=g(np.clip((t-c[0])/c[1],-clip,clip))/c[2]
    best.append((np.mean((pred-y)**2),kind,inv))
   best.sort(key=lambda a:a[0]);z[:,j]=best[0][2]
   if verbose: print(j,[(k,round(e**.5,7)) for e,k,_ in best])
  return z
 return evaluate(np.r_[s['train']['x'],s['test']['x']],np.ones(len(u),bool))
if __name__=='__main__':
 import pickle
 s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=cal(s,True)[:len(s['train']['x'])]; a=aff(x); i,j=np.triu_indices(x.shape[1]); q=np.c_[a,x[:,i]*x[:,j]]; y=s['train']['y']; t=s['train']['traj'];
 for z in (a,q):
  print('errors train, test',[(abs(z[:52]@fit(z[:52],d[:52])-d[:52]).mean(),abs(z[52:]@fit(z[:52],d[:52])-d[52:]).mean()) for d in (y,t)]);print('coeff y',fit(z,y));print('traj sv',np.linalg.svd(t-z@fit(z,t),compute_uv=False))
 for panel in ('panel_b','panel_c'):
  p=s[panel];u=p['u']; q=np.c_[aff(u),u[:,i]*u[:,j]]
  for k,v in p.items():
   if k=='u':continue
   print(panel,k,'quadratic coef',np.round(fit(q,v).T,3))
