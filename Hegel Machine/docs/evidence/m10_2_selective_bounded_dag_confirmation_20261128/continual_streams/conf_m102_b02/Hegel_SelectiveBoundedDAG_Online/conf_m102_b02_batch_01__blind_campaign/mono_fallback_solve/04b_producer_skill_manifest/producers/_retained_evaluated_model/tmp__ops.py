from explore import *
from scipy.optimize import minimize_scalar
F={'asinh':np.arcsinh,'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'erf':__import__('scipy.special',fromlist=['erf']).erf,'rootodd':lambda a:a/np.sqrt(1+a*a),'ratodd':lambda a:a/(1+a*a),'log2':lambda a:np.log1p(a*a),'root2':lambda a:np.sqrt(1+a*a)-1,'rat2':lambda a:a*a/(1+a*a),'cos':lambda a:1-np.cos(a),'exp2':lambda a:-np.expm1(-a*a)}
def design(u,name,k,lin):
 return np.column_stack([np.ones(len(u)),*([u] if lin else []),F[name](u*k)])
def fit(u,v,name,lin,flex=False):
 def resid(k):
  d=design(u,name,np.exp(k),lin);return (d@np.linalg.lstsq(d,v,rcond=None)[0]-v).ravel()
 ks=np.linspace(-2,2.2,91); jj=np.argmin([np.mean(resid(k)**2) for k in ks]); lo=ks[max(jj-1,0)];hi=ks[min(jj+1,len(ks)-1)];k=minimize_scalar(lambda k:np.mean(resid(k)**2),bounds=(lo,hi),method='bounded').x
 if flex:k=least_squares(resid,np.full(4,k),bounds=(-4,2.5),max_nfev=100).x
 d=design(u,name,np.exp(k),lin); b=np.linalg.lstsq(d,v,rcond=None)[0];e=(d@b-v)/(1-(d*np.linalg.pinv(d).T).sum(1))[:,None];return np.sqrt(np.mean(e**2)),np.exp(k),b
if __name__=='__main__':
 for p,key in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
  u,v=s[p]['u'],s[p][key]; r=sorted((fit(u,v,n,l)[0],n,l) for n in F for l in (False,True));print(p,key)
  for e,n,l in r[:7]:
   err,k,b=fit(u,v,n,l); ef,kf,bf=fit(u,v,n,l,True);print(n,l,e,k,ef,kf,'b',b.round(3).tolist())
 t=s['eval_times'];T=Y[:,1:];V=np.linalg.svd(T,full_matrices=False)[2][:2];print('V',V)
 funcs={'one':np.ones_like(t),'t':t,'sqrt':np.sqrt(t),'log':np.log1p(t),'frac':t/(1+t),'exp':-np.expm1(-t),'atan':np.arctan(t),'t2':t*t}
 for k,v in funcs.items():print(k,np.linalg.norm(v-(v@V.T)@V)/np.linalg.norm(v))
 for name,f in [('log',lambda k:np.log1p(k*t)),('exp',lambda k:-np.expm1(-k*t)),('frac',lambda k:t/(1+k*t)),('atan',lambda k:np.arctan(k*t)),('pow',lambda k:t**k)]:
  def err(k):
   v=f(k);return np.linalg.norm(v-(v@V.T)@V)/np.linalg.norm(v)
  grid=np.geomspace(.01,12,170);best=min(range(len(grid)),key=lambda i:err(grid[i]));res=minimize_scalar(err,bounds=(grid[max(0,best-1)],grid[min(len(grid)-1,best+1)]),method='bounded');print(name,res.x,res.fun)
