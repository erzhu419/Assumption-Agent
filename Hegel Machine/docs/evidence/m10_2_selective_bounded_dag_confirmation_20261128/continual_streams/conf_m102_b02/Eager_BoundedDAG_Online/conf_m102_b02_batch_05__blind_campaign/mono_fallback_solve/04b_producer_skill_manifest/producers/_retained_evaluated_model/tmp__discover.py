from probe import *
from scipy.optimize import minimize_scalar
from scipy.spatial.distance import cdist

def AF(u): return D(np.c_[u,u*np.roll(u,-1,axis=1),u**3])
def coords(s,x):
 p=s['panel_a']; u,v=p['u'],p['v']; F=AF(u); _,_,e=cal(s,x); w=fit(F[e<2],v[e<2]); z=np.linalg.solve(w[1:5].T,(x-w[0]).T).T
 for j in range(12):
  J=np.broadcast_to(w[1:5],(len(z),4,4)).copy(); K=w[5:9]; rr=np.roll(z,-1,axis=1)
  J+=rr[:,:,None]*K[None,:,:]+np.roll(z[:,:,None]*K[None,:,:],1,axis=1)+3*z[:,:,None]**2*w[None,9:13,:]
  dz=np.linalg.solve(J.transpose(0,2,1),(AF(z)@w-x)[...,None])[...,0];z-=np.clip(dz,-.4,.4)
 return z

def panel(p,key,funcs):
 u,y=p['u'],p[key]; out=[]
 for label,f,lo,hi in funcs:
  def score(k):
   G=f(k,u);W=np.linalg.pinv(G);r=G@W@y-y;h=np.sum(G*W.T,1);return np.mean(abs(r/(1-h[:,None])))
  o=min([minimize_scalar(score,bounds=(a,b),method='bounded') for a,b in zip(np.linspace(lo,hi,9)[:-1],np.linspace(lo,hi,9)[1:])],key=lambda o:o.fun); out.append((o.fun,label,o.x,f))
 return sorted(out,key=lambda a:a[0])

def bases(s,z,verbose=False):
 bfun=[]
 for label,f in [('sin',np.sin),('tanh',np.tanh),('erf',erf),('asinh',np.arcsinh),('atan',np.arctan),('rat',lambda x:x/(1+x*x))]:
  for i,extras in enumerate([lambda u:u,lambda u:np.c_[u,u*np.roll(u,-1,axis=1)],lambda u:np.c_[u,u*u]]): bfun.append((label+str(i),lambda q,u,f=f,ex=extras:D(np.c_[ex(u),f(q*u)]),.15,5))
 cfun=[]
 for label,f in [('cos',lambda k,u:np.cos(k*u)),('rat',lambda k,u:u**2/(1+k*u*u)),('log',lambda k,u:np.log1p(k*u*u)),('exp',lambda k,u:np.exp(-k*u*u)),('ratwhole',lambda k,u:(u+.45*u*u)/(1+k*u*u)),('ratcub',lambda k,u:u**3/(1+k*u*u))]:
  for n in [1,2]: cfun.append((label+str(n),lambda q,u,f=f,n=n:D(np.c_[u,f(q,u)]) if n==1 else D(np.c_[u,u*u,f(q,u)]),.01,5))
 B,C=s['panel_b'],s['panel_c']; result={'z':z};
 for name,p,key,fn in [('b',B,'v',bfun),('c2',C,'v2',cfun)]:
  ans=panel(p,key,fn)
  if verbose: print(name,[(round(r,7),l,round(q,5)) for r,l,q,f in ans]); print('best coefficients',np.round(fit(ans[0][3](ans[0][2],p['u']),p[key]),4))
  for k,(sc,lab,q,f) in enumerate(ans[:3]):result[name+str(k)]=f(q,z)@fit(f(q,p['u']),p[key])
 f=lambda a:D(a*np.roll(a,-1,axis=1));result['c1']=f(z)@fit(f(C['u']),C['v1']);return result

if __name__=='__main__':
 tr=S['train']; X=np.r_[tr['x'],S['test']['x']]; z=coords(S,X); V=bases(S,z,True); print('z range',z.min(0),z.max(0)); T=np.c_[tr['y'],tr['traj']];
 def e(p,y): return np.mean(abs(p-y),0)[0],np.mean(abs(p[:,1:]-y[:,1:])),np.mean(abs(p[:,-1]-y[:,-1]))
 def kn(F,g): return (F@F.T)/F.shape[1] if g==0 else np.exp(-g*cdist(F,F,'sqeuclidean')/F.shape[1]);
 for label,ks in [('z',['z']),('b',['b0']),('c2',['c20']),('c1',['c1']),('zc1',['z','c1']),('zc2',['z','c20']),('zb',['z','b0']),('all',['z','b0','c1','c20'])]:
  x=np.column_stack([V[k] for k in ks]); x=(x-x[:52].mean(0))/x[:52].std(0); ans=[]
  for transform in [lambda x:x,lambda x:np.tanh(x),lambda x:np.arcsinh(2*x)]:
   x1=transform(x); L=1+kn(x1,0)
   for name,K in [('linear',L),('quad',L**2),('cube',L**3)]+[(str(g),L**2+kn(x1,g)) for g in [.04,.15,.5,1.5]]:
    for reg in [1e-7,1e-4,.003,.03,.2]:
     pred=K[52:72,:52]@np.linalg.solve(K[:52,:52]+reg*np.eye(52),T[:52]); score=e(pred,T[52:]);ans.append((score,name,reg,transform))
  print(label,'y best',[(a[0],a[1],a[2]) for a in sorted(ans,key=lambda a:a[0][0])[:3]],'traj best',[(a[0],a[1],a[2]) for a in sorted(ans,key=lambda a:a[0][1])[:3]])
