import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement, combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=52

def lin(x):return np.c_[np.ones(len(x)),x]
def fit(d,v):return np.linalg.lstsq(d,v,rcond=None)[0]
def p1(x):return np.c_[np.ones(len(x)),x[:,0]*x[:,1],x[:,0]*x[:,3],x[:,1]*x[:,2],x[:,2]*x[:,3]]
a=s['panel_a']; b=s['panel_b']; c=s['panel_c']; u=c['u'];v=c['v2']
ac=[]
for i in range(4):
 d=lin(a['u'][:,i:i+1]); k=np.ones(len(d),bool)
 for _ in range(8):
  w=fit(d[k],a['v'][k,i]); r=abs(d@w-a['v'][:,i]); k=r<max(.0005,4*np.median(r))
 ac.append(w)
ac=np.array(ac); x=(s['train']['x']-ac[:,0])/ac[:,1]; z=lin(x)@fit(lin(b['u']),b['v']); c1=p1(z)@fit(p1(u),c['v1'])
family={
'quadratic':(lambda x,p:x+p[0]*x*x,[.45]),
'cubic':(lambda x,p:x+p[0]*x*x+p[1]*x**3,[.45,.1]),
'exp':(lambda x,p:np.expm1(p[0]*x)/p[0],[.9]),
'rat':(lambda x,p:x/(1-p[0]*x),[.4]),
'ratsq':(lambda x,p:(x+p[0]*x*x)/(1+p[1]*x*x),[.45,-.08]),
'expplus':(lambda x,p:x+p[1]*(np.exp(p[0]*x)-1-p[0]*x),[.9,1.]),
'pow':(lambda x,p:(np.maximum(.02,1+p[0]*x)**p[1]-1)/(p[0]*p[1]),[.4,3.]),
'atan':(lambda x,p:np.arctan(p[0]*x)+p[1]*x*x,[.7,.3]),
}
res={}
for k,(f,p) in family.items():
 def r(p):
  d=lin(f(u,p));return (d@fit(d,v)-v).ravel()
 q=least_squares(r,p,max_nfev=300); res[k]=(np.sqrt(np.mean(q.fun**2)),q.x,lin(f(z,q.x))@fit(lin(f(u,q.x)),v));print(k,res[k][0:2])
c2=min(res.values(),key=lambda x:x[0])[2]; cy=lin(z)@fit(lin(u),v)
T=s['train']['traj']; Y=s['train']['y']; t=s['eval_times'];
B=np.c_[-np.expm1(-1.25*t),t/(1+.22*t)]; print('tele basis rms',np.sqrt(np.mean((B@fit(B,T.T)-T.T)**2)))
B=np.c_[-np.expm1(-1.815*t),-np.expm1(-.424*t)]; S=fit(B,T.T).T; targets=np.c_[Y,S]; print('latent ranges',targets.min(0),targets.max(0))
def poly(d,degree):return np.column_stack([np.ones(len(d))]+[np.prod(d[:,i],axis=1) for j in range(1,degree+1) for i in combinations_with_replacement(range(d.shape[1]),j)])
def prod(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
paths={'x':x,'z':z,'c1':c1,'c2':c2,'c1x':p1(x)@fit(p1(u),c['v1']),'c2x':lin(family[min(res,key=lambda k:res[k][0])][0](x,min(res.values(),key=lambda x:x[0])[1]))@fit(lin(family[min(res,key=lambda k:res[k][0])][0](u,min(res.values(),key=lambda x:x[0])[1])),v)}
print('corr by path');
for k,d in paths.items():print(k,np.round(np.corrcoef(np.c_[d,targets].T)[:4,4:],3))
results=[]
for name,d0 in list(paths.items())+[(a+' '+b,np.c_[paths[a],paths[b]]) for a,b in combinations(paths,2)]:
 for deg in range(1,4 if d0.shape[1]==4 else 3):
  d=poly(d0,deg); scale=d[:n].std(0); scale[scale<1e-7]=1; d=d/scale
  for reg in [1e-7,.003,.1,1.]:
   w=np.linalg.solve(d[:n].T@d[:n]+reg*np.eye(d.shape[1]),d[:n].T@targets[:n]); p=d@w; e=abs(p[n:]-targets[n:]).mean(0); et=abs(p[n:,1:]@B.T-T[n:]).mean(); results.append((e[0],et,*e[1:],name,deg,reg))
for k in range(4):print('best',k,*sorted(results,key=lambda x:x[k])[:7],sep='\n')
np.savez('/tmp/paths.npz',**paths,T=T,Y=Y,t=t,S=S,B=B)
