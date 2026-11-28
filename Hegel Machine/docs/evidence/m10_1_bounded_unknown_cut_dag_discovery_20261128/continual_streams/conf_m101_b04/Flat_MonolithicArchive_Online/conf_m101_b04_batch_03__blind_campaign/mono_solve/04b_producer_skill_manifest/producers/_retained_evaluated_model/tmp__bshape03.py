from operators03 import *
from scipy.optimize import minimize_scalar
u,v=(S['panel_b'][k] for k in ('u','v'))
funcs={'sin':lambda u,a:np.sin(a*u),'tanh':lambda u,a:np.tanh(a*u),'rational':lambda u,a:u/(1+a*u*u),'asinh':lambda u,a:np.arcsinh(a*u),'atan':lambda u,a:np.arctan(a*u),'sinlin':lambda u,a:u+a*np.sin(u),'tanhlin':lambda u,a:u+a*np.tanh(u),'rational2':lambda u,a:u/np.sqrt(1+a*u*u)}
for name,fn in funcs.items():
 for lin in (False,True):
  def des(u,a):return np.c_[aff(u) if lin else np.ones(len(u)),fn(u,a)]
  def fitshape(u,v):
   def loss(a):
    q=des(u,a);e=q@fit(q,v)-v;return (e**2).mean()
   sc=minimize_scalar(loss,bounds=(.04,4),method='bounded');q=des(u,sc.x);return sc.x,fit(q,v),sc.fun**.5
  a,w,e=fitshape(u,v); loo=[]
  for i in range(len(u)):
   k=np.arange(len(u))!=i; aa,ww,_=fitshape(u[k],v[k]);loo.append((des(u[i:i+1],aa)@ww-v[i]).ravel())
  print(name,lin,round(a,5),round(e,6),round(np.mean(np.square(loo))**.5,6),flush=True)
# A shared coordinate law, fit as a linear combination of fixed odd functions.
for extra in [lambda u:u**3,lambda u:np.sin(2*u),lambda u:np.tanh(2*u),lambda u:u/(1+u*u),lambda u:np.sin(2.8*u),lambda u:np.tanh(1.4*u)]:
 q=np.c_[aff(u),extra(u)];w=fit(q,v);print('coefficient proportionality',np.round(np.sum(w[1:5]*w[5:],axis=1)/np.sum(w[1:5]**2,axis=1),4),np.round(np.linalg.norm(w[5:]-w[1:5]*(np.sum(w[1:5]*w[5:],axis=1)/np.sum(w[1:5]**2,axis=1))[:,None],axis=1),5))
x=S['train']['x'];y=S['train']['y'];tr=S['train']['traj'];t=S['eval_times'];z=panels()[1](x);p=quad(z);p0=aff(z); rng=np.random.default_rng(43); splits=[np.arange(72),*[rng.permutation(72) for _ in range(10)]];scores=[]
for typ,pp in [('aff',p0),('quad',p),('poly3',np.c_[p,*[np.prod(z[:,j],axis=1) for j in itertools.combinations_with_replacement(range(4),3)]])]:
 for l,alpha in itertools.product([.035,.075,.15,.3,.6,1.2],[.00002,.0003,.002,.015]):
  rec=[]
  for ix in splits:
   a,b=ix[:52],ix[52:];Y,B=relax(tr[a],t);target=np.c_[y[a],Y];pred=gp(x[a],x[b],pp[a],pp[b],target,alpha,l);rec.append([abs(pred[:,0]-y[b]).mean(),abs(pred[:,1:]@B.T-tr[b]).mean()])
  scores.append((typ,l,alpha,np.array(rec)))
for j in (0,1):
 for typ,l,alpha,e in sorted(scores,key=lambda s:s[3][:,j].mean())[:10]: print(j,typ,l,alpha,np.round([e[0,j],e[:,j].mean(),e[:,j].max()],5))
