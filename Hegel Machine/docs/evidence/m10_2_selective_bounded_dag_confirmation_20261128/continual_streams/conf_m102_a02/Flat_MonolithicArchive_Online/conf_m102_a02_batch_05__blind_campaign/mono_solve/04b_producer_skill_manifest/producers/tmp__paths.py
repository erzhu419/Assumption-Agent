import pickle,numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=S['train']; ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
def af(x):return np.c_[np.ones(len(x)),x]
def quad(x,cross=False):
 i,j=np.triu_indices(x.shape[1],int(cross));return np.c_[af(x),x[:,i]*x[:,j]]
def cal(x):
 u,v=[S['panel_a'][k] for k in ('u','v')];cs=[]
 for j in range(4):
  F=af(u[:,j,None]);r=least_squares(lambda c:F@c-v[:,j],[0,1],loss='soft_l1',f_scale=.0002).fun;keep=abs(r)<.001;cs.append(ls(F[keep],v[keep,j]))
 a,b=np.array(cs).T;return (x-a)/b
nl={'gauss':lambda x,p:x*np.exp(-p*x*x),'rational':lambda x,p:x/(1+p*x*x),'root':lambda x,p:x/np.sqrt(1+p*x*x),'tanh':lambda x,p:np.tanh(p*x),'sin':lambda x,p:np.sin(p*x),'asinh':lambda x,p:np.arcsinh(p*x)}
def op(u,v,kind,full=True,verbose=False):
 def F(x,p):return np.c_[af(x) if full else np.ones(len(x)),nl[kind](x,p)]
 def get(p):
  f=F(u,p); c=ls(f,v);return f,c
 def fun(p):
  f,c=get(p);return (f@c-v).ravel()
 p=least_squares(fun,[1.],bounds=(.05,4),gtol=1e-12).x;f,c=get(p);Q=np.linalg.qr(f)[0];err=(f@c-v)/(1-(Q*Q).sum(1))[:,None]
 if verbose: print(kind,full,np.round(p,5),round(np.sqrt(np.mean(err**2)),7),round(np.sqrt(np.mean(fun(p)**2)),7))
 return lambda x:F(x,p)@c, np.sqrt(np.mean(err**2))
def operators(verbose=False):
 out={}
 for key,pk in [('B','v'),('C','v1'),('D','v2')]:
  pn='panel_b' if key=='B' else 'panel_c';u,v=S[pn]['u'],S[pn][pk];opts=[]
  if verbose:print('OP',key)
  for k in nl:
   for f in [True,False]:opts.append(op(u,v,k,f,verbose))
  for x in [u]+([S[pn]['v1']] if key=='D' else []):
   for k in [af,quad,lambda x:quad(x,True)]:
    F=k(x);c=ls(F,v);Q=np.linalg.qr(F)[0];e=np.sqrt(np.mean(((F@c-v)/(1-(Q*Q).sum(1))[:,None])**2));
    if verbose: print('poly',F.shape[1],e,'train',np.sqrt(np.mean((F@c-v)**2)))
    if x is u: opts.append((lambda z, k=k,c=c:k(z)@c,e))
  out[key]=min(opts,key=lambda t:t[1])[0]
 return out
if __name__=='__main__':
 o=operators(True); z=cal(T['x']); d={'Z':z};d.update({k:f(z) for k,f in o.items()});d.update({k+j:f(d[j]) for k,f in o.items() for j in o});d.update({j+k:np.c_[d[j],d[k]] for i,j in enumerate('ZBCD') for k in 'ZBCD'[i+1:] if j+k not in d}); d.update({'Z+'+j:np.c_[z,d[j]] for j in o});d['C+D']=np.c_[d['C'],d['D']];
 rng=np.random.default_rng(562);splits=[np.arange(72)]+[rng.permutation(72) for _ in range(10)];
 for k,g in d.items():
  def ev(F,y,lam=0):
   def pred(p):
    a,b=p[:52],p[52:];A=F[a];w=ls(A,y[a]) if not lam else np.linalg.solve(A.T@A+lam*np.diag(np.r_[0,np.ones(A.shape[1]-1)]),A.T@y[a]);return abs(F[b]@w-y[b]).mean()
   E=[pred(p) for p in splits];return np.round([E[0],np.mean(E[1:]),max(E[1:])],5)
  print(k,'states',ev(af(g),T['traj']),ev(quad(g),T['traj'],.01),'y',ev(af(g),T['y']),ev(quad(g,True),T['y'],.002),ev(quad(g),T['y'],.002))
