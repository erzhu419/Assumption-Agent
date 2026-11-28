import pickle,itertools as it,numpy as np
from scipy.optimize import minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; Y=np.c_[s['train']['y'],s['train']['traj']]; I=list(it.combinations_with_replacement(range(4),2)); ls=lambda X,y:np.linalg.lstsq(X,y,rcond=None)[0]
def des(x,*bs):return np.column_stack((np.ones(len(x)),x,*bs))
def q(x):return np.column_stack([x[:,i]*x[:,j] for i,j in I])
def a(x):return des(x,q(x),x**3)
def fit(g,p,k):
 d=s[p];z=g(d['u']);v=d[k];ii=np.arange(len(z));
 if p=='panel_a':
  for _ in range(8):
   e=np.linalg.norm(z@ls(z[ii],v[ii])-v,axis=1);ii=np.argsort(e)[:45]
 b=ls(z[ii],v[ii]);return lambda u:g(u)@b
A=fit(a,'panel_a','v'); C=fit(lambda u:des(u,q(u)),'panel_c','v1'); D=fit(lambda u:des(u,q(u)),'panel_c','v2');
print('A RMSE sorted',np.sort(np.linalg.norm(A(s['panel_a']['u'])-s['panel_a']['v'],axis=1))[[0,40,44,45,47]])
for p,k,funcs in [('panel_b','v',{'tanh':np.tanh,'sin':np.sin,'erf':__import__('scipy').special.erf,'softsign':lambda z:z/(1+abs(z)),'asinh':np.arcsinh,'rat':lambda z:z/(1+z*z),'rootrat':lambda z:z/np.sqrt(1+z*z),'exp':lambda z:-np.sign(z)*np.expm1(-abs(z))}),('panel_c','v2',{'evenroot':lambda z:z*z/np.sqrt(1+z*z),'evenrat':lambda z:z*z/(1+z*z),'cos':lambda z:1-np.cos(z),'evenexp':lambda z:-np.expm1(-z*z)})]:
 u=s[p]['u'];v=s[p][k];ans=[]
 for name,f in funcs.items():
  for base in (False,True):
   def g(x,t):return des(x,f(t*x)) if base else np.c_[np.ones(len(x)),f(t*x)]
   def loss(t):
    z=g(u,t);return np.mean((z@ls(z,v)-v)**2)**.5
   t=minimize_scalar(loss,bounds=(.02,5),method='bounded',options={'xatol':1e-8}).x
   ans.append((loss(t),name,base,t))
 for out in sorted(ans)[:8]:print(p,out)
 err,name,base,t=min(ans);f=funcs[name];g=(lambda f,t,base:lambda x: des(x,f(t*x)) if base else np.c_[np.ones(len(x)),f(t*x)])(f,t,base)
 if p=='panel_b':B=fit(g,p,k)
 else:D=fit(g,p,k)
def score(z):
 z=z.copy(); z-=z[:52].mean(0);z/=np.maximum(z[:52].std(0),1e-8);u,sv,v=np.linalg.svd(z[:52],full_matrices=False); ans=[]
 for l in (1e-6,.001,.03,.3):
  pred=Y[:52].mean(0)+(z[52:]@v.T*(sv/(sv*sv+l)))@(u.T@(Y[:52]-Y[:52].mean(0)));e=np.abs(pred-Y[52:]);ans.append(tuple(np.round([e[:,0].mean(),e[:,1:].mean()],4)))
 return ans
for n,z in [('raw',x),('A',A(x)),('B',B(x)),('BA',B(A(x))),('AB',A(B(x)))]:
 c=C(z);d=D(z)
 for nn,Z in [('linear',z),('polynomial',np.c_[z,q(z),z**3]),('panels',np.c_[z,c,d]),('cross',np.c_[z,c,d,(z[:,:,None]*c[:,None,:]).reshape(len(z),-1),(z[:,:,None]*d[:,None,:]).reshape(len(z),-1)])]:print(n,nn,score(Z))
