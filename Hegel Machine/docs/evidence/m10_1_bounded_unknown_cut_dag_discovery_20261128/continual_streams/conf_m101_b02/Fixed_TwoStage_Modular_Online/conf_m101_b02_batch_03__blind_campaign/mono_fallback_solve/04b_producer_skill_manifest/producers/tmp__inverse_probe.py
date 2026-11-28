import pickle,numpy as np
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations
from sklearn.linear_model import Lasso
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; t=s['eval_times']; T=s['train']['traj'];y=s['train']['y']; N=52
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
D=lambda *a:np.column_stack([np.ones(len(a[0])),*a])
U=-np.expm1(-t[:,None]*np.array([.41837,1.81795])); H=ls(U,T.T).T; Y=np.c_[y,H]; scale=Y[:N].std(0); Y=Y/scale
p=s['panel_c'];u=p['u'];
funcs={'oddexp':lambda x,k:-np.sign(x)*np.expm1(-k*abs(x)),'tanh':lambda x,k:np.tanh(k*x),'sin':lambda x,k:np.sin(k*x),'atan':lambda x,k:np.arctan(k*x),'even':lambda x,k:x*x/np.sqrt(1+k*x*x),'cos':lambda x,k:np.cos(k*x),'rat':lambda x,k:x*x/(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'gauss':lambda x,k:np.exp(-k*x*x)}
ops=[]
for name in ['v1','v2']:
 v=p[name]; cand=[]
 for n,f in funcs.items():
  for lin in [False,True]:
   df=lambda x,k:D(x,f(x,k)) if lin else D(f(x,k)); loss=lambda k:np.linalg.norm(df(u,k)@ls(df(u,k),v)-v); o=minimize_scalar(loss,bounds=(.02,4),method='bounded'); cand.append((o.fun,n,lin,o.x))
 cand.sort(); print(name,cand[:6]);_,n,lin,k=cand[0];f=funcs[n]; df=lambda x,k,f=f,lin=lin:D(x,f(x,k)) if lin else D(f(x,k));w=ls(df(u,k),v);ops.append(lambda x,df=df,k=k,w=w:df(x,k)@w)
b=s['panel_b'];B=ls(D(b['u']),b['v']); blocks={'x':x,'b':D(x)@B,'bi':(x-B[0])@np.linalg.inv(B[1:])};
pa=s['panel_a']; a=pa['u']; v=pa['v']; A=lambda a:D(a,a*a,a**3,np.array([a[:,i]*a[:,j] for i,j in combinations(range(4),2)]).T);F=A(a); W=ls(F,v)
for _ in range(70):
 q=np.maximum(np.linalg.norm(v-F@W,axis=1),.00025)**-.5;W=ls(F*q[:,None],v*q[:,None]);
blocks['a']=A(x)@W
from scipy.optimize import root
from itertools import product
np.set_printoptions(precision=3,suppress=True)
print('Y quantiles',np.quantile(Y,[0,.05,.25,.5,.75,.95,1],axis=0),'Y correlation',np.corrcoef(Y.T)); print('calibration constant/linear',W[:5]);print('x covariance',np.cov(x.T))
ai=np.array([root(lambda u:(A(u[None])@W-r).ravel(),r).x for r in x]); print('inverse residual',np.max(abs(A(ai)@W-x)), 'range',np.quantile(ai,[0,.5,1],axis=0));
cfg={'x':x,'b':blocks['b'],'bi':blocks['bi'],'a':blocks['a'],'ai':ai}
cfg["bai"]=D(ai)@B;cfg["biai"]=(ai-B[0])@np.linalg.inv(B[1:]);
for name,z in list(cfg.items()):
 c,d=(op(z) for op in ops);cfg['c'+name]=c;cfg['d'+name]=d;cfg['m'+name]=np.c_[z,c,d];cfg['cd'+name]=np.c_[c,d]
from sklearn.preprocessing import PolynomialFeatures
from scipy.spatial.distance import cdist
from scipy.linalg import solve
P=lambda z,q:PolynomialFeatures(q).fit_transform(z)
Z=np.c_[y,T]
folds=[(np.arange(52),np.arange(52,72))]+[(p[:56],p[56:]) for p in [np.random.default_rng(i).permutation(72) for i in range(4)]]
print('trajectory singular spectrum',np.linalg.svd(T,compute_uv=False));print('y vs time',ls(D(T),y),np.mean(abs(y-D(T)@ls(D(T),y))))
for k in [2,3,4]:
 def time(p):return -np.expm1(-t[:,None]*p)
 def obj(p):
  F=time(p);return (F@ls(F,T[:52].T)-T[:52].T).ravel()
 o=least_squares(obj,np.geomspace(.2,3,k),bounds=(.001,20));F=time(o.x); print('time',o.x,np.std(o.fun),np.mean(abs(F@ls(F,T[52:].T)-T[52:].T)))
 def scores(G=None,K=None,reg=.001):
  vals=[]
  for a,b in folds:
   if K is not None:r=K[np.ix_(b,a)]@solve(K[np.ix_(a,a)]+reg*np.eye(len(a)), Z[a]-Z[a].mean(0),assume_a='pos')+Z[a].mean(0)
   else:
    g=G[a];l=reg*np.eye(g.shape[1]);l[0,0]=0;r=G[b]@solve(g.T@g+l,g.T@Z[a],assume_a='pos')
   e=abs(r-Z[b]);vals.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean()])
  return np.array(vals)
res=[]
for name in ['x','ai','bai','biai','cbai','dbai','mbai','cdbai','mai','cdai','mb','mbiai']:
 z=cfg[name]; z=z/np.maximum(z.std(0),.001); d=cdist(z,z)/np.sqrt(z.shape[1]);
 for order in [1,2,3,4]:
  f=P(z,order)
  for reg in [.0003,.01,.2,1.]:
   if f.shape[1]<=110:e=scores(G=f,reg=reg);res.append((name,'poly'+str(order),reg,e))
   else:e=scores(K=f@f.T,reg=reg);res.append((name,'poly'+str(order),reg,e))
 for h in [.35,.6,1,1.7,2.8,5]:
  w=d/h
  for tag,K in [('rbf',np.exp(-w*w/2)),('m32',(1+w)*np.exp(-w)),('m52',(1+w+w*w/3)*np.exp(-w))]:
   for reg in [.00003,.002,.04]:res.append((name,tag+str(h),reg,scores(K=K,reg=reg)))
for out in range(3):
 print('OUTPUT',out)
 for n,m,r,e in sorted(res,key=lambda v:.4*v[3][0,out]+.4*v[3][1:,out].mean()+.2*v[3][:,out].max())[:12]:print(n,m,r,np.round(e[:,out],4),np.round(e.mean(0),4))
print('inverse only:')
for n,m,r,e in res:
 if n=='ai' and m.startswith('poly') and r in [.0003,.2]:print(n,m,r,np.round(e[0],4),np.round(e[1:].mean(0),4))
print('bai corr',np.round(np.corrcoef(np.c_[cfg['bai'],cfg['cbai'],cfg['dbai'],Y].T)[:12,-3:],3))
