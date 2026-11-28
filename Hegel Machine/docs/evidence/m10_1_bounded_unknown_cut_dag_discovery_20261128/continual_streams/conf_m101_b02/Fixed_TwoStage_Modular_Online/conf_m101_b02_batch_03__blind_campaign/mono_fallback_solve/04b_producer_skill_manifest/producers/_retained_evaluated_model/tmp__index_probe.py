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
funcs={'tanh':(np.tanh,lambda q:1-np.tanh(q)**2),'sin':(np.sin,np.cos),'atan':(np.arctan,lambda q:1/(1+q*q)),'softsign':(lambda q:q/np.sqrt(1+q*q),lambda q:(1+q*q)**-1.5),'oddexp':(lambda q:-np.sign(q)*np.expm1(-abs(q)),lambda q:np.exp(-abs(q)))}
for target in range(3):
 results=[]; yt=Y[:,target];
 for nm,z in cfg.items():
  X=D((z-z[:N].mean(0))/np.maximum(z[:N].std(0),1e-7)); tr=X[:N]; p=X.shape[1]; w0=ls(tr,yt[:N]);
  for link,(f,df) in funcs.items():
   for baseline in [False,True] if p==5 else [False]:
    l=p if baseline else 1; L=tr if baseline else tr[:,:1]; LE=X if baseline else X[:,:1]
    def pred(w,E,X):return E@w[:l]+w[l]*f(X@w[l+1:]);
    def fun(w):return np.r_[pred(w,L,tr)-yt[:N],.005*w[1:]]
    def jac(w):
     g=tr@w[l+1:];return np.r_[np.c_[L,f(g),w[l]*df(g)[:,None]*tr],.005*np.eye(len(w))[1:]]
    best=None
    for a in ([.35,.85,1.5] if p==5 else [.5,1.1]):
     initial=np.r_[np.zeros(l),1.4,a*w0];initial[0]=yt[:N].mean();opt=least_squares(fun,initial,jac=jac,max_nfev=105);
     if best is None or opt.cost<best.cost:best=opt
    w=best.x; r=abs(pred(w,LE,X)-yt);results.append((r[:N].mean(),r[N:].mean(),nm,link,baseline,w))
 print('TARGET',target,'scale',scale[target]);print('best train');
 for r in sorted(results,key=lambda r:r[0])[:10]:print(r)
 print('best check');
 for r in sorted(results,key=lambda r:r[1])[:8]:print(r)
