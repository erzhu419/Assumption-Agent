import pickle,numpy as np
from scipy.optimize import least_squares
D=lambda x:np.c_[np.ones(len(x)),x]
L=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; a=s['panel_a']; M=D(a['u']); p=L(M,a['v']); q=np.argsort(np.linalg.norm(M@p-a['v'],axis=1))[:-3]; p=L(M[q],a['v'][q]); z=np.linalg.solve(p[1:].T,(s['train']['x']-p[0]).T).T
forms={'tanh':np.tanh,'asinh':np.arcsinh,'atan':np.arctan,'sin':np.sin,'sqrt':lambda x:np.sqrt(1+x*x)-1,'logcosh':lambda x:np.log(np.cosh(x)),'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'expodd':lambda x:x*(-np.expm1(-x*x)),'logodd':lambda x:x*np.log1p(x*x),'abssat':lambda x:x/(1+abs(x))}
P={}; W={}
for panel,key in [('b','v'),('c','v1'),('c','v2')]:
 u=s['panel_'+panel]['u']; v=s['panel_'+panel][key]; results=[]
 for name,f in forms.items():
  for k,m in [(1,0),(1,1),(1,2),(4,0),(4,1),(4,2),(4,4)]:
   def warp(x,q):return f(x*q[:k]);
   def feat(x,q):
    w=warp(x,q); return D(w if m==0 else np.c_[x,w] if m==2 else x+q[k:]*w)
   nb=4 if m==4 else int(m==1)
   for b0 in ([-1.,1.] if nb else [0.]):
    r=least_squares(lambda q: (feat(u,q)@L(feat(u,q),v)-v).ravel(),[1.3]*k+[b0]*nb,bounds=([.1]*k+[-20.]*nb,[6.]*k+[20.]*nb),max_nfev=110); X=feat(u,r.x); h=(np.linalg.svd(X,full_matrices=False)[0]**2).sum(1); e=np.sqrt(np.mean((r.fun.reshape(v.shape)/(1-h[:,None]))**2)); results.append((e,name,k,m,r.x,feat(z,r.x)@L(X,v),warp(z,r.x)))
 results.sort(key=lambda x:x[0]); print(panel,key,[(round(e,8),f,k,m,q.round(4)) for e,f,k,m,q,*_ in results[:5]]); P[key]=results[0][-2]; W[key]=results[0][-1]
t=s['eval_times']; H=np.c_[-np.expm1(-.41*t),-np.expm1(-1.81*t)]; y=np.c_[s['train']['y'],L(H,s['train']['traj'].T).T]; n=len(y); folds=[np.arange(n)[-20:],*np.array_split(np.random.default_rng(81).permutation(n),4)]
print('rates, times, amplitude means and scales',t,y.mean(0),y.std(0));
F={'z':z,'z+B+C':np.c_[z,*P.values()],'warp':np.c_[z,*W.values()],'z+generic':np.c_[z,z*z,np.tanh(1.5*z),np.sin(3*z),np.cos(3*z)]}; F.update(P)
for name,x in F.items():
 for typ in ('linear','quadratic','rbf'):
  errs=[]
  for va in folds:
   tr=np.setdiff1d(np.arange(n),va); xx=x/(x[tr].std(0)+1e-8); X=D(xx if typ!='quadratic' else np.c_[xx,*[xx[:,i]*xx[:,j] for i in range(x.shape[1]) for j in range(i,x.shape[1])]]);
   if typ=='rbf': X=np.c_[X,np.exp(-np.mean((xx[:,None]-xx[None,tr])**2,axis=2))]
   w=np.linalg.solve(X[tr].T@X[tr]+.02*np.eye(X.shape[1]),X[tr].T@y[tr]); e=X[va]@w-y[va]; errs.append([np.abs(e[:,0]).mean(),np.abs(e[:,1:]@H.T).mean()]);
  print(name,typ,np.array(errs)[0].round(4),np.max(errs,axis=0).round(4))
np.savez('/tmp/cal_features.npz',z=z,y=y,H=H,**{f'P{k}':v for k,v in P.items()},**{f'W{k}':v for k,v in W.items()})
