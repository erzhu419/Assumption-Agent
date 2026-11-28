import numpy as np,pickle,runpy
from scipy.optimize import minimize_scalar,least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; a=s['panel_a']; u,v=a['u'],a['v']; h=np.c_[np.ones(len(u)),u]; w=np.ones(len(u))
for _ in range(15):
 b=np.linalg.lstsq(h*w[:,None],v*w[:,None],rcond=None)[0]; r=np.linalg.norm(v-h@b,axis=1); w=1/np.sqrt(1+(r/max(.0002,np.median(r)))**2)
z=(tr['x']-b[0])/np.diag(b[1:]); y=np.c_[tr['y'],tr['traj']]; n=len(z); print('n',n)
p=s['panel_c']; u=p['u']; U=np.r_[u,z]; transforms={'z':z};
for key in ['v1','v2']:
 v=p[key]; print(key,'min max',v.min(0),v.max(0));
 for name,fn in [('tanh',np.tanh),('sin',np.sin),('exp',lambda x:np.expm1(-x*x))]:
  def f(t,A):return np.c_[np.ones(len(A)),A,fn(t*A)]
  def e(t):
   h=f(t,u);return np.mean((h@np.linalg.lstsq(h,v,rcond=None)[0]-v)**2)
  o=minimize_scalar(e,bounds=(.05,2),method='bounded'); print(name,round(o.x,6),np.sqrt(o.fun)); h=f(o.x,u); transforms[key+name]=f(o.x,z)@np.linalg.lstsq(h,v,rcond=None)[0]
 for fn,inv,name in [(np.tanh,np.arctanh,'posttanh'),(np.sin,np.arcsin,'postsin')]:
  H=np.c_[np.ones(len(u)),u]; B=np.linalg.lstsq(H,v,rcond=None)[0]; out=[]; rms=[]
  for j in range(4):
   t0=np.r_[0.,1.,B[:,j]]
   o=least_squares(lambda t:t[0]+t[1]*fn(H@t[2:])-v[:,j],t0,max_nfev=350)
   out.append(o.x[0]+o.x[1]*fn(np.c_[np.ones(n),z]@o.x[2:]));rms.append(np.sqrt(np.mean(o.fun**2)))
  print(name,np.round(rms,6));transforms[key+name]=np.array(out).T
sp=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py')['run_node']; splits=[]
for p in [np.arange(n)]+[np.random.default_rng(i).permutation(n) for i in range(8)]:
 ia,it=p[:-20],p[-20:]; a=sp({'__input__':dict(s,train={'x':tr['x'][ia],'y':y[ia,0],'traj':y[ia,1:]},test={'x':tr['x'][it]})}); splits.append((ia,it,np.c_[a['y_hat'],a['state_pred']]))
ww=np.r_[.5,np.repeat(.65,y.shape[1]-1)]
def scores(K,reg):
 sc=[]
 for ia,it,b in splits:
  ym=y[ia].mean(0); pred=ym+K[np.ix_(it,ia)]@np.linalg.solve(K[np.ix_(ia,ia)]+reg*np.eye(len(ia)),y[ia]-ym);
  sc.append([[np.mean(e[:,0]),np.mean(e[:,1:])] for e in [abs(pred-y[it]),abs(ww*pred+(1-ww)*b-y[it])]])
 return np.array(sc)
rows=[]
for a,A in transforms.items():
 for b,B in [('raw',A),('z',np.c_[A,z]),('warp',np.c_[A,z+.5*np.expm1(-.2962*z*z),np.tanh(z+.5*np.expm1(-.2962*z*z))])]:
  B=(B-B.mean(0))/np.maximum(B.std(0),1.e-9); D=B@B.T/B.shape[1]; d=D.diagonal()[:,None]+D.diagonal()[None,:]-2*D
  for j,K in [('lin',D),('quad',(1+D)**2),('cubic',(1+D)**3),('rbf',np.exp(-.5*d))]:
   for r in [.0001,.001,.01,.05]:
    sc=scores(K,r); rows.append((a,b,j,r,sc))
for _,j in [('standalone',0),('blend',1)]:
 print('MODE',j)
 for a,b,k,r,sc in sorted(rows,key=lambda row:row[-1][:,j].mean()+.3*row[-1][:,j].max())[:12]:print(a,b,k,r,'official',sc[0,j].round(4),'mean',sc[:,j].mean(0).round(4),'max',sc[:,j].max(0).round(4))
