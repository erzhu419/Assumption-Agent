import pickle,numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; t=s['eval_times']; z=s['train']['traj']; Y=np.c_[y,z]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,i],axis=1) for k in range(1,d+1) for i in combinations_with_replacement(range(x.shape[1]),k)])
def fit(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def err(p,q): return np.r_[np.mean(abs(p[:,0]-q[:,0])),np.mean(abs(p[:,1:]-q[:,1:])),np.mean(abs(p[:,-1]-q[:,-1]))]
models={}
for key in ['panel_a','panel_b','panel_c']:
 p=s[key]
 for v in sorted(set(p)-{'u'}):
  a=p['u']; b=p[v]; F=poly(a,1); w=np.arange(len(a))
  if key=='panel_a':
   for _ in range(8):
    res=np.linalg.norm(F@fit(F[w],b[w])-b,axis=1); w=np.argsort(res)[:38]
  C=fit(F[w],b[w]); r=np.linalg.norm(F@C-b,axis=1); G=poly(a,2); D=fit(G[w],b[w]); print(key,v,'linear matrix',C.round(5).tolist(),'sv',np.linalg.svd(C[1:],compute_uv=False).round(5),'residual quantiles',np.quantile(r,[0,.5,.75,.9,1]).round(6),'quad rmse',np.mean((G[w]@D-b[w])**2)**.5,'quad rows',D[5:].round(4).tolist());models[key+v]=(C,D)
A=models['panel_av'][0]; B=models['panel_bv'][0]
vari={'x':x,'a+':poly(x,1)@A,'a-':(x-A[0])@np.linalg.inv(A[1:]),'b+':poly(x,1)@B,'b-':(x-B[0])@np.linalg.inv(B[1:])}
for n in ['a+','a-']:
 vari[n+'b+']=poly(vari[n],1)@B;vari[n+'b-']=(vari[n]-B[0])@np.linalg.inv(B[1:]);vari[n+'c']=np.c_[*[poly(vari[n],2)@models['panel_c'+v][1] for v in ['v1','v2']]]
from scipy.linalg import solve
from itertools import combinations_with_replacement as cwr
print('times',t,'centered traj singular',np.linalg.svd(z-z.mean(0),compute_uv=False)); print('corr y',np.corrcoef(Y.T)[0].round(3))
for j in range(4):
 p=s['panel_c']; a=p['u']; b=p['v'+str(1+j//2)]; q=poly(a,2); print('C SVD residual',j,np.linalg.svd(b-poly(a,1)@fit(poly(a,1),b),compute_uv=False).round(6)); print('C quadratic',fit(q,b)[5:].round(5).T); break
p=s['panel_c']; a=p['u']; b=p['v2']; print('C2 quadratic coefficient',fit(poly(a,2),b)[5:].round(5).T,'sv residual quad',np.linalg.svd(b-poly(a,2)@fit(poly(a,2),b),compute_uv=False));
rng=np.random.default_rng(64); splits=[(np.arange(52),np.arange(52,72))]+[(q[:52],q[52:]) for q in (rng.permutation(72) for _ in range(4))];
def scores(K,T=Y,regs=(1e-4,.001,.01,.05,.2)):
 out=[]
 for reg in regs:
  e=[]
  for i,j in splits:
   pred=K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),T[i],assume_a='pos'); e.append(err(pred,T[j]))
  e=np.array(e); out.append((reg,e[0],e.mean(0),e.max(0)))
 return out
res=[]
for name in ['x','a+','a-','a-b+','a-b-','a-c']:
 a=vari[name]
 kernels={}
 for d in [2,3,4,5,6]:
  F=poly(a,d); std=np.maximum(F.std(0),.07); std[0]=1
  for degree_decay in [.20,.5,1]:
   ds=np.array([0]+[k for k in range(1,d+1) for _ in cwr(range(a.shape[1]),k)]); W=(degree_decay**np.maximum(ds-2,0))/std; G=F*W; kernels[('poly',d,degree_decay)]=G@G.T
 if a.shape[1]==4:
  for law in ['tanh','sin','asinh','atan','erf','rational']:
   from scipy.special import erf
   fun={'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'atan':np.arctan,'erf':erf,'rational':lambda q:q/(1+q*q)}[law]
   for k in [.75,1.25,2,3]:
    for shift in [-.35,0,.35]:
     u=fun(k*(a-shift)); Q=poly(a,2); G=np.c_[Q,u, u*a, np.einsum('ni,nj->nij',u,a).reshape(len(a),-1),u**2]; kernels[(law,k,shift)]=G@G.T
  for l in [.5,.85,1.4,2.4]:
   for p in [.2,.45,1,2.3]:
    dd=cdist(a*[1,p,p,1],a*[1,p,p,1],'sqeuclidean'); K=np.exp(-.5*dd/l**2); Q=poly(a,2); kernels[('rbf',l,p)]=K+Q@Q.T
 for key,K in kernels.items():
  for reg,e,avg,mx in scores(K):res.append((name,key,reg,e,avg,mx))
for j in [0,1]:
 print('BEST',j)
 for name,key,reg,e,avg,mx in sorted(res,key=lambda q:q[4][j]+.5*q[5][j])[:18]:print(name,key,reg,e.round(5),avg.round(5),mx.round(5))
