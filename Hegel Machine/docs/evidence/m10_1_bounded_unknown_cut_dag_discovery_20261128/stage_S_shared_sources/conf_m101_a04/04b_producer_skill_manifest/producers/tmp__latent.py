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
from scipy.optimize import minimize_scalar
from scipy.linalg import solve
p=s['panel_c']; u=p['u'];
for k in [3,3.2,3.25,3.3,3.4,3.5]:
 f=np.c_[1+0*u[:,0],u,np.sin(k*u),np.cos(k*u)]; print('panel C1 Fourier',k,np.std(f@fit(f,p['v1'])-p['v1']))
k1=minimize_scalar(lambda k:np.mean((np.c_[1+0*u[:,0],u,np.sin(k*u),np.cos(k*u)]@fit(np.c_[1+0*u[:,0],u,np.sin(k*u),np.cos(k*u)],p['v1'])-p['v1'])**2),bounds=(2.5,4),method='bounded').x
print('freq',k1)
def cfeat(a,k,law):return np.c_[1+0*a[:,0],a,np.sin(k*a),np.cos(k*a)] if law==1 else np.c_[1+0*a[:,0],a,np.cos(k*a)]
c1=fit(cfeat(u,k1,1),p['v1']); c2=fit(cfeat(u,1.26,2),p['v2']);print('C1 sin',c1[5:9].round(3),'cos',c1[9:].round(3))
for second in [lambda k:t,lambda k:t*t,lambda k:t*np.exp(-k*t),lambda k:np.sin(t),lambda k:1-np.exp(-.25*t),lambda k:np.ones_like(t)]:
 def tm(k):return np.array([-np.expm1(-k*t),second(k)])
 def fun(k):
  F=tm(k);return np.sqrt(np.mean((z@np.linalg.pinv(F)@F-z)**2))
 r=minimize_scalar(fun,bounds=(.3,4),method='bounded'); print('time',r.x,r.fun)
rng=np.random.default_rng(64); splits=[(np.arange(52),np.arange(52,72))]+[(q[:52],q[52:]) for q in (rng.permutation(72) for _ in range(4))]
def compare(f,reg):
 K=f@f.T; e=[]
 for i,j in splits:
  e.append(err(K[np.ix_(j,i)]@solve(K[np.ix_(i,i)]+reg*np.eye(len(i)),Y[i],assume_a='pos'),Y[j]))
 return np.array(e)
res=[]
for n,a in vari.items():
 if a.shape[1]!=4:continue
 C=cfeat(a,k1,1)@c1; D=cfeat(a,1.26,2)@c2
 for h,g in [('c1',C),('c2',D),('pair',np.c_[C,D]),('x+c1',np.c_[a,C]),('x+c2',np.c_[a,D]),('x+pair',np.c_[a,C,D]),('x',a)]:
  for d in [1,2,3]:
   f=poly(g,d); ds=np.array([0]+[k for k in range(1,d+1) for _ in combinations_with_replacement(range(g.shape[1]),k)]); f=f*.4**np.maximum(ds-1,0)
   for reg in [1e-5,.0003,.003,.03]:
    E=compare(f,reg);res.append((n,h,d,reg,E[0],E.mean(0),E.max(0)))
 for k in [1.5,1.8,2.1,2.4,2.7,3,3.3,3.6]:
  sn=np.sin(k*a); cn=np.cos(k*a)-1; H=np.c_[sn,cn]; P=poly(a,2); I=(H[:,:,None]*a[:,None,:]).reshape(len(a),-1)
  for nm,f in [('fourier',np.c_[P,H]),('mixed',np.c_[P,H,.6*I]),('fourier2',np.c_[P,H,.6*I,.3*poly(H,2)[:,9:]])]:
   for reg in [1e-5,.0003,.003,.03]:
    E=compare(f,reg);res.append((n,nm,k,reg,E[0],E.mean(0),E.max(0)))
for j in [0,1]:
 print('BEST',j)
 for n,h,d,reg,e,avg,mx in sorted(res,key=lambda q:q[5][j]+.6*q[6][j])[:12]:print(n,h,d,reg,e.round(5),avg.round(5),mx.round(5))
