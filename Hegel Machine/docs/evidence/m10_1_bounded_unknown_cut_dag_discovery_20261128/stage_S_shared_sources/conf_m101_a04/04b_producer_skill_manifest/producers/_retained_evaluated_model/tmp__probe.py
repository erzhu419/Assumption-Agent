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
print('t',t,'zsv',np.linalg.svd(z,full_matrices=False)[1]); print('Z rows',z[:8].round(5).tolist()); print('Y std',Y.std(0)); print('summary delta',s['train']['summary']-np.r_[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); print('y from z',err(poly(z,1)@fit(poly(z,1)[:52],Y[:52]),Y)[0]);print('y z correlation',np.corrcoef(Y.T)[0]);
for name,a in vari.items():
 print(name,'ranges',a.min(0).round(3),a.max(0).round(3))
 if a.shape[1]==4:print('coef raw',fit(poly(a,1),Y).round(3).T[[0,1,8]],'quad',fit(poly(a,2),Y)[5:].round(3).T[[0,1,8]])
 for d in [1,2,3,4]:
  f=poly(a,d); print('poly',d,err(f[52:]@fit(f[:52],Y[:52]),Y[52:]).round(5),end='; ')
 print()
 for label,kd in [('rbf',lambda d,l:np.exp(-.5*d/l**2)),('m52',lambda d,l:(1+np.sqrt(5*d)/l+5*d/(3*l*l))*np.exp(-np.sqrt(5*d)/l))]:
  D=cdist(a,a,'sqeuclidean'); best=[]
  for l in [.35,.6,1,1.6,2.5,4,8]:
   k=kd(D,l)
   for reg in [1e-5,1e-3,.02,.1]:
    e=err(k[52:,:52]@solve(k[:52,:52]+reg*np.eye(52),Y[:52],assume_a='pos'),Y[52:]); best.append((e[0],l,reg,e[1:].round(5).tolist()))
  print(label,sorted(best)[:2])
