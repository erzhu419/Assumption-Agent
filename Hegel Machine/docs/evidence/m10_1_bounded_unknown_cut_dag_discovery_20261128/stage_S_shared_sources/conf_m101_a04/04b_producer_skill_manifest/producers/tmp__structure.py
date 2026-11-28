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
print('times',t,'sv of centered traj',np.linalg.svd(z-z.mean(0),compute_uv=False));print('summary delta',s['train']['summary']-np.r_[y.mean(),y.std(),z[:,-1].mean(),z[:,-1].std()]); print('y corr',np.corrcoef(Y.T)[0].round(3));
for name,a in vari.items():
 if a.shape[1]!=4: continue
 f=poly(a,3); co=fit(f,Y); labs=['1']+[''.join(map(str,ii)) for k in range(1,4) for ii in combinations_with_replacement(range(4),k)]; print(name,'full fit',err(f@co,Y).round(5));
 for j in [0,1,3,8]: print(j, sorted(zip(labs,co[:,j].round(4)),key=lambda q:-abs(q[1]))[:13]);
 print('singular values of coefficients in time',np.linalg.svd(co[:,1:],compute_uv=False).round(4));
# Test whether directly accounting for the observed trajectory exposes a simple scalar readout.
for name,a in vari.items():
 if a.shape[1]!=4: continue
 for j in [0,3,7]:
  for d in [1,2]:
   f=poly(np.c_[a,z[:,j]],d); co=fit(f[:52],y[:52]); print('observed state',name,j,d,round(np.mean(abs(f[52:]@co-y[52:])),5))
