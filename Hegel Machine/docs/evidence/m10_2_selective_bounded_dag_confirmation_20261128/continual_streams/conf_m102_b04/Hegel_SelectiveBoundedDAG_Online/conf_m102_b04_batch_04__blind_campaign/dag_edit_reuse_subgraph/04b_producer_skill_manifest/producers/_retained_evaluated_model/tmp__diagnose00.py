import numpy as np,pickle,itertools
np.set_printoptions(precision=4,suppress=True,linewidth=140)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def poly(x,d):
 return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,d+1) for j in itertools.combinations_with_replacement(range(x.shape[1]),k)])
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
a=s['panel_a']; u=a['u']; v=a['v']; f=poly(u,1); good=np.arange(len(u));
for _ in range(8):
 b=fit(f[good],v[good]);good=np.argsort(np.linalg.norm(v-f@b,axis=1))[:45]
print('robust A',b,'row residual',np.sort(np.linalg.norm(v-f@b,axis=1)))
for p in ['panel_b','panel_c']:
 for k,v in s[p].items():
  if k=='u':continue
  for d in [1,2]:
   q=poly(s[p]['u'],d); h=np.sum(np.linalg.svd(q,full_matrices=False)[0]**2,axis=1); b=fit(q,v); print(p,k,d,'LOO',np.mean(abs((q@b-v)/(1-h[:,None])))); print(b if d==2 else '')
tr=s['train']; x=tr['x']; z=tr['traj']; U,D,V=np.linalg.svd(z,full_matrices=False); print('trajectory singular values and basis',D,V[:2]); Y=np.c_[tr['y'],z@V[:2].T];
for d in range(1,6):
 f=poly(x,d); pred=[]
 for r in [0.0001,.001,.01,.1,1.]:
  p=f[52:] @ np.linalg.solve(f[:52].T@f[:52]+r*np.eye(f.shape[1]),f[:52].T@Y[:52]);pred.append(np.mean(abs(p-Y[52:]),axis=0))
 print(d,pred)
