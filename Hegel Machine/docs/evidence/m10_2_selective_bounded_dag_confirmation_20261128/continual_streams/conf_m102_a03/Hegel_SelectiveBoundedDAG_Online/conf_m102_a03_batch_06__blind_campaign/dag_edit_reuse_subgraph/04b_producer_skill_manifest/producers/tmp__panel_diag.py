import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def fit(A,y):return np.linalg.lstsq(A,y,rcond=1e-10)[0]
def exam(u,v,typ):
 n,d=u.shape; cols=[np.ones(n)]; names=['1'];
 for deg in range(1,typ+1):
  for ix in itertools.combinations_with_replacement(range(d),deg):
   if deg==3 and len(set(ix))!=1:continue
   cols.append(np.prod(u[:,ix],axis=1)); names.append(''.join(map(str,ix)))
 A=np.column_stack(cols); b=fit(A,v); e=np.linalg.norm(A@b-v,axis=1); idx=np.argsort(e)[:int(n*.86)]; b=fit(A[idx],v[idx]); e=np.linalg.norm(A@b-v,axis=1); print('degree',typ,'rms, median, high',np.sqrt(np.mean(e**2)),np.median(e),np.sort(e)[-8:]);
 if typ==1 or (typ==3 and n>30):print(list(zip(names,b.round(4))))
for k in ('panel_a','panel_b','panel_c'):
 for label,v in s[k].items():
  if label!='u':
   print(k,label)
   for degree in range(1,4):exam(s[k]['u'],v,degree)
print('svd trajectory',np.linalg.svd(s['train']['traj'],compute_uv=False)); print('x quantiles',np.quantile(s['train']['x'],[0,.1,.5,.9,1],axis=0))
