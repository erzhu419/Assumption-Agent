import pickle,numpy as np
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
np.set_printoptions(precision=4,suppress=True,linewidth=140)
def poly(x,d):
 n,p=x.shape
 terms=[t for i in range(1,d+1) for t in combinations_with_replacement(range(p),i)]
 return np.column_stack([np.ones(n)]+[x[:,t].prod(1) for t in terms]),['1']+[''.join(str(a) for a in t) for t in terms]
def robust(A,y):
 w=np.ones(len(y))
 for scale in [1,.3,.1,.03,.01,.003,.001,.0003]*1:
  for _ in range(15):
   B=np.linalg.lstsq(A*w[:,None],y*w[:,None],rcond=None)[0]; e=y-A@B
   w=1/np.sqrt(1+(np.linalg.norm(e,axis=1)/scale)**2)
 return B,e
if __name__=='__main__':
 for k in ['panel_a','panel_c']:
  p=s[k]; x=p['u']; print(k)
  for key in p.keys()-{'u'}:
   y=p[key]
   for d in range(1,4 if k=='panel_a' else 3):
    a,nm=poly(x,d); b,e=robust(a,y) if k=='panel_a' else (np.linalg.lstsq(a,y,rcond=None)[0],None)
    if e is None:e=y-a@b
    print(key,d,'median / sorted residual norm',np.median(np.abs(e)),np.sort(np.linalg.norm(e,axis=1))[-8:]); print(list(zip(nm,b.tolist())) if False else np.column_stack([nm,b.round(4)]))
 if True:
  for k in ['train','test']:
   x=s[k]['x']; print(k,x.min(0),x.max(0))
  print('times',s['eval_times'])
