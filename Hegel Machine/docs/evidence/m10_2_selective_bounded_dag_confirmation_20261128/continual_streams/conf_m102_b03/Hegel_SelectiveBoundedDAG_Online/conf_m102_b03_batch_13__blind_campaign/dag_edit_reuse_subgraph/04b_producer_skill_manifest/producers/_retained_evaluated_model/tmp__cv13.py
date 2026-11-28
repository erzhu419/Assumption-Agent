import sys,pickle,numpy as np
sys.path[:0]=['/holdout_task','/holdout_task/_frozen_nodes']
from node_00 import _features
from node_01 import run_node as predict
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['x']); print({k:v.shape for k,v in s['panel_c'].items()},flush=True)
def bench(m,v):
 results=[]
 for a in [np.arange(n), np.random.default_rng(42).permutation(n),np.random.default_rng(120).permutation(n)]:
    tr,te=a[:-20],a[-20:]; st=dict(s,train={k:t[k][tr] for k in ('x','y','traj')},test={'x':t['x'][te]})
    p=predict({'__input__':st,'node_00':_features(st,m,v)}); results.append([np.mean(abs(p['y_hat']-t['y'][te])),np.mean(abs(p['state_pred']-t['traj'][te]))])
 return np.array(results)
if __name__=='__main__':
 out=[]
 for m in range(6):
  for v in range(7):
   A=bench(m,v); out.append((A.max(),m,v,np.round(A,5).tolist()))
 print(*sorted(out)[:20],sep='\n')
