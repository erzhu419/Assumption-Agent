import sys,pickle,glob
import numpy as np
from coords09 import coords,aff,ls
sys.path.insert(0,'/tmp/hegel_continual_archive')
from conf_m102_b04_batch_01_solver import gp
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; X=tr['x']; Y=tr['y']; Z=tr['traj']; n=52
out=coords(s,X,True); _,_,vt=np.linalg.svd(Z[:n],full_matrices=False); U=Z@vt[:2].T
print('y vs traj',ls(aff(U[:n]),Y[:n]),np.mean(abs(aff(U[n:])@ls(aff(U[:n]),Y[:n])-Y[n:])),flush=True)
sets=dict(out)
for j in out:
 if j in ('x','q'):continue
 for l in ('x','q'):sets[l+'+'+j]=np.c_[out[l],out[j]]
for l in ('x','q'):
 sets[l+'all']=np.c_[out[l],out['b'+l],out['c'+l],out['d'+l]]
sets['all']=np.c_[out['x'],out['q'],out['bq'],out['cq'],out['dq'],out['bx'],out['cx'],out['dx']]
res=[]; preds=[]
for key,q in sets.items():
 q=(q-q[:n].mean(0))/q[:n].std(0)
 for add in (True,False):
  y=gp(q[:n],q[n:],Y[:n],add); uu=np.column_stack([gp(q[:n],q[n:],U[:n,j],add) for j in range(2)]); z=uu@vt[:2]; py,pz=np.mean(abs(y-Y[n:])),np.mean(abs(z-Z[n:])); res.append((py,pz,key,add));preds.append(np.c_[y,z]);
for j in (0,1):print('top',j,sorted(res,key=lambda r:r[j])[:12]);
np.savez('/tmp/gps09.npz',preds=preds,names=[str(r[2:4]) for r in res]);
