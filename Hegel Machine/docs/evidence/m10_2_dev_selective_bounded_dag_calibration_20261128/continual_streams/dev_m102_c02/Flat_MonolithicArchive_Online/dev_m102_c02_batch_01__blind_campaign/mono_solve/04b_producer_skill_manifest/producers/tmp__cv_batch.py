import pickle, sys
import numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as m
s = pickle.load(open('/holdout_task/public_batch.pkl', 'rb'))[0]['study']; tr=s['train']; x,y,t=(tr[k] for k in ['x','y','traj'])
u=s['panel_b']['u']; b,g=m._calibrate(s['panel_a']); z=(x-b)/g; ztest=(s['test']['x']-b)/g
print('panel range',abs(u).max(0),'train range',abs(z).max(0),'test range',abs(ztest).max(0))
splits=[('gate',np.arange(52,72))] + [('random'+str(k),np.random.default_rng(k).permutation(72)[:20]) for k in range(12)]
splits += [('large'+str(k),np.argsort(abs(z[:,k]))[-12:]) for k in range(4)]
splits += [('joint',np.argsort(np.linalg.norm(z,axis=1))[-15:])]
for kind in ['powered','root']:
 for yf in ['sparse','quadratic']:
  for tf in ['linear','sparse','quadratic']:
   all=[]
   for nm,test in splits:
    train=np.setdiff1d(np.arange(72),test); ss=dict(s,train=dict(tr,x=x[train],y=y[train],traj=t[train]),test={'x':x[test]}); py,pt=m._predict(ss,kind,yf,tf)
    et=abs(pt-t[test]); all.append([abs(py-y[test]).mean(),et.mean(),et.mean(0).max()])
   print(kind,yf,tf,'gate',np.round(all[0],5),'rnd max',np.round(np.max(all[1:13],0),5),'extreme max',np.round(np.max(all[13:],0),5))
