import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; y=tr['y']; t=tr['traj']; n=len(y); ids=np.arange(n)
z,*_=m._blocks(s); edge=np.argsort(np.linalg.norm(z[:n],axis=1)); rng=np.random.default_rng(18403)
ordered=[(ids[:i],ids[i:]) for i in (52,56)]; shuffled=[(np.setdiff1d(ids,j),j) for _ in range(5) for j in np.array_split(rng.permutation(n),4)]
all_errors=[]
for it,iv in ordered+shuffled+[(edge[:-12],edge[-12:])]:
    tt={k:tr[k][it] for k in ('x','y','traj')}; q=tt['traj'][:,-1]; yy=tt['y']; tt['summary']=np.array([yy.mean(),yy.std(),q.mean(),q.std()]); r=m.solve_batch(dict(s,train=tt,test={'x':tr['x'][iv]})); yh=r['y_hat']; sh=r['state_pred']; q=sh[:,-1]
    np.testing.assert_allclose(r['diagnostics_vec'],[yh.mean(),yh.std(),q.mean(),q.std()]); e=abs(t[iv]-sh); all_errors.append([np.mean(abs(y[iv]-yh)),e.mean(),e[:,-1].mean()])
a=np.array(all_errors); print('columns: scalar / trajectory / terminal MAE'); print('52/20, 56/16:',a[:2]); print('shuffled maxima:',a[2:-1].max(0)); print('high-amplitude boundary:',a[-1]); assert a[0,0]<=.038
