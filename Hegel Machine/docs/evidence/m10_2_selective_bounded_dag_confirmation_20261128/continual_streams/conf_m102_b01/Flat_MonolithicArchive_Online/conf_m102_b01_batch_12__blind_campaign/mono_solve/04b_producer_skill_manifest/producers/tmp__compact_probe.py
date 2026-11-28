import importlib.util,pickle,numpy as np
from scipy.optimize import least_squares
spec=importlib.util.spec_from_file_location('solver','/holdout_task/batch_solver.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; u=s['panel_c']['u'];v=s['panel_c']['v1']; w,q=m._coords(s,tr['x']); wt,qt=m._coords(s,s['test']['x']); t=s['eval_times']
def dc(x,r):
 a,b,k=r; return m._aff(np.tanh(a*x)+k*np.sin(b*np.roll(x,-1,axis=-1)))
def resid(r,u,v):
 d=dc(u,r); return (d@m._ls(d,v)-v).ravel()
r=least_squares(resid,[1.75,2.1,.3],args=(u,v),gtol=1e-13).x
for rates in (r,np.array([1.75,2.1,.3])):
 d=dc(u,rates); inv=np.linalg.pinv(d); e=d@inv@v-v; loo=e/(1-(d*inv.T).sum(1)[:,None]); print('compact C1',rates,'RMSE',np.std(e),'LOO',np.std(loo))
r=np.array([1.75,2.1,.3]); co=m._ls(dc(u,r),v); qc=dc(w,r)@co
print('test max |w|',np.max(abs(wt),axis=0),'train',np.max(abs(w),axis=0),'test C delta',np.max(abs(qt-dc(wt,r)@co),axis=0))
rg=np.random.default_rng(98); order=[np.arange(len(w))]+[rg.permutation(len(w)) for _ in range(16)];
for a in [np.linalg.norm(w,axis=1),np.linalg.norm(tr['x'],axis=1),np.max(abs(qc),axis=1),abs(tr['y']),abs(tr['traj'][:,-1])]: order.append(np.argsort(a));
for opt in ['baseline','compact','compact_tied','compact_tied_direct']:
 vals=[]
 for ix in order:
  a,b=ix[:52],ix[52:]; qx=q if opt=='baseline' else qc; opp=m._interactions(qx);
  dy=m._aff(np.c_[qx,w,opp] if opt=='baseline' else np.c_[qx,opp if opt=='compact' else opp.sum(1)]); dp=m._aff(np.c_[qx,w] if opt=='baseline' else qx)
  y=dy[b]@m._ls(dy[a],tr['y'][a]); p=m._trajectory(dp[a],dp[b],t,tr['traj'][a]) if opt!='compact_tied_direct' else dp[b]@m._ls(dp[a],tr['traj'][a]); ey=abs(y-tr['y'][b]);ep=abs(p-tr['traj'][b]);vals.append([ey.mean(),ep.mean(),ep.mean(0).max(),ey.max(),ep.max()]);
 vals=np.array(vals);print(opt, '52/20',vals[0], 'worst random', vals[1:17].max(0),'worst extreme',vals[17:].max(0));print('extreme MAE',np.round(vals[17:,:2],6))
