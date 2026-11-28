import numpy as np,pickle,glob,runpy,time
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=S['train']; x,y,T=[tr[k] for k in ('x','y','traj')]; np.set_printoptions(precision=4,suppress=True)
print('times',S['eval_times'],'telemetry',S['telemetry_vec'],'summary',tr['summary']); print('x range',x.min(0),x.max(0),'test range',S['test']['x'].min(0),S['test']['x'].max(0));print('y',y.min(),y.max(),y.std());print('traj means, std, sv',T.mean(0),T.std(0),np.linalg.svd(T-T.mean(0),compute_uv=False));print(np.c_[x,y,T][:8]);
for k in 'abc':
 p=S['panel_'+k]; u=p['u']; A=np.c_[np.ones(len(u)),u];print(k,'range',u.min(0),u.max(0))
 for v in list(p.values())[1:]:
  c=np.linalg.lstsq(A,v,rcond=None)[0]; r=np.linalg.norm(v-A@c,axis=1); print(c,'resid',np.quantile(r,[0,.5,.8,.9,1]))
for p in sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py')):
 t=time.time()
 try:
  s=dict(S,train={k:(v[:52] if k!='summary' else v) for k,v in tr.items()},test={'x':x[52:]}); out=runpy.run_path(p)['solve_batch'](s); print(p.split('/')[-1],np.mean(abs(out['y_hat']-y[52:])),np.mean(abs(out['state_pred']-T[52:])),time.time()-t,flush=True)
 except Exception as e: print(type(e),e)
