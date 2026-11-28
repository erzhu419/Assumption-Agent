import runpy,pickle,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x,y,T=[tr[k] for k in ['x','y','traj']]; h,hs=m['_designs'](s,x); z=m['_maps'](s,x)[0]; n=len(x); rng=np.random.default_rng(1)
splits=[np.arange(52,72)]+[rng.choice(n,20,replace=False) for _ in range(30)]+[np.argsort(q)[-20:] for q in [np.linalg.norm(z,axis=1),*z.T,*(-z.T)]]
E=[]
for ts in splits:
 ix=np.setdiff1d(np.arange(n),ts); Q=np.array([hh[ts]@m['_fit'](hh[ix],y[ix]) for hh in hs]); st=m['_trajectory'](h[ix],h[ts],T[ix],s['eval_times']); E.append(np.r_[np.mean(abs(Q-y[ts]),axis=1),np.mean(abs(Q.mean(0)-y[ts])),np.mean(abs(st-T[ts])),np.mean(abs(h[ts]@m['_fit'](h[ix],T[ix])-T[ts]))]);
E=np.array(E); print('public',E[0]); print('random mean',E[1:31].mean(0),'random max',E[1:31].max(0)); print('directional',E[31:]); print('z extent',z.min(0),z.max(0)); print('times',s['eval_times']); print('traj sv',np.linalg.svd(T,compute_uv=False));
