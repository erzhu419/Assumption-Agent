import importlib.util,pickle,numpy as np
sp=importlib.util.spec_from_file_location('candidate','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; y=t['y']; traj=t['traj']; x=t['x']; a,g=m._calibrate(s['panel_a']); z=m._operator(s['panel_b'])((x-a)/g)
rng=np.random.default_rng(1448); folds=[('public',np.arange(72))]+[(f'random{i}',rng.permutation(72)) for i in range(16)]+[('largest-z',np.argsort(np.linalg.norm(z,axis=1))),('largest-x',np.argsort(np.linalg.norm((x-a)/g,axis=1)))]+[(f'z{k}{side:+}',np.argsort(side*z[:,k])) for k in range(4) for side in [-1,1]]
rs=[]; ts=[]
for label,ix in folds:
 i,j=ix[:56],ix[56:]; o=m.solve_batch(dict(s,train={k:t[k][i] for k in ('x','y','traj')},test={'x':x[j]})); yp,st,d=(o[k] for k in ('y_hat','state_pred','diagnostics_vec')); assert yp.shape==(16,) and st.shape==(16,8) and d.shape==(4,); assert all(np.isfinite(v).all() for v in (yp,st,d)); np.testing.assert_array_equal(d,[yp.mean(),yp.std(),st[:,-1].mean(),st[:,-1].std()]); E=abs(st-traj[j]); rs.append([abs(yp-y[j]).mean(),E.mean(),E.mean(axis=0).max()]);
 variants=[]
 for A in (m._affine(z),z):
  Q,R=np.linalg.qr(A[i]); target=Q.T@traj[i]; U,S,V=np.linalg.svd(target,full_matrices=False)
  for rank in (2,len(S)):
   c=m._ls(R,(U[:,:rank]*S[:rank])@V[:rank]); e=abs(A[j]@c-traj[j]); variants.append(e.mean())
 ts.append(variants)
 print(label,np.round(rs[-1],6),np.round(variants,6))
print('worst metric',np.max(rs,axis=0),'trajectory variant worst',np.max(ts,axis=0),'means',np.mean(ts,axis=0))
