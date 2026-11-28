import numpy as np, pickle, runpy
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']]; f=list(m['_features'](s,x)); rng=np.random.default_rng(35); parts=np.array_split(rng.permutation(52),4); splits=[(np.arange(52),np.arange(52,72)),(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(np.arange(52),z),z) for z in parts]; te=parts[1]; tr=splits[3][0]; print('row 11 raw',x[11], 'target', y[11]); print('row 11 features',f[0][11,:8]);
for r in [0.0001,0.001,0.003,0.01,0.03,0.1]:
 errs=[]
 for tr,te in splits:
  pred=np.mean([m['_readout'](z[np.r_[tr,te]],y[tr],len(tr),r) for z in f],axis=0); e=abs(pred-y[te]); errs.append([e[:,0].mean(),e[:,1:].mean()]);
  if 11 in te and len(tr)==39: row=e[te==11,0][0]
 print(r,np.round(errs,5).tolist(),'worst inner',np.max(errs[2:],axis=0),'row 11',row)
for k,z in enumerate(f):
 tr,te=splits[3]; g=(z-z[tr].mean(0))/np.maximum(z[tr].std(0),1e-5); U,w,V=np.linalg.svd(g[tr],full_matrices=False); H=(g[te]@V.T*(w/(w*w+.001)))@U.T; l=np.linalg.norm(H,axis=1); print('bridge',k,'relative prediction amplification',l[te==11],np.median(l),'largest',te[np.argsort(l)[-3:]])
