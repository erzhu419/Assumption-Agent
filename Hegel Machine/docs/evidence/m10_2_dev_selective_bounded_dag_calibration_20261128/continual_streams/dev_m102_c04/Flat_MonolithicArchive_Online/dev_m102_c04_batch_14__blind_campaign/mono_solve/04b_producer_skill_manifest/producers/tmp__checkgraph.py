import sys, pickle, copy, numpy as np
sys.path.insert(0, '/holdout_task')
import batch_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; r=s['train']; n=len(r['y']); rng=np.random.default_rng(9238)
splits=[('prescribed',np.arange(52),np.arange(52,n))]
for k in range(12):
 j=rng.permutation(n); splits.append(('random'+str(k),j[:52],j[52:]))
z=m._calibrate(s['panel_a'],r['x']); p,q=m._panels(s['panel_c'],z); K=(p[:,:,None]*q[:,None,:]).reshape(n,16); dot=np.sum(p*q,axis=1); base=m._design(p,q,dot); qb=m._design(q)
records=[]
for label,i,j in splits:
 v=copy.deepcopy(s); v['train']={k:r[k][i] for k in ('x','y','traj')}; v['test']['x']=r['x'][j]; pred=m.solve_batch(v); ey=abs(pred['y_hat']-r['y'][j]); et=abs(pred['state_pred']-r['traj'][j]); records.append([ey.mean(),et.mean(),et.mean(axis=0).max(),ey.max(),et.max()]);
 t,amp=m._temporal_fit(s['eval_times'],r['traj'][i]);
 spatial=[]
 for label2,A,target in [('scalar',base,r['y'][i]),('fast',base,amp[:,0]),('slow',qb,amp[:,1])]:
  candidates=[A]+[np.column_stack((A,K[:,k])) for k in range(16)];
  selection=[]
  for k,B in enumerate(candidates):
   C=B[i]; H=np.einsum('ij,ji->i',C,np.linalg.pinv(C)); w=m._fit(C,target); e=(C@w-target)/(1-H); selection.append((np.mean(e*e)*(1.02 if k else 1),k))
  k=min(selection)[1]; B=candidates[k]; spatial.append((k-1,B[j]@m._fit(B[i],target)))
 ys,ss=spatial[0][1],np.column_stack((spatial[1][1],spatial[2][1]))@t.T; print(label, np.round(records[-1],6), 'sparse', [p[0] for p in spatial],np.mean(abs(ys-r['y'][j])),np.mean(abs(ss-r['traj'][j])),np.mean(abs(ss-r['traj'][j]),axis=0).max())
print('dense worst',np.max(records,axis=0));
