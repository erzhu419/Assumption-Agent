import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
from batch_solver import solve_batch,_features,_calibrate,_aff,_ls
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; T=tr['traj']; ix=np.arange(len(x)); a,b=_features(s,x); z=_calibrate(s['panel_a'],x); c=(a[:,:,None]*b[:,None,:]).reshape(len(x),16)
splits=[('required',ix[52:])]+[(f'random{i}',np.random.default_rng(i).permutation(len(x))[52:]) for i in range(20)]
for j in range(4):
 order=np.argsort(x[:,j]); splits.extend([(f'x{j}_tails',np.r_[order[:10],order[-10:]]),(f'x{j}_low',order[:20]),(f'x{j}_high',order[-20:])])
for name, w in [('latent',z),('interactions',c)]: splits.append(('joint_'+name,np.argsort(np.linalg.norm(w,axis=1))[-20:]))
F=_aff(np.c_[a,b,c]); D=_aff(np.c_[a,b,a*b]); records={}
for name,te in splits:
 ti=np.setdiff1d(ix,te); st=dict(s,train={k:(v if k=='summary' else v[ti]) for k,v in tr.items()},test={'x':x[te]}); out=solve_batch(st); p=out['state_pred']; Q=np.linalg.svd(T[ti],full_matrices=False)[2][:2]; V=T[ti]@Q.T
 assert np.allclose(p,F[te]@_ls(F[ti],V)@Q)
 E={'artifact':p,'diagonal_only':D[te]@_ls(D[ti],V)@Q}
 W=F[ti,1:]; Z=F[te,1:]; mu=W.mean(0); sd=W.std(0); W=(W-mu)/sd; Z=(Z-mu)/sd
 for lam in [0.0001,0.001,0.01,0.1]: E[f'ridge_{lam}']=(V.mean(0)+Z@np.linalg.solve(W.T@W+lam*np.eye(W.shape[1]),W.T@(V-V.mean(0))))@Q
 for model,pred in E.items():
  e=np.abs(pred-T[te]); records.setdefault(model,[]).append((name,e.mean(),e.mean(0).max()))
 if name=='required':
  for model,pred in E.items(): print('required',model,'trajectory_mae',np.abs(pred-T[te]).mean())
 if name in ('required','joint_latent','joint_interactions'): print('scalar_mae',name,np.abs(out['y_hat']-y[te]).mean())
for model,row in records.items():
 print('worst',model,max(row,key=lambda r:r[1]),'worst_horizon',max(row,key=lambda r:r[2]),'median',np.median([r[1] for r in row]))
