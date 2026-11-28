import pickle,runpy,numpy as np
m=runpy.run_path('/holdout_task/batch_solver.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['y']); ix=np.arange(n); rng=np.random.default_rng(949)
f=m['_design'](s,tr['x']); Y=np.c_[tr['y'],tr['traj']]; print('design rank',np.linalg.matrix_rank(f), 'condition',np.linalg.cond(f)); print('singular values',np.linalg.svd(f,compute_uv=False)[-10:]);
for title,sets in [('required',[ix[52:]]),('blocks',[ix[i:i+20] for i in [0,13,26,39,52]]),('random',[rng.choice(n,20,False) for _ in range(30)]),('boundary',[np.argsort(np.linalg.norm(tr['x'],axis=1))[-20:]] )]:
 out=[]
 for val in sets:
  t=np.setdiff1d(ix,val); b=np.linalg.svd(Y[t,1:],full_matrices=False)[2][:3]; pp=f[val]@m['_ls'](f[t],np.c_[Y[t,0],Y[t,1:]@b.T]); e=abs(np.c_[pp[:,0],pp[:,1:]@b]-Y[val]); out.append([e[:,0].mean(),e[:,1:].mean(),e[:,1:].mean(0).max(), e[:,0].max(), e[:,1:].mean(1).max()]);
 print(title,'median',np.median(out,0),'worst',np.max(out,0))
B=m['_ls'](f,Y); print('full train MAE',abs(f@B-Y).mean(0)); print('gh coef y and last',np.round(B[-16:][:,[0,-1]].reshape(4,4,2),4)); print('direct y from traj', np.mean(abs(m['_with_bias'](Y[:,1:])@m['_ls'](m['_with_bias'](Y[:,1:]),Y[:,0])-Y[:,0])))
