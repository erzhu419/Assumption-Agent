import importlib.util, pickle, numpy as np
spec=importlib.util.spec_from_file_location('bs','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; n=len(tr['x']); z=m._calibrate(s['panel_a'],np.r_[tr['x'],s['test']['x']]); Y=np.c_[tr['y'],tr['traj']]; np.set_printoptions(precision=5,suppress=True,linewidth=150)
idx=np.arange(n); splits=[(idx[:52],idx[52:]),(idx[:56],idx[56:])]
for hold in np.array_split(np.random.default_rng(51).permutation(n),4): splits.append((np.setdiff1d(idx,hold),hold))
hold=np.argsort(np.linalg.norm(z[:n],axis=1))[-16:]; splits.append((np.setdiff1d(idx,hold),hold))

def evaluate(f, target=Y):
 out=[]
 for a,b in splits:
  e=np.abs(target[b]-f[b]@m._fit(f[a],target[a])); out.append([e[:,0].mean(),e[:,1:].mean(),e[:,-1].mean(),e[:,1:].mean(0).max()])
 return np.array(out)

if __name__=='__main__':
 f=m._basis(z[:n]); print('calibrated ranges',z.min(0),z.max(0)); print('split MAE y, traj, final, worst horizon',evaluate(f)); print('weights y, first, final',m._fit(f[:52],Y[:52])[:,[0,1,-1]]); print('traj singular directions',np.linalg.svd(Y[:52,1:],full_matrices=False)[2][:2]); print('training noise',np.std(Y[:52]-f[:52]@m._fit(f[:52],Y[:52]),axis=0))
