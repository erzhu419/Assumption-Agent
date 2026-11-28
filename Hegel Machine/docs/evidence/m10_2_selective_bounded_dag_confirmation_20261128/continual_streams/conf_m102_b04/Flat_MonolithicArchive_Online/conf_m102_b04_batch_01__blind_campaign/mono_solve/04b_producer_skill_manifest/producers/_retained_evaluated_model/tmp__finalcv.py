import importlib.util,pickle,numpy as np
spec=importlib.util.spec_from_file_location('batch','/holdout_task/batch_solver.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; Y=np.c_[tr['y'],tr['traj']]
def run(splits):
 E=[]
 for i,j in splits:
  pred=m.solve_batch(dict(s,train={**tr,**{k:tr[k][i] for k in ['x','y','traj']}},test={'x':x[j]})); err=abs(np.c_[pred['y_hat'],pred['state_pred']]-Y[j]);E.append(np.r_[err[:,0].mean(),err[:,1:].mean(),err[:,1:].mean(0),err[:,0].max()]);
 return np.array(E)
sp=[(np.arange(52),np.arange(52,72))]; rng=np.random.default_rng(31903)
for k in range(32):
 p=rng.permutation(72);sp.append((p[:52],p[52:]))
if __name__=='__main__':
 e=run(sp); print('required',e[0]); print('random mean',e[1:].mean(0),'max',e[1:].max(0)); print('B coordinates span',[np.ptp(w,axis=0) for w in m._coordinates(s,x)]); print('test output', m.solve_batch(s)['diagnostics_vec'])
