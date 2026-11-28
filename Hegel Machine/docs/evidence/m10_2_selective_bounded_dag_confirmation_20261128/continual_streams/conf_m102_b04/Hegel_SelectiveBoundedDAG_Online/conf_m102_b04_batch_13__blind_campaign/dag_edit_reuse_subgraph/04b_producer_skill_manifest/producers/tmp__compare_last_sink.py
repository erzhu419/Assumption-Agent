import runpy,pickle,numpy as np,shutil
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; nf=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']; idx=np.arange(len(s['train']['x'])); paths=[p+'node_01.py','/tmp/refine_sink.py','/tmp/last_sink.py']; fs=[runpy.run_path(j)['run_node'] for j in paths]; errors=[]
cases=[(idx[:-20],idx[-20:])]; cases += [(np.setdiff1d(idx,h),h) for h in np.array_split(np.random.default_rng(42).permutation(len(idx)),4)]
for i,j in cases:
 data=dict(s); data['train']={k:s['train'][k][i] for k in ('x','y','traj')}; data['test']={'x':s['train']['x'][j]}; z=nf({'__input__':data}); row=[]
 for f in fs:
  o=f({'__input__':data,'node_00':z}); row.append([abs(o['y_hat']-s['train']['y'][j]).mean(),abs(o['state_pred']-s['train']['traj'][j]).mean()])
 errors.append(row)
a=np.array(errors); print('public',a[0],'random mean',a[1:].mean(0),'random worst',a[1:].max(0),flush=True)
score=.6*a[1:].max((0,2))+.4*a[1:].mean((0,2)); passing=np.where(np.max(a[0],axis=1)<=.038)[0]; best=int(passing[np.argmin(score[passing])]) if len(passing) else int(np.argmin(score)); print('retained',paths[best],flush=True)
if best:shutil.copyfile(paths[best],paths[0])
