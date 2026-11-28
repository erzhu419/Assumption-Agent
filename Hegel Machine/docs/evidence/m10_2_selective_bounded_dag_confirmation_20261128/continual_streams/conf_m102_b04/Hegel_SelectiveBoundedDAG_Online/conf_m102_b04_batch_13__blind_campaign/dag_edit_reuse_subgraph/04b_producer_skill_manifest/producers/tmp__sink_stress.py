import runpy,pickle,numpy as np
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; nf=runpy.run_path(p+'_frozen_nodes/node_00.py')['run_node']; n=len(s['train']['x']); idx=np.arange(n)
fs=[runpy.run_path(j)['run_node'] for j in [p+'node_01.py','/tmp/refine_sink.py']]
errors=[[],[]]; cases=[(idx[:-20],idx[-20:])]
for seed in [42,197]:
 for j in np.array_split(np.random.default_rng(seed).permutation(n),4): cases.append((np.setdiff1d(idx,j),j))
for i,j in cases:
 data=dict(s); data['train']={k:s['train'][k][i] for k in ('x','y','traj')}; data['test']={'x':s['train']['x'][j]}
 z=nf({'__input__':data})
 for k,f in enumerate(fs):
  o=f({'__input__':data,'node_00':z}); errors[k].append([np.abs(o['y_hat']-s['train']['y'][j]).mean(),np.abs(o['state_pred']-s['train']['traj'][j]).mean()])
a=np.array(errors); print('public',a[:,0,:],'random_mean',a[:,1:,:].mean(1),'random_worst',a[:,1:,:].max(1),flush=True)
valid=np.maximum(a[:,0,:].max(1),.55*a[:,1:,:].max((1,2))+.45*a[:,1:,:].mean((1,2)))
if valid[1]<valid[0] and (a[1,0,:]<.038).all() or (valid[1]<valid[0] and a[1,0,:].max()<a[0,0,:].max()):
 from shutil import copyfile
 copyfile('/tmp/refine_sink.py',p+'node_01.py')
 print('Promoted training-only model selection',flush=True)
