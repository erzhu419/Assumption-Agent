import pickle, numpy as np
from pathlib import Path
root=Path('/holdout_task'); s=pickle.load(open(root/'public_batch.pkl','rb'))[0]['study']; f=[]
for file in ('node_00.py','_frozen_nodes/node_01.py','_frozen_nodes/node_02.py'):
 d={};exec(compile((root/file).read_text(),str(file),'exec'),d);f.append(d['run_node'])
for val in (np.arange(52,72),np.arange(0,72,4),np.arange(2,72,4)):
 u=dict(s); tr=s['train']; train=np.setdiff1d(np.arange(len(tr['x'])),val);u['train']={k:tr[k][train] for k in ('x','y','traj')};u['test']={'x':tr['x'][val]};a=f[0]({'__input__':u});b=f[1]({'__input__':u,'node_00':a});o=f[2]({'node_00':a,'node_01':b});print('slice',val[0],len(val),np.mean(abs(o['y_hat']-tr['y'][val])),np.mean(abs(o['state_pred']-tr['traj'][val])))
