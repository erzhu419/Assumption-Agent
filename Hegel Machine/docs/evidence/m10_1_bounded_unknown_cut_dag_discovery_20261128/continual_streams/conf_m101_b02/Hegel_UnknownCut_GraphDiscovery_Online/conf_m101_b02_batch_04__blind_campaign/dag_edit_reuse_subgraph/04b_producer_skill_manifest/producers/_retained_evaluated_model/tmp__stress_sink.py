import copy, pickle, numpy as np
from sklearn.model_selection import KFold

def node(p):
    ns={}
    exec(compile(open(p).read(),p,'exec'),ns)
    return ns['run_node']
upstream = node('/holdout_task/_frozen_nodes/node_00.py')
sink = node('/holdout_task/node_01.py')
with open('/holdout_task/public_batch.pkl','rb') as fh:
    s=pickle.load(fh)[0]['study']
y,t=s['train']['y'],s['train']['traj']; n=len(y)
splits=[('tail',np.arange(n-20),np.arange(n-20,n))]+[(str(k),i,j) for k,(i,j) in enumerate(KFold(5,shuffle=True,random_state=13).split(y))]
for name,i,j in splits:
    d=copy.deepcopy(s)
    d['train']={key:s['train'][key][i] for key in ('x','y','traj')}
    d['train']['summary']=np.array([y[i].mean(),y[i].std(),t[i,-1].mean(),t[i,-1].std()])
    d['test']={'x':s['train']['x'][j]}
    p=sink({'__input__':d,'node_00':upstream({'__input__':d})})
    e=abs(p['state_pred']-t[j]); print(name,round(abs(p['y_hat']-y[j]).mean(),6),round(e.mean(),6),np.round(e.mean(0),4))
