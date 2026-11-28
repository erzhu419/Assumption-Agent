import pickle, runpy
from itertools import combinations_with_replacement
import numpy as np
root = '/holdout_task/'
node = runpy.run_path(root + 'node_00.py')['run_node']
sink = runpy.run_path(root + '_frozen_nodes/node_01.py')['run_node']
s = pickle.load(open(root + 'public_batch.pkl', 'rb'))[0]['study']
a = s['train']; x = a['x']; y = a['y']; t = a['traj']; n = len(x); rows = np.arange(n)
h = node({'__input__': dict(s,test={'x': x[:0]})})['h']; h[:,4:] /= 2
folds = [('official', rows[-20:])]
for j in range(x.shape[1]):
    order = np.argsort(x[:,j]); folds += [(f'{j}-',order[:10]),(f'{j}+',order[-10:])]
folds += [('mixed',v) for v in np.array_split(np.random.default_rng(13).permutation(rows),4)]
def evaluate(z, val, fit=None):
    if fit is None: fit = np.setdiff1d(rows,val)
    out = sink({'node_00': {'h':z[np.r_[fit,val]],'n_train':len(fit),'y':y[fit],'traj':t[fit]}})
    dt = abs(out['state_pred']-t[val]); return [abs(out['y_hat']-y[val]).mean(),dt.mean(),dt.mean(axis=0).max()]
options = [('baseline',h*np.r_[np.ones(4),2*np.ones(4)])]
for s1 in (.35,.5,.7,1.,1.4,2.,2.8,4.):
    for ratio in (.7,1.,1.4,2.,2.8):
        options.append(((s1,ratio),h*np.r_[np.full(4,s1),np.full(4,s1*ratio)]))
errs = np.array([[evaluate(z,val) for _,val in folds] for _,z in options]); tails=errs[:,1:9]
print('baseline per fold',[(f,np.round(e,5)) for (f,_),e in zip(folds,errs[0])]); print('scaling challengers, exploratory only')
for i in sorted(range(len(options)),key=lambda i:tails[i,:,:2].max())[:10]:
    print(options[i][0], 'official',np.round(errs[i,0],5),'tails',np.round(tails[i].max(axis=0),5),'mixed',np.round(errs[i,9:].max(axis=0),5))
# Assess the scaling selection rule independently, using only outer training rows.
for (name,val),base in zip(folds,errs[0]):
    fit=np.setdiff1d(rows,val); inn=[]
    for j in range(x.shape[1]):
        order=fit[np.argsort(x[fit,j])]; inn.extend((order[:8],order[-8:]))
    scores=[np.max([evaluate(z,iv,np.setdiff1d(fit,iv))[:2] for iv in inn]) for _,z in options]
    i=int(np.argmin(scores)); print('nested',name,options[i][0],np.round(evaluate(options[i][1],val,fit),5),'base',np.round(base,5))
