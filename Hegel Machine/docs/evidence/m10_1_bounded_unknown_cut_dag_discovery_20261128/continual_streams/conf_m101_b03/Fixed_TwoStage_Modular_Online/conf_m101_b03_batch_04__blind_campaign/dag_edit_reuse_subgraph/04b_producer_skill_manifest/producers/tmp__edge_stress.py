import pickle, runpy
import numpy as np
from scipy.linalg import solve
root='/holdout_task/'
model=runpy.run_path(root+'node_01.py'); features=model['_features']; predict=model['_predict']
with open(root+'public_batch.pkl','rb') as f: study=pickle.load(f)[0]['study']
d=runpy.run_path(root+'_frozen_nodes/node_00.py')['run_node']({'__input__':study})
n=d['n_train']; x=d['z'][:n]; y=d['y'][:,None]; T=d['traj']; A=features(x)
folds=[np.arange(n-20,n), *[j for seed in (39,101,211) for j in np.array_split(np.random.default_rng(seed).permutation(n),6)], np.argsort(np.linalg.norm(x,axis=1))[-12:]]
def trial(tag, fit):
    E=[]
    for j in folds:
        i=np.setdiff1d(np.arange(n),j)
        p=fit(i,j,y); q=fit(i,j,T); e=np.abs(q-T[j])
        E.append([np.mean(np.abs(p-y[j])),e.mean(),e.mean(0).max()])
    E=np.array(E); print(tag,'tail',E[0].round(5),'random',E[1:-1].max(0).round(5),'outer',E[-1].round(5))
trial('current',lambda i,j,v:predict(A[i],v[i],A[j]))
for add_name,extras in [('none',None),('diagonal quartics',x**4)]:
    B=A if extras is None else np.c_[A,extras]
    for strength in (0,0.0001,0.001,0.01):
        def fit(i,j,v):
            scale=np.sqrt(np.mean(B[i]**2,axis=0)); P=B[i]/scale
            if strength==0: co=np.linalg.lstsq(P,v[i],rcond=None)[0]
            else: co=solve(P.T@P+strength*len(i)*np.eye(len(scale)),P.T@v[i],assume_a='pos')
            return B[j]/scale@co
        trial((add_name,strength),fit)
