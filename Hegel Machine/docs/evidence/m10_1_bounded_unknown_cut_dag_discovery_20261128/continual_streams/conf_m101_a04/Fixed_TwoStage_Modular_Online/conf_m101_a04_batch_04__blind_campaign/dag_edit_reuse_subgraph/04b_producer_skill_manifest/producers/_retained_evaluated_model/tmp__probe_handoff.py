import itertools as it
import pickle
import runpy
import numpy as np

s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
node = runpy.run_path('/holdout_task/node_00.py')
read = runpy.run_path('/holdout_task/_frozen_nodes/node_01.py')['_readout_basis']
out = node['run_node']({'__input__': s})
n = out['n_train']; h = out['h'][:n]; blocks = np.split(h, 4, axis=1)
y = np.column_stack((out['y'], out['traj']))
p = s['panel_a']; a = node['_fit_a'](p); err = np.linalg.norm(p['v']-node['_affine'](p['u'])@a,axis=1)
print('panel A sorted forward residuals',np.sort(err)[[0,23,40,44,45,46,47]],'cond',np.linalg.cond(a[1:]))
print('target singular values',np.linalg.svd(y-y.mean(0),compute_uv=False))

def report(name, f, pen=1, fit=True):
 k = f@f.T
 scores=[]
 for lam, Y in ((.0003,y[:,:1]),(.001,y[:,1:])):
  C=np.linalg.inv(k+pen*lam*np.eye(n)); e=(C@Y)/C.diagonal()[:,None]; g=np.abs(e)
  scores.extend((g.mean(), g.mean(0).max()))
 if fit:
  coef=np.linalg.lstsq(f,y,rcond=1.e-8)[0]; rms=np.sqrt(np.mean((f@coef-y)**2,axis=0))
  print(name, f.shape[1],np.round(scores,5),'fit rms',np.round(rms[[0,1,-1]],5))
 return scores

for m in range(1,5):
 for inds in it.combinations(range(4),m):
  x=np.column_stack([blocks[i] for i in inds]); report(''.join('ub12'[i] for i in inds),read(x))
for form,x in [('u+b+v1+v2',sum(blocks)),('b+v1+v2',sum(blocks[1:])),('v1+v2',sum(blocks[2:])),('linear-all',h),('primitive',np.c_[blocks[0],node['_v1_basis'](blocks[0])[:,1:],np.cos(2.4*blocks[0])])]:
 report(form, np.c_[np.ones(n),x] if form=='linear-all' else read(x))
for scale in (.03,.1,.25,.5,2.):
 for main in (None,0,1,2,3):
  x=np.column_stack([b*(1. if i==main else scale) for i,b in enumerate(blocks)]); score=report('',read(x),fit=False)
  print('scale',scale,'main',main,np.round(score,5))
