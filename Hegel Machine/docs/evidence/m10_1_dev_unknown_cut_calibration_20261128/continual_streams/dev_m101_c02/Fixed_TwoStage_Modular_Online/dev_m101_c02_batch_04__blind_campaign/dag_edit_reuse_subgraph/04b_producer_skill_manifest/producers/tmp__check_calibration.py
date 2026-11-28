import sys
sys.path.insert(0, '/holdout_task')
from panels import p, np
from node_00 import _fit_a, _invert_a, _poly, _fit_b, _basis_b
from scipy.optimize import minimize_scalar
from itertools import combinations_with_replacement
from importlib.util import spec_from_file_location, module_from_spec
s=spec_from_file_location('sink','/holdout_task/_frozen_nodes/node_01.py'); sink=module_from_spec(s); s.loader.exec_module(sink)
a,b=p['panel_a'],p['panel_b']; u,v=b['u'],b['v']; omega,beta=_fit_b(u,v); A=_fit_a(a['u'],a['v']); x=p['train']['x']; z=_invert_a(x,A)
print('A inversion closure train/test and minimum Jacobian singular value:')
for raw in (x,p['test']['x']):
 q=_invert_a(raw,A); J=np.stack([(_poly(q+e*1e-5)@A-_poly(q-e*1e-5)@A)/2e-5 for e in np.eye(4)],axis=2); print(np.max(abs(_poly(q)@A-raw)),np.linalg.svd(J,compute_uv=False).min())
terms=list(combinations_with_replacement(range(4),2))
for j in range(4):
 I=(0,3) if j<2 else (1,2); Q=_basis_b(u,omega); base=np.linalg.norm(v[:,j]-Q@beta[:,j]); opts=[]
 for name,g in [(str(t),u[:,t].prod(1)) for t in terms]+[('cos'+str(k),np.cos(omega*u[:,k])) for k in range(4)]+[('pairproj^2',(Q[:,1:]@beta[1:,j])**2)]:
  D=np.c_[Q,g]; c=np.linalg.lstsq(D,v[:,j],rcond=None)[0]; r=v[:,j]-D@c; h=(D*np.linalg.pinv(D).T).sum(1); opts.append((np.linalg.norm(r/(1-h)),name,c[-1],np.linalg.norm(r)))
 print('B output',j,'base norm',base,'best added terms',sorted(opts)[:5])

def assess(label,data):
 rows=[]
 for val in np.array_split(np.arange(len(x)),6):
  tr=np.setdiff1d(np.arange(len(x)),val); st=dict(p); st['train']={k:p['train'][k][tr] for k in ('x','y','traj')}; st['test']={'x':x[val]}; st['train']['summary']=p['train']['summary']; out=sink.run_node({'__input__':st,'node_00':{'z_tr':data[tr],'z_te':data[val]}})
  rows.append([np.mean(abs(out['y_hat']-p['train']['y'][val])),np.mean(abs(out['state_pred']-p['train']['traj'][val]))])
 print(label,'worst blocked MAEs',np.max(rows,axis=0),'folds',np.asarray(rows))
assess('harmonic baseline',_basis_b(z,omega)@beta)
