import copy, importlib.util, pathlib, pickle, numpy as np
p=pathlib.Path('/holdout_task/batch_solver.py'); spec=importlib.util.spec_from_file_location('solver',p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
extra=pathlib.Path('/tmp/warp15.txt').read_text(); exec(extra,mod.__dict__)
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; T=s['train']; rng=np.random.default_rng(545); rows=[]; weights=np.array([0.,.25,.50,.75,1.]);
for _ in range(6):
 a,b=np.split(rng.permutation(52),[39]); q=copy.deepcopy(s); q['train'].update({k:T[k][a] for k in ('x','y','traj')}); q['test']={'x':T['x'][b]}
 base=mod.solve_batch(q); warp=mod._warp_candidate(q)
 rows.append([[np.mean(abs((1-w)*base[k]+w*warp[k]-T[t][b])) for k,t in [('y_hat','y'),('state_pred','traj')]] for w in weights]);
R=np.array(rows); score=np.max(R.mean(0),axis=1)+.2*np.max(R.max(0),axis=1); i=int(np.argmin(score)); w=weights[i] if score[i]<.97*score[0] and np.all(R[:,i].mean(0)<R[:,0].mean(0)) else 0.
print('Inner MAE means:',R.mean(0).tolist(), 'worst slices:',R.max(0).tolist(),'warp weight:',w,flush=True)
if w:
 text=p.read_text(); p.write_text(text.replace('def solve_batch(study:dict)->dict:', 'def _previous(study:dict)->dict:')+extra+f'\n\ndef solve_batch(study:dict)->dict:\n    a=_previous(study); b=_warp_candidate(study); w={w!r}\n    return _out((1-w)*a["y_hat"]+w*b["y_hat"], (1-w)*a["state_pred"]+w*b["state_pred"])\n')
