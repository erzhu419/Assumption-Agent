import pickle, runpy
import numpy as np
base='/holdout_task/'
f=[runpy.run_path(base+p)['run_node'] for p in ('_frozen_nodes/node_00.py','node_01.py','node_02.py')]
s=pickle.load(open(base+'public_batch.pkl','rb'))[0]['study']; t=s['train']; n=len(t['x']); i=np.arange(n); r=np.random.default_rng(503)
m=runpy.run_path(base+'node_01.py'); w=m['_invert'](t['x'],m['_calibrate'](s['panel_a']))
cuts=[i[j::4] for j in range(4)]+[r.permutation(n)[:20] for _ in range(24)]+[i[-20:],np.argsort(np.linalg.norm(w,axis=1))[-14:]]
errs=[]; per_time=[]
for te in cuts:
 tr=np.setdiff1d(i,te); a={**s,'train':{k:t[k][tr] for k in ('x','y','traj')},'test':{'x':t['x'][te]}}
 raw=f[0]({'__input__':a}); fit=f[1]({'__input__':a,'node_00':raw}); out=f[2]({'node_00':raw,'node_01':fit}); yt=out['y_hat']; p=out['state_pred']; dt=abs(p-t['traj'][te]);
 assert np.allclose(out['diagnostics_vec'], [yt.mean(),yt.std(),p[:,-1].mean(),p[:,-1].std()]); errs.append([abs(yt-t['y'][te]).mean(),dt.mean()]); per_time.append(dt.mean(0))
print('Final artifact: 30 split maximum MAEs',np.max(errs,axis=0),'radial',errs[-1],'per-time maxima',np.max(per_time,axis=0)); assert np.max(errs) < .038; assert np.max(per_time) < .038
