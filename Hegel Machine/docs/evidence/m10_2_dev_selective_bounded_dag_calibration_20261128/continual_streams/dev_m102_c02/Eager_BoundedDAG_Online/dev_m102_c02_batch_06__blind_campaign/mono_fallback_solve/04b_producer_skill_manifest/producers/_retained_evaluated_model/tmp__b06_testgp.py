import pickle,sys,numpy as np
sys.path.insert(0,'/tmp');import b06_previous as b
from b06_gp import _adaptive
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];t=s['train'];Y=np.c_[t['y'],t['traj']];D=b._coordinates(s,t['x']);k,h=s['telemetry_vec'][:2];rng=np.random.default_rng(295)
for ii in [np.arange(len(Y)),rng.permutation(len(Y)),rng.permutation(len(Y)),rng.permutation(len(Y))]:
 q=_adaptive({a:z[ii] for a,z in D.items()},Y[ii[:52]],k,h,debug=ii[0]==0);E=np.mean(abs(q-Y[ii[52:]]),axis=0);print('adaptive split',round(E[0],6),round(E[1:].mean(),6),np.round(E[1:],6),flush=True)
