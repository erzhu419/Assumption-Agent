import pickle,runpy,numpy as np
from scipy.optimize import least_squares
p='/holdout_task/'
s=pickle.load(open(p+'public_batch.pkl','rb'))[0]['study']; n0=runpy.run_path(p+'node_00.py'); n1=runpy.run_path(p+'node_01.py')['run_node']; n2=runpy.run_path(p+'_frozen_nodes/node_02.py')['run_node']; u=s['panel_b']['u']; v=s['panel_b']['v']; x=s['train']['x']; yy=s['train']['y']; tt=s['train']['traj']; z=n0['_inverse'](x,n0['_calibration'](s['panel_a'])); N=len(x)
folds=[('public',np.arange(N-20,N)),('early',np.arange(20)),('mid',np.arange(26,46)),('interleaved',np.arange(0,N,3)),('large-radius',np.argsort(np.linalg.norm(z,axis=1))[-14:])]
def score(make=None):
    out=[]
    for name,te in folds:
        tr=np.setdiff1d(np.arange(N),te); sub={**s,'train':{'x':x[tr],'y':yy[tr],'traj':tt[tr]},'test':{'x':x[te]}}
        a=n0['run_node']({'__input__':sub}) if make is None else make(tr,te)
        b=n1({'__input__':sub,'node_00':a}); r=n2({'node_00':a,'node_01':b}); e=np.abs(r['state_pred']-tt[te]); assert r['y_hat'].shape==yy[te].shape and e.shape==tt[te].shape and all(np.all(np.isfinite(w)) for w in r.values()); out.append((name,round(np.mean(np.abs(r['y_hat']-yy[te])),5),round(e.mean(),5),round(e.mean(axis=0).max(),5)))
    return out
base=n0['run_node']({'__input__':{**s,'test':{'x':x}}}); i,j=np.triu_indices(z.shape[1]); ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
print('BASELINE',score())
metrics = score()
print('FINAL SLICES', metrics)
assert max(max(y, t, h) for _, y, t, h in metrics) < .038
