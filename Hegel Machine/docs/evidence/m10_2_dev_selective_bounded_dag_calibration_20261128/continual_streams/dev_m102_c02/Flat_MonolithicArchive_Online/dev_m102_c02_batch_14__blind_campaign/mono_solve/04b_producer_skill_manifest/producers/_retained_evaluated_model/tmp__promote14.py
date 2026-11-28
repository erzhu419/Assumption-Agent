import importlib.util,pickle,numpy as np

def load(path):
    spec=importlib.util.spec_from_file_location('solver',path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; y=tr['y']; t=tr['traj']; ix=np.arange(len(x))
a,b=[load(p) for p in ('/tmp/batch14_verified.py','/holdout_task/batch_solver.py')]
z=b._calibrated(s['panel_a'],x)
groups=[('public',[ix[-20:]]),('random',np.array_split(np.random.default_rng(11).permutation(len(x)),4))]
for key,v in [('measured_tails',x),('latent_tails',z)]:
    groups.append((key,[np.argsort(col)[part] for col in v.T for part in (slice(-20,None),slice(20))]))

def test(mod,parts):
    vals=[]
    for te in parts:
        r=np.setdiff1d(ix,te)
        out=mod.solve_batch(dict(s,train={k:v[r] for k,v in tr.items() if k!='summary'},test={'x':x[te]}))
        ey=np.abs(out['y_hat']-y[te]); et=np.abs(out['state_pred']-t[te])
        vals.append([ey.mean(),et.mean(),et.mean(0).max(),et[:,-1].mean()])
    return np.max(vals,axis=0)
for name,parts in groups:
    old,new=test(a,parts),test(b,parts)
    print(name,'baseline',old,'challenger',new,flush=True)
    assert np.all(np.isfinite(new)) and np.all(new[1:] <= old[1:]+1e-10)
    assert new[0] < old[0]
    if name=='public': assert np.all(new[:2] <= .038)
