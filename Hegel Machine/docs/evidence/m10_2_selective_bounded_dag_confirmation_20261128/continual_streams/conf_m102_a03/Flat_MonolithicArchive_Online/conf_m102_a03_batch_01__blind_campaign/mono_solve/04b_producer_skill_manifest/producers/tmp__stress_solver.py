import importlib.util, pickle
import numpy as np

def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

new=load('candidate','/holdout_task/batch_solver.py'); old=load('baseline','/tmp/baseline_solver.py')
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; n=len(x)
a=s['panel_a']; u=a['u']; v=a['v']

def forward(coef,z):
    b=np.roll(z,-1,axis=1)
    return coef[:4]+coef[4:8]*z+coef[8]*b+coef[9]*z*b+coef[10]*z**3

clean=np.linalg.norm(v-u,axis=1)<3
loo=[]
for i in np.flatnonzero(clean):
    take=np.arange(len(u))!=i
    coeff=new._calibration({'u':u[take],'v':v[take]})
    loo.append(np.max(abs(forward(coeff,u[i:i+1])-v[i:i+1])))
coef=new._calibration(a)
print('calibration clean LOO max abs',max(loo),'clean RMSE',np.sqrt(np.mean((forward(coef,u[clean])-v[clean])**2)))
z=new._decode(x,coef); zt=new._decode(s['test']['x'],coef)
print('train/test forward inversion max errors',np.max(abs(forward(coef,z)-x)),np.max(abs(forward(coef,zt)-s['test']['x'])))
assert max(loo)<.001
assert np.max(abs(forward(coef,zt)-s['test']['x']))<1e-8
splits={'specified':np.arange(52,72),'first20':np.arange(20),'mid20':np.arange(26,46),'radial20':np.argsort(np.linalg.norm(z,axis=1))[-20:]}
for k in range(4):
    for sign in [-1,1]:
        splits[f'axis{k}_{sign:+}']=np.argsort(sign*z[:,k])[-20:]
for name, m in [('baseline',old),('structured',new)]:
    bag=[]
    for k, test in splits.items():
        train=np.setdiff1d(np.arange(n),test)
        p=m.solve_batch(dict(s,train={key:(val[train] if key!='summary' else val) for key,val in tr.items()},test={'x':x[test]}))
        y=p['y_hat']; st=p['state_pred']; assert np.allclose(p['diagnostics_vec'],[y.mean(),y.std(),st[:,-1].mean(),st[:,-1].std()])
        e=abs(y-tr['y'][test]); f=abs(st-tr['traj'][test]); vals=np.array([e.mean(),f.mean(),f[:,-1].mean(),f.mean(0).max()]); bag.append(vals)
        if k=='specified' or name=='structured': print(name,k,np.round(vals,7))
    worst=np.max(bag,axis=0); print(name,'worst across slices: y, traj, final, max-time',worst)
    assert np.max(worst)<.038
