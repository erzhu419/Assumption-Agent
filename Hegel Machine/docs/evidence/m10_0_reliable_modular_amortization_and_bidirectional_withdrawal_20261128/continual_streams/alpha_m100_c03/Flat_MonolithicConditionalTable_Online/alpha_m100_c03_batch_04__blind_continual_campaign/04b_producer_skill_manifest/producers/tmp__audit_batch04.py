import pickle, runpy
from pathlib import Path
import numpy as np

p = Path('/holdout_task')
m = runpy.run_path(str(p / 'batch_solver.py'))
d = pickle.load(open(p / 'public_batch.pkl', 'rb'))[0]['dataset']
c = d['calibration_load_cells']
y, r = c['true_channel_loads'], c['bridge_raw']
raw_sum = r.sum(1)
axis = np.count_nonzero(y, axis=1) == 1
base = m['_bridge_terms'](y)
pairs = [(i,j) for i in range(4) for j in range(i+1,4)]
other = np.column_stack([y[:,0]*y[:,2], y[:,1]*y[:,3]])
contrasts = np.column_stack([(y[:,i]+sgn*y[:,j])**3 for i,j in pairs for sgn in (-1,1)])

def loo(a):
    pred = np.empty(len(y))
    for i in range(len(y)):
        k = np.arange(len(y)) != i
        pred[i] = a[i] @ np.linalg.lstsq(a[k], raw_sum[k], rcond=None)[0]
    return pred - raw_sum

def scores(e):
    return [float(np.sqrt(np.mean(e[s]**2))) for s in (axis,~axis)]

for name, a in [('affine',base[:,:5]),('diagonal cubic',base[:,:9]),('current sparse',base),('all paired products',np.c_[base,other]),('paired sum/difference cubics',np.c_[base,other,contrasts])]:
    print(name, 'LOO RMSE isolated/mixed', scores(loo(a)))
# Choose the rational denominator inside each outer fold by inner LOO.
qs = [(y[:,i]-y[:,j]) for i,j in pairs] + [y.sum(1)]
models = [base] + [base / (1 + a * q[:,None]**2) for q in qs for a in (.001,.01,.1,1.)]
rat, picked = [], []
for i in range(len(y)):
    k = np.arange(len(y)) != i
    def inner(a):
        mat = a[k]; u,s,v = np.linalg.svd(mat,full_matrices=False)
        u = u[:,s > 1e-12*s[0]]
        e = (raw_sum[k] - u @ (u.T @ raw_sum[k])) / np.maximum(1-(u*u).sum(1),1e-12)
        return max(np.sqrt(np.mean(e[f]**2)) for f in (axis[k],~axis[k]))
    j = min(range(len(models)),key=lambda j: inner(models[j]))
    a = models[j]; picked.append(j)
    rat.append(a[i] @ np.linalg.lstsq(a[k],raw_sum[k],rcond=None)[0]-raw_sum[i])
print('nested sparse versus rational selection LOO RMSE isolated/mixed', scores(np.asarray(rat)), 'nonbaseline picks', sum(j != 0 for j in picked))

b = m['_bridge'](c)
for name in ('train','test'):
    for key in ('past_channel_raw','hold_channel_raw'):
        raw = d[name][key].reshape(-1,4)
        decoded = m['_decode'](raw,b)
        e = m['_bridge_terms'](decoded) @ b - raw
        print(name,key,'forward closure max',float(np.max(abs(e))), 'channel range',float(decoded.min()),float(decoded.max()))
        assert np.max(abs(e)) < 1e-6

tr,te=d['train'],d['test']; labs=tr['lab_id']; x=m['_features'](tr,te)[:len(labs)]
past,hold=m['_loads'](tr,b); dt,t=d['past_dt'],d['eval_times']; obs=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; err=np.empty_like(rate); ce=np.empty_like(rate)
for l in np.unique(labs):
    h=labs==l; k=~h; tau,w=m['_fit'](x[k],past[k],hold[k],dt,t,obs[k],rate[k]); a,z=m['_design'](tau,x[h],past[h],hold[h],dt,t); err[h]=rate[h]-z@w; ce[h]=np.median(obs[h]-a@w,axis=1)
aux=tr['aux_chamber_telemetry']
response=np.column_stack((ce,err))
def within(a):
    a=a.copy()
    for l in np.unique(labs):
        h=labs==l; a[h]-=a[h].mean(0)
    return a
for label,a,f in [('pooled',aux,response),('within lab',within(aux),within(response))]:
    corr=np.corrcoef(a.T,f.T)[:4,4:]; print(label,'aux vs OOF curve/rate residual corr max',np.max(abs(corr),axis=0).round(4),'signed',corr.round(3).tolist())
