import importlib.util
import pickle
import numpy as np
root = '/holdout_task/'
spec = importlib.util.spec_from_file_location('solver', root+'alpha_m97_c02_batch_04_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
with open(root+'public_alpha_m97_c02_batch_04.pkl','rb') as f:
    d = pickle.load(f)[0]['dataset']
p, h, x, s0 = m._prepare(d)
tr = d['train']; n = len(tr['lab_id']); dt = d['past_dt']; t = d['eval_times']
p,h,x,s0 = p[:n],h[:n],x[:n],s0[:n]
a = m._activation(d,p,dt); y = tr['strain_curves_obs']-s0[:,None]; r = tr['instant_rate_obs']
def rms(x): return np.sqrt(np.mean(x*x))
def report(v,w): return [round(rms(v),7),round(rms(w),7)]
v, w = np.zeros_like(y), np.zeros_like(r)
for lab in np.unique(tr['lab_id']):
    q = tr['lab_id']==lab; f = ~q
    lt,b=m._fit(p[f],h[f],x[f],dt,t,a[f],y[f],r[f]); cy,cr=m._design(lt,p[q],h[q],x[q],dt,t,a[q]); v[q],w[q]=cy@b-y[q],cr@b-r[q]
    print('held lab',lab,'curve/rate RMSE',report(v[q],w[q]),'taus',np.exp(lt),'active',sum(a[q]>0))
for j in np.unique(tr['stratum']):
    q=tr['stratum']==j;print('OOF stratum',j,report(v[q],w[q]))
for label,q in [('zero',a==0),('nonzero',a>0),('strong',a>.8),('imputed',np.isnan(tr['coupon_features_raw']).any(axis=1))]:
    if q.any(): print('OOF',label,sum(q),report(v[q],w[q]))
lt,b=m._fit(p,h,x,dt,t,a,y,r);cy,cr=m._design(lt,p,h,x,dt,t,a)
print('full fit',np.exp(lt),b,'curve/rate RMSE',report(cy@b-y,cr@b-r)); print('OOF each horizon',np.sqrt(np.mean(v*v,axis=0)))
