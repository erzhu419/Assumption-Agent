import pickle, runpy
import numpy as np
m = runpy.run_path('/holdout_task/batch_solver.py')
a = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]
d, g = a['dataset'], a['campaign_diagnostics']
tr, te, t, dt = d['train'], d['test'], d['eval_times'], d['past_dt']
B = m['_bridge'](d['calibration_load_cells'])
xtr, _ = m['_features'](tr, te)
S = m['_prepare'](tr, xtr, B)
strain = tr['strain_curves_obs']-tr['strain_0'][:,None]
rate = tr['instant_rate_obs']
tau, th = m['_fit'](S, dt, t, strain, rate)
p, h, f = m['_prepare'](g, m['_features'](g,g)[0], B)
q = f@th[2:]
y = np.c_[g['instant_rate_obs'], g['strain_curves_obs']-g['strain_0'][:,None]]
G = np.vstack((1/tau, -np.expm1(-t[:,None]/tau)))
T = np.r_[1.,t]
F = G*tau - T[:,None]
base = h[:,None]*(G@th[:2])+q[:,None]*T
w = np.r_[.45,np.ones(len(t))]
alt = {'reset':q[:,None,None]*F}
for key, values in [('chronological',p),('reversed',p[:,::-1])]:
 z = np.zeros((len(h),2))
 for s,step in enumerate(dt):
  z = np.exp(-step/tau)*z - np.expm1(-step/tau)*values[:,s,None]
 alt[key] = np.concatenate((alt['reset'], z[:,None,:]*G),axis=2)
contrast = p[:,-1]-p[:,0]
alt['pairwise contrast'] = np.concatenate((alt['reset'], contrast[:,None,None]*G),axis=2)
alt['stress-drift interaction'] = np.concatenate((alt['reset'], (q*(h-np.mean(h)))[:,None,None]*F),axis=2)
alt['free initial drift gain'] = np.concatenate((alt['reset'], (q[:,None]*T)[:,:,None]),axis=2)
rational = [q[:,None,None]*F/(1+alpha*q[:,None,None]**2) for alpha in (0,.01,.03,.1,.3,1.)]
def predict(A, keep):
 W = A[keep]*w[None,:,None]
 return base + A@np.linalg.lstsq(W.reshape(-1,W.shape[2]), ((y-base)[keep]*w).ravel(),rcond=None)[0]
def metrics(e): return np.sqrt(np.array([np.mean(e[:,0]**2),np.mean(e[:,1:]**2),np.mean(e[:,-1]**2)]))
for kind, labels in [('lab',g['lab_id']),('stratum',g['stratum']),('coupon',np.arange(len(g['lab_id'])))]:
 print(kind, 'worst then pooled RMSE: rate, curve, final')
 for name in [*alt,'rational gate (nested)']:
  E = np.empty_like(y); groups = []
  for group in np.unique(labels):
   mask = labels == group
   if name in alt: A = alt[name]
   else:
    loss = []
    for A in rational:
     sq = []
     for inner in np.unique(g['lab_id'][~mask]):
      inner_mask = (~mask)&(g['lab_id']==inner)
      sq.extend(np.ravel(((y-predict(A,~(mask|inner_mask)))[inner_mask]*w)**2))
     loss.append(np.mean(sq))
    A = rational[np.argmin(loss)]
   E[mask]=(y-predict(A,~mask))[mask]
   groups.append(metrics(E[mask]))
  print(name,np.round(np.max(groups,axis=0),5),np.round(metrics(E),5))
E=np.zeros((len(rate),len(t)+1))
for lab in np.unique(tr['lab_id']):
 j = tr['lab_id']==lab
 ts, bs = m['_fit'](tuple(s[~j] for s in S),dt,t,strain[~j],rate[~j])
 C,R = m['_design'](np.log(ts),tuple(s[j] for s in S),dt,t)
 E[j] = np.c_[rate[j]-R@bs,strain[j]-C@bs]
 print('material lab',lab,'tau relative shift',np.round(ts/tau-1,5),'median abs rate/curve',np.round([np.median(abs(E[j,0])),np.median(abs(E[j,1:]))],5))
