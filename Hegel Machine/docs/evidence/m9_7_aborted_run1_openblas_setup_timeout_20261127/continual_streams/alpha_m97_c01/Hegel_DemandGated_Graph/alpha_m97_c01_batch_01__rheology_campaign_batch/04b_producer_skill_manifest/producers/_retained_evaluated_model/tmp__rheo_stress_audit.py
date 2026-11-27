import importlib.util, pickle
import numpy as np
s=importlib.util.spec_from_file_location('solver','/holdout_task/alpha_m97_c01_batch_01_solver.py'); m=importlib.util.module_from_spec(s); s.loader.exec_module(m)
it=pickle.load(open('/holdout_task/public_alpha_m97_c01_batch_01.pkl','rb'))[0]; d=it['dataset']; tr=d['train']; lab=tr['lab_id']; n=len(lab); dt,t=d['past_dt'],d['eval_times']; data=tuple(a[:n] for a in m._prepare(d)); past,hold,x,act=data; y=tr['strain_curves_obs']-tr['strain_0'][:,None]; rate=tr['instant_rate_obs']; cc=np.isfinite(tr['coupon_features_raw']).all(1); L=np.log1p(t)
# Pairwise sums and differences are already in the four-feature linear span.
# Probe genuinely new terms: pairwise interactions, curvatures and saturated damage response.
inter=np.column_stack([x[:,i]*x[:,j] for i in range(4) for j in range(i,4)])
damage=np.maximum(np.abs(past)-d['crit_stress'],0)**2@dt
sat=np.column_stack((act*hold/(1+damage**2),act*hold/(1+hold**2)))
rows=[]
for k in np.unique(lab):
    keep=lab!=k; p=m._fit(tuple(a[keep] for a in data),dt,t,y[keep],rate[keep],'log'); a,b,*_=m._design(p[:2],data,dt,t,'log'); ey=y-m._dot(a,p[2:]); er=rate-m._dot(b,p[2:]); drift=np.median(ey/L,axis=1); fit=keep & cc
    print('held lab', k, 'signed complete implied drift/rate medians', np.round([np.median(drift[~keep&cc]),np.median(er[~keep&cc])],6))
    for name,u in [('baseline',np.zeros((n,1))),('coupon interactions',inter),('rational damage',sat)]:
        coef=np.linalg.lstsq(u[fit],drift[fit],rcond=None)[0]; corr=u@coef
        for sl,v in [('active complete',cc&(act>0)),('inactive complete',cc&(act==0)),('missing',~cc)]:
            sel=~keep&v
            rows.append((name,sl,np.median(abs(ey[sel]-corr[sel,None]*L)),np.median(abs(er[sel]-corr[sel]))))
for name in ('baseline','coupon interactions','rational damage'):
    for sl in ('active complete','inactive complete','missing'):
        z=np.array([(a,b) for nm,ss,a,b in rows if nm==name and ss==sl]); print(name,sl,'worst lab median strain/rate',np.round(z.max(axis=0),6))
