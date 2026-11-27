import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import alpha_m97_c02_s00_solver as s
D=pickle.load(open('/holdout_task/public_alpha_m97_c02_s00.pkl','rb'))[0]['dataset']; T=D['train']; n=len(T['lab_id']); dt,t=D['past_dt'],D['eval_times']; p,h=s._physical(T,s._calibration(D)); a=s._align(D); X=np.c_[np.ones(len(a)),a][:n]; y=T['strain_curves_obs']-T['strain_0'][:,None]; v=T['instant_rate_obs']; labs=T['lab_id']; miss=~np.isfinite(T['coupon_features_raw']); missing=miss.any(axis=1)

def fit(mask):
 u,w=s._fit(p[mask],h[mask],dt,t,X[mask],y[mask],v[mask]); A,B=s._design(u,p,h,dt,t,X); return np.exp(u),w,A@w-y,B@w-v

def summarize(r,q,mask):
 return np.round(np.r_[np.median(abs(r[mask])),np.max(np.median(abs(r[mask]),axis=0)),np.median(abs(q[mask]))],6)
for l in range(4):
 mask=labs==l; u,w,r,q=fit(~mask); print('leave lab',l,'taus',np.round(u,5),'all',summarize(r,q,mask),'complete',summarize(r,q,mask&~missing),'missing n',sum(mask&missing));
 if (mask&missing).any():print('  missing',summarize(r,q,mask&missing))
u,w,r,q=fit(~missing); print('complete-case coefficients',u,w); print('complete in sample',summarize(r,q,~missing)); print('missing case detail: lab,stratum,missing cols,observed-minus-pred row drift and rate');
for i in np.flatnonzero(missing):print(labs[i],T['stratum'][i],np.flatnonzero(miss[i]),round(-np.median(r[i]/t),5),round(-q[i],5))
print('cell counts',np.unique(np.c_[labs,T['stratum']],return_counts=True,axis=0))
