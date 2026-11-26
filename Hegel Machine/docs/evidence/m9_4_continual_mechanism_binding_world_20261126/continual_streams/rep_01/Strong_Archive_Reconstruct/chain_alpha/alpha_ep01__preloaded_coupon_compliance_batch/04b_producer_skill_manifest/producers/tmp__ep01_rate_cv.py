import pickle
import numpy as np
from alpha_ep01_solver import _aligned,_physical,_Design,_fit,_mad,_crossing

p=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]
d,g=p['dataset'],p['gt']; tr,te=d['train'],d['test']; labs=tr['lab_id']; n=len(labs); t=d['eval_times']; x,c=_aligned(d)
a,b=(_Design(*_physical(d,q),d['past_dt'],t,f,q['initial_memory_states']) for q,f in [(tr,x[:n]),(te,x[n:])]); y=np.c_[tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']]
def rmse(e): return np.sqrt(np.mean(e*e))
for lab in np.unique(labs):
    keep=(labs!=lab)&c[:n]; logtau,w=_fit(a,y,keep); residual=y-a.matrix(logtau)@w
    center=np.median(residual[keep,-1]); rate=residual[:,-1]-center
    curve=np.median(residual[:, -6:-1]/t[-5:],axis=1)
    print('lab',lab,'rate noise MAD',round(_mad(rate[keep]),5),'curve drift noise MAD',round(_mad(curve[keep]),5))
    for sel in (labs==lab,(labs==lab)&c[:n],(labs==lab)&~c[:n]):
        if sel.any(): print(' slice',sel.sum(),'endpoint drift-proxy RMSE for rate gains 0,.25,.5,.75,1:',np.round([rmse(curve[sel]-k*rate[sel])*t[-1] for k in (0,.25,.5,.75,1)],5))
lt,w=_fit(a,y,c[:n]); base=b.matrix(lt)@w; rate=te['instant_rate_obs']-base[:,-1]; initial=te['strain_0']; crit=te['critical_strain']; denom=np.maximum(np.abs(b.hold),.5)
for cm,mi in [(0,0),(0,.5),(0,.75),(0,1),(.1,1),(.25,1),(.5,1),(1,1)]:
    curve=initial[:,None]+base[:,:-1]+rate[:,None]*np.where(c[n:],cm,mi)[:,None]*t
    sec=(curve[:,-1]-initial)/denom; crossing=_crossing(t,initial,curve,crit)
    print('public gains',cm,mi,'strain/secant/crossing RMSE',np.round([rmse(curve-g['te_clean']),rmse(sec-g['te_secant']),rmse(crossing-g['te_cross'])],6),'worst strain by lab',round(max(rmse((curve-g['te_clean'])[te['lab_id']==i]) for i in np.unique(te['lab_id'])),6))
