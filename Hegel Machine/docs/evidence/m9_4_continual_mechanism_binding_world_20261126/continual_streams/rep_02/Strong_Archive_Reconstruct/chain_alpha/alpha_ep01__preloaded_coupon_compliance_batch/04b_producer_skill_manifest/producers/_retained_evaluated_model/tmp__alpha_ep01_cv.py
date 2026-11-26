import pickle, sys, numpy as np
sys.path.insert(0, '/holdout_task')
import alpha_ep01_solver as s
d=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']; train=d['train']; t=d['eval_times']
a,b,y,c=s._prepare(d); c=c[:len(y)]; labs=train['lab_id']; strata=train['stratum']
tau,w=s._fit(a,y,c); print('full fit',tau,w)
raw=np.empty_like(y)
for lab in np.unique(labs):
    fit=(labs!=lab)&c
    tau,w=s._fit(a,y,fit); print('hold lab',lab,'taus',tau,'coef',w)
    raw[labs==lab]=(a.matrix(np.log(tau))@w)[labs==lab]
r=y-raw
print('residual median/MAD by horizon',np.round(np.median(r,axis=0),4),np.round(s._mad(r),4))
for key, mask in [('complete',c),('missing',~c)]+[(f'lab{lab}',labs==lab) for lab in np.unique(labs)]+[(f'stratum{st}',strata==st) for st in np.unique(strata)]:
    z=r[mask]; print(key,len(z),'median rate/late',np.median(z[:,[-1,-2]],axis=0).round(4),'rmse at drift gains 0 .25 .5 1', [np.sqrt(np.mean((z[:,:-1]-g*z[:,-1,None]*t)**2)).round(5) for g in (0,.25,.5,1)])
