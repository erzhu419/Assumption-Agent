import pickle,numpy as np
import ep01_baseline as m
d=pickle.load(open('/holdout_task/public_alpha_ep01.pkl','rb'))[0]['dataset']; p=d['train']; des,_,y,c=m._prepare(d); t=d['eval_times']; labs=p['lab_id']; err=np.empty_like(y)
for lab in range(4):
 v=labs==lab; tau,b=m._fit(des,y,c&~v); err[v]=y[v]-des.matrix(np.log(tau))[v]@b
r=err[:,:-1]; slope=np.median(r[:,4:]/t[4:],axis=1); end=r[:,-1]; q=np.max(abs(r),axis=1); ix=np.argsort(q[c]); print('largest complete residuals: rank row, trajectory, rate residual');
for i in np.where(c)[0][ix[-10:]]: print(i,labs[i],np.round(err[i],3))
for mask in [c,~c]:
 print('n',mask.sum(),'large errors per horizon',(abs(r[mask])>.10).sum(0),'rate-slope stats',np.quantile(np.column_stack([slope[mask], err[mask,-1], slope[mask]-err[mask,-1]]),[.1,.5,.9],axis=0))
 for w in [0,.25,.5,.75,1]:
  rr=r[mask]-w*err[mask,-1,None]*t; print(w,'MAE',np.mean(abs(rr)), 'median',np.median(abs(rr)),'endpoint MAE',np.mean(abs(rr[:,-1])),'endpoint median',np.median(abs(rr[:,-1])))
print('near thresholds lab x missingness');
for l in range(4): print(l,[(int((mask & (labs==l)).sum()),np.round(np.quantile(abs(end[mask & (labs==l)]),[.5,.8,.95]),3)) for mask in [c,~c]])
