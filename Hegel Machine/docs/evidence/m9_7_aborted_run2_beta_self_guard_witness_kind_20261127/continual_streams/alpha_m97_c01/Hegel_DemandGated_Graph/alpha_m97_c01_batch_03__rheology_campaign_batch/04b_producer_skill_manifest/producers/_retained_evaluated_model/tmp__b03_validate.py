import importlib.util,json,pickle
import numpy as np
root='/holdout_task/'
spec=importlib.util.spec_from_file_location('b03',root+'alpha_m97_c01_batch_03_solver.py')
s=importlib.util.module_from_spec(spec); spec.loader.exec_module(s)
d=pickle.load(open(root+'public_alpha_m97_c01_batch_03.pkl','rb'))[0]['dataset']
a=d['train']; n=len(a['lab_id']); times=d['eval_times']; p,h,x,z=s._prepare(d)
m=s._Model(d['past_dt'],times,p[:n],h[:n],x[:n]); y=np.column_stack((a['strain_curves_obs']-z[:n,None],a['instant_rate_obs']))
rmse=lambda r: float(np.sqrt(np.nanmean(r*r)))
tau,b=s._fit(m,y); full=m.design(tau)@b-y
print('full fit',tau,b,'residual RMSE by time then rate',np.sqrt(np.nanmean(full**2,axis=0)).round(5),'max residual',np.nanmax(abs(full),axis=0).round(5))
r=np.empty_like(y)
for lab in np.unique(a['lab_id']):
 hold=a['lab_id']==lab
 t,c=s._fit(m,y,np.broadcast_to(~hold[:,None],y.shape)); r[hold]=(m.design(t)@c-y)[hold]
 print('LOLO lab',lab,'tau',t.round(5),'curve RMSE',round(rmse(r[hold,:-1]),6),'initial-rate RMSE',round(rmse(r[hold,-1]),6),'max weight shift',round(max(abs(c-b)),6))
missing=np.isnan(a['coupon_features_raw']).any(axis=1)
for name,sel in [('missing coupon',missing),('complete coupon',~missing),('negative hold',h[:n]<0),('nonnegative hold',h[:n]>=0)]:
 print(name,'n',sum(sel),'early/late/rate RMSE',*[round(rmse(q),6) for q in (r[sel,:3],r[sel,7:-1],r[sel,-1])])
for j in (0,len(times)-1,len(times)):
 mask=np.ones_like(y,dtype=bool);mask[:,j]=False
 t,c=s._fit(m,y,mask); print('withheld horizon',j,'RMSE',round(rmse((m.design(t)@c-y)[:,j]),6),'taus',t.round(5))
base=s._calibrate(d); print('in-batch calib',base.round(6))
for sweep in json.loads(s._HISTORY.read_text()):
 ll,xx,yy=(np.asarray(sweep[k]) for k in ('lab_id','cmd_stress','true_stress'))
 print('prefix',sweep.get('episode_short_name'),np.array([s._line(xx[ll==c],yy[ll==c]) for c in range(4)]).round(6))
