import sys,pickle,numpy as np
sys.path.insert(0,'/holdout_task')
import batch_solver as m
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; tr,te=D['train'],D['test']; c=D['calibration_load_cells']; y,x=c['bridge_raw'],c['true_channel_loads']; dt,t=D['past_dt'],D['eval_times']
p,good=m._fit_bridge(y,x)
score=np.linalg.norm(x,axis=1); order=np.argsort(score); groups=np.array_split(order,5)
for i,g in enumerate(groups):
    learn=np.ones(len(x),bool); learn[g]=False; p0,_=m._fit_bridge(y[learn],x[learn]); g=g[good[g]]
    e=m._load_from_raw(y[g],p0).sum(axis=1)-x[g].sum(axis=1)
    print('cal norm slice',i,len(g),'stress RMSE/max',np.sqrt(np.mean(e**2)),max(abs(e)))
a,b=m._features(tr,te); P,H,F=m._calibrated_inputs(tr,a,p); C=tr['strain_curves_obs']-tr['strain_0'][:,None]; R=tr['instant_rate_obs']
q,w=m._fit(P,H,F,dt,t,C,R); print('full fit',q,w)
for l in np.unique(tr['lab_id']):
    valid=tr['lab_id']==l; z=~valid
    Q,W=m._fit(P[z],H[z],F[z],dt,t,C[z],R[z]); A,B=m._design(Q,P[valid],H[valid],F[valid],dt,t); ec=abs(A@W-C[valid]);er=abs(B@W-R[valid]); print('held-out lab',l,'tau',Q,'curve med/90/max horizon median',np.median(ec),np.percentile(ec,90),max(np.median(ec,axis=0)),'rate med/90',np.median(er),np.percentile(er,90))
