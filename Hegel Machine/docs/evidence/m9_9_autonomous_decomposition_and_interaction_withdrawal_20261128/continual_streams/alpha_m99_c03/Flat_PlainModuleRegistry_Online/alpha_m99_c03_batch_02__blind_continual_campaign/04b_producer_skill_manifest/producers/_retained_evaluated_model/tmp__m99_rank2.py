from m99_probe import *
from scipy.optimize import minimize_scalar
n=len(tr['lab_id']); bridge=m._calibration(cal); X=m._features(tr,te); ct=m._coupon_inputs(tr,X[:n],bridge); taus,theta=m._fit(ct,dt,t,tr['strain_curves_obs']-tr['strain_0'][:,None],tr['instant_rate_obs']); print('intrinsic',taus,theta,'eval times',t)
for e in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 a=np.asarray(e['facility_telemetry_trace']); print('apparatus', {k:str(v)[:220] for k,v in e.items() if k!='facility_telemetry_trace'},'mean',a.mean(0).round(5))
print('fields diagnostic',list(g),'test',list(te))
G=m._features(tr,g)[n:]; cg=m._coupon_inputs(g,G,bridge); P,H,D=cg; F,R=m._design(taus,cg,dt,t); drift=D@theta[2:]; E=np.c_[g['strain_curves_obs']-g['strain_0'][:,None]-F@theta,g['instant_rate_obs']-R@theta]; T=np.r_[t,1.];
def basis(u): return np.c_[np.r_[-np.expm1(-t/u),1/u],T]
def project(u):
 B=basis(u); q=np.linalg.lstsq(B,E.T,rcond=None)[0].T; return q,E-q@B.T
u=minimize_scalar(lambda u:np.sum(project(u)[1]**2),bounds=(.2,4),method='bounded').x; q,err=project(u); print('rank2 time',u,'RMSE horizon/rate',np.sqrt(np.mean(err**2,axis=0))); print('drift vs linear coefficient',np.polyfit(drift,q[:,1],1),'max linear drift residual',np.max(abs(q[:,1]-np.polyval(np.polyfit(drift,q[:,1],1),drift))));
raw=g['hold_channel_raw']; hist=g['past_channel_raw']; ch=lambda raw:raw@bridge[1:]+bridge[0]; hc=ch(raw); pc=ch(hist); gap=R[:,:2]*taus; Z={'drift':drift,'hold':H,'gap0':gap[:,0],'gap1':gap[:,1]};
for a in range(4):
 Z[f'x{a}']=G[:,a]; Z[f'h{a}']=hc[:,a]; Z[f'pmean{a}']=pc[:,:,a].mean(1); Z[f'plast{a}']=pc[:,-1,a]
 for b in range(a):
  Z[f'h{a}-h{b}']=hc[:,a]-hc[:,b]; Z[f'h{a}+h{b}']=hc[:,a]+hc[:,b]; Z[f'h{a}*h{b}']=hc[:,a]*hc[:,b]; Z[f'x{a}-x{b}']=G[:,a]-G[:,b]; Z[f'x{a}+x{b}']=G[:,a]+G[:,b]; Z[f'x{a}*x{b}']=G[:,a]*G[:,b]
for j in (0,1):
 r=q[:,j]-np.polyval(np.polyfit(drift,q[:,j],1),drift); print('coefficient',j,'drift fit',np.polyfit(drift,q[:,j],1),'residual std',r.std(),'top correlations',sorted([(abs(np.corrcoef(r,z)[0,1]),k) for k,z in Z.items()],reverse=True)[:12])
print('rows drift, exp amplitude, linear coefficient, hold channels\n',np.c_[drift,q,hc].round(4))
