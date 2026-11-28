exec(open('/tmp/chamber_probe.py').read().split("# Diagnostic reference")[0])
from scipy.optimize import least_squares
pc,hc=m._loads(c,b); y=c['strain_curves_obs']-c['strain_0'][:,None]; v=c['instant_rate_obs']; Y=np.c_[y,.35*v]; n=len(v)
def shape(taus,drift):
    return np.vstack((np.column_stack((-np.expm1(-t[:,None]/taus),t)) if drift else -np.expm1(-t[:,None]/taus),.35*(np.r_[1/taus,1] if drift else 1/taus)))
def fit(taus,drift):
    A=shape(taus,drift); co=Y@np.linalg.pinv(A).T; e=co@A.T-Y; return co,e
for drift in [False,True]:
    for free in [False,True]:
        tt=np.exp(least_squares(lambda p:fit(np.exp(p),drift)[1].ravel(),np.log(tau),bounds=(np.log(.07),np.log(25)),loss='soft_l1',f_scale=.004).x) if free else tau
        co,e=fit(tt,drift); print('shape',drift,free,tt,'curve/rate RMS',np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-1]**2))/.35,'column RMS',np.sqrt(np.mean(e**2,axis=0)).round(5));
        if not drift:
            xc=m._features(c,c)[:n]; ff=np.c_[np.ones(n),hc,pc,xc];
            print('co against loads'); print(np.c_[co,hc,pc][:7].round(4));
            for name,X in [('hold',ff[:,:2]),('all loads',ff[:,:8]),('features',ff),('baseline gaps',np.c_[np.ones(n),m._design(tau,xc,pc,hc,dt,t)[1][:,:2]*tau]),('hold and past norm',np.c_[ff[:,:2],pc,pc**2]),('saturated',np.c_[ff[:,:2],np.tanh(pc)])]:
                coef=np.linalg.lstsq(X,co,rcond=None)[0]; H=np.sum(X*np.linalg.pinv(X).T,axis=1); rr=(X@coef-co)/(1-H[:,None]); print(name,'LOOCV RMS',np.mean(rr**2,axis=0)**.5,'coef',coef.T.round(4));
print('telemetry summaries'); import json
for q in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
    z=np.array(q['facility_telemetry_trace']); print(q['completed_campaign_short_name'],'means',z.mean(0).round(3),'std',z.std(0).round(3))
