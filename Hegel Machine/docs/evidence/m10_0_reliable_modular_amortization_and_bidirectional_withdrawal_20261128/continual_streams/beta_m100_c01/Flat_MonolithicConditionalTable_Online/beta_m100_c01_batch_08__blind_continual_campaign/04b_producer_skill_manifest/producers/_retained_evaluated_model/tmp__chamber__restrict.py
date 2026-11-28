from fitwall import *
base=np.array([1.0353,1.13614,1.07992]);
def expand(v):
 a,b,c=v
 return np.array([[a,b,0],[a,c,0],[1,0,0],[1,0,0]])
for hold in [-1,0,1,2]:
 use=np.arange(len(w))!=hold
 fit=least_squares(lambda v:(predict(expand(v))-obs)[use].ravel(),base,ftol=1e-12,xtol=1e-12,gtol=1e-12)
 pred=predict(expand(fit.x)); r=pred-obs
 print('restricted LOPO',hold, 'params', np.array2string(fit.x,precision=9),'pulse RMSE',np.sqrt(np.mean(r*r,axis=1)), 'shift RMSE max',np.sqrt(np.mean(r.reshape(len(w),len(s['mu_shifts']),-1)**2,axis=-1)).max())
for a in ('x','y'):
 k=s['kappa_'+a]; v=s['vel_'+a]; h=s['L'+a]/(len(k)-1)
 print('axis',a,'h',h,'k',k[[0,1,-2,-1]],'vel',v[[0,-1]],'robin',s.get('robin_'+a)); print('endpoint normalized inward FVM',[(k[0]+k[1])/(2*k[0])+max(-v[0],0)*h/k[0],(k[-1]+k[-2])/(2*k[-1])+max(v[-1],0)*h/k[-1]],'endpoint normalized periodic wrap',[1+max(v[0],0)*h/(.5*(k[0]+k[-1])),1+max(-v[-1],0)*h/(.5*(k[0]+k[-1]))])
d=json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')); print('latest rows excluding trace');
for row in d[-3:]:
 print({k:v for k,v in row.items() if k!='facility_telemetry_trace'}); tr=np.array(row['facility_telemetry_trace']); print('mean',tr.mean(0),'last',tr[-1],'min',tr.min(0),'max',tr.max(0))
print('public keys',list(s))
