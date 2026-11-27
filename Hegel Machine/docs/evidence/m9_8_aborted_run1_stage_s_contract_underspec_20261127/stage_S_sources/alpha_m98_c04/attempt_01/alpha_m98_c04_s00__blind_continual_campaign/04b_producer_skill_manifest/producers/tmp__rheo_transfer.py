exec(open('/tmp/rheo_probe.py').read().split('for kind in')[0])
from scipy.optimize import least_squares

def predict(g,z,tau,w): return s._design(g,z,b,dt,t,tau)@w

def select(g,idx): return {k:v[idx] for k,v in g.items()}

def fit_projected(g,z):
 y=s._obs(g); h=np.r_[t,1.]; pair=np.arange(len(t)-1)
 def unpack(q): return np.exp(q[:2]),q[2:4]
 def fun(q):
  tau,w=unpack(q); r=y-s._design(g,z,b,dt,t,tau)[:,:,:2]@w
  # Adjacent normalized contrasts eliminate each coupon's unknown drift.
  rc=(r[:,pair+1]*t[pair]-r[:,pair]*t[pair+1])/np.hypot(t[pair],t[pair+1]); rate=(r[:,-1]*t[0]-r[:,0])/np.hypot(t[0],1)
  return np.column_stack((rc,rate)).ravel()
 q=least_squares(fun,np.r_[np.log([.65,5]),.8,.8],bounds=([np.log(.05001)]*2+[-10]*2,[np.log(24.999)]*2+[10]*2),loss='cauchy',f_scale=.012,max_nfev=150).x
 return unpack(q)

def drift_obs(g,z,tau,w):
 y=s._obs(g)-s._design(g,z,b,dt,t,tau)[:,:,:2]@w
 return np.median(np.column_stack((y[:,:-1]/t,y[:,-1])),axis=1)

bt,bw,_=s._fit(tr,x[0],b,dt,t)
for nam,g,z in [('bench',tr,x[0]),('pilot',pi,x[2])]:
 tau,w=fit_projected(g,z); do=drift_obs(g,z,tau,w); base=bw[2]+z@bw[3:]; print(nam,'projected tau/weight',tau,w)
 if nam=='pilot':
  for l,u,d,v,zz in zip(g['lab_id'],g['stratum'],do,base,z): print(l,u,'observed, bench, diff, ratio',np.round([d,v,d-v,d/v],4),zz.round(2))
print('test cell counts',[(l,u,int(np.sum((te['lab_id']==l)&(te['stratum']==u)))) for l in np.unique(te['lab_id']) for u in np.unique(te['stratum'])]); print('telemetry train/test summaries:')
for g in [tr,te]:
 a=g['aux_chamber_telemetry'];print('global',a.mean(0).round(3),a.std(0).round(3),'first',a[:4].round(3));print('labs',[(int(l),a[g['lab_id']==l].mean(0).round(3)) for l in np.unique(g['lab_id'])]);print('strata',[(int(l),a[g['stratum']==l].mean(0).round(3)) for l in np.unique(g['stratum'])])
