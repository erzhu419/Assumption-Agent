import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x=c['ref_rosette']; y=c['raw_rosette']; d=c['single_bank_sweeps']; z=d['ref_signal'].reshape(4,-1).T; w=d['raw_readout'].reshape(4,-1).T; good=np.max(abs(y),axis=1)<4; qa=abs(w)<4; folds=np.random.default_rng(20).permutation(len(x))%5
odd={'exp':lambda x,k:x*np.exp(-k*x*x), 'rat':lambda x,k:x/(1+k*x*x), 'rat2':lambda x,k:x/(1+k*x*x)**2,'rat3':lambda x,k:x/(1+k*x*x)**3,'tanh':lambda x,k:np.tanh(k*x), 'sin':lambda x,k:np.sin(k*x),'exp4':lambda x,k:x*np.exp(-k*x**4),'root':lambda x,k:x/np.sqrt(1+k*x*x)}
even={'exp':lambda x,k:-np.expm1(-k*x*x),'rat':lambda x,k:x*x/(1+k*x*x),'cos':lambda x,k:1-np.cos(k*x),'tanh':lambda x,k:np.tanh(k*x*x),'sq':lambda x,k:x*x,'soft':lambda x,k:np.log1p(k*x*x)}
def fit(q,od,ev,extra=None):
 def des(v,j,k,axis=False):
  a=v[:,j]; b=v[:,(j+2)%4]; cols=[np.ones(len(v)),a,odd[od](a,k[0])]
  if extra: cols += [odd[extra](a,k[2])]
  cols += [b,even[ev](b,k[1])]; out=np.column_stack(cols)
  if axis:out[:,-2:]=0
  return out
 def solve(k,ret=False):
  rr=[]; ms=[]
  for j in range(4):
   a=np.vstack((des(x[q],j,k),des(z[qa[:,j]],j,k,True))); t=np.r_[y[q,j],w[qa[:,j],j]]; m=np.linalg.lstsq(a,t,rcond=None)[0]; rr.extend(a@m-t); ms.append(m)
  return ms if ret else np.array(rr)
 starts=[(.8,.8,.3),(1.6,1.6,2.0)] if extra else [(.8,.8)]
 sols=[least_squares(solve,p,bounds=(.04,8),max_nfev=160) for p in starts]; k=min(sols,key=lambda r:r.cost).x; ms=solve(k,True)
 return lambda v:np.column_stack([des(v,j,k)@ms[j] for j in range(4)]),k,np.array(ms),np.sqrt(np.mean(solve(k)**2))
rows=[]
for od in odd:
 for ev in even:
  f,k,m,r=fit(good,od,ev); rows.append((r,od,ev,None))
for od,e in [('exp','exp'),('exp','rat'),('exp','tanh'),('exp','sin'),('exp','exp4'),('rat','tanh'),('rat','rat'),('tanh','tanh'),('tanh','sin')]:
 for ev in ('exp','rat','cos'):
  f,k,m,r=fit(good,od,ev,e); rows.append((r,od,ev,e))
for r,od,ev,e in sorted(rows,key=lambda r:r[0])[:12]:
 f,k,m,r=fit(good,od,ev,e); rs=[]
 for j in range(5):
  q=folds==j; g,*_=fit(good&~q,od,ev,e); v=np.linalg.norm(g(x[q&good])-y[q&good],axis=1); rs.append([np.median(v),v.max()])
 print(od,ev,e,'rmse',r,'cv',np.max(rs,axis=0),'k',k,'coeff',m.round(5).tolist(),flush=True)
print('pulses:',[(p.keys(),np.shape(p.get('y_pulse_raw'))) for p in (s.get('reference_tracer_pulses') or [])])
