import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline,PchipInterpolator
D=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['dataset']; c=D['calibration_load_cells']; x=c['true_channel_loads']; y=c['bridge_raw']; s=c['single_channel_sweeps']; good=np.max(abs(y),axis=1)<8; axis=(x!=0).sum(1)==1

def terms(t,q,k):
 u=q[0]*t; a=np.tanh(u); v=q[-1]*t
 extra={'cubic':[t**3], 'cubrat':[t**3/(1+v*v)], 'rat':[t/(1+v*v)], 'sin':[np.sin(v)], 'atan':[np.arctan(v)], 'sqrt':[v/np.sqrt(1+v*v)], 'softsign':[t/(1+abs(v))], 'damped':[t*np.exp(-v*v)], 'ratlin':[t,t/(1+v*v)], 'sinlin':[t,np.sin(v)], 'positive_tail':[t,t**3/(1+v*v)]}[k]
 return np.column_stack([np.ones_like(t),a,*extra])

def data(j):
 m=(s['lab_id']==j)&(abs(s['cmd_stress'])<8); a=axis&good&(x[:,j]!=0)&(x[:,j]!=-1.6); return np.r_[s['true_stress'][m],x[a,j]], np.r_[s['cmd_stress'][m],y[a,j]]

def fit(t,z,k):
 def co(q): return np.linalg.lstsq(terms(t,q,k),z,rcond=None)[0]
 def residual(q):return terms(t,q,k)@co(q)-z
 p=min((least_squares(residual,q,bounds=(.1,5),max_nfev=180) for q in ([[1,1],[2,1],[1,2]] if k!='cubic' else [[1]])),key=lambda r:r.cost).x
 return p,co(p)
for k in ('cubic','cubrat','rat','sin','atan','sqrt','softsign','damped','ratlin','sinlin','positive_tail'):
 stats=[]; ps=[]
 for j in range(4):
  t,z=data(j); p,b=fit(t,z,k); err=terms(t,p,k)@b-z; ps.append(np.r_[p,b]); h=[]
  for mag in np.unique(np.round(abs(t),7)):
   m=np.round(abs(t),7)!=mag; q,a=fit(t[m],z[m],k); h.extend(terms(t[~m],q,k)@a-z[~m])
  stats.append([np.sqrt(np.mean(err**2)),max(abs(np.array(h)))])
 print(k,np.round(stats,5).tolist(),np.round(ps,4).tolist())
for kind in ('spline','cubic','cubrat','rat','damped','sin','positive_tail'):
 diag=[]
 for j in range(4):
  t,z=data(j); i=np.argsort(t)
  if kind=='spline': v=CubicSpline(t[i],z[i])(x[:,j]); diag.append(v); continue
  p,b=fit(t,z,kind); diag.append(terms(x[:,j],p,kind)@b)
 r=y-np.array(diag).T
 maxes=[]
 for j in range(4):
  a=(j+2)%4; T=np.column_stack([x[:,a],x[:,a]**2]); b=np.linalg.lstsq(T[good],r[good,j],rcond=None)[0]; res=(r[:,j]-T@b)[good]; maxes.append([np.sqrt(np.mean(res**2)),max(abs(res))]);
 print('coupled',kind,np.round(maxes,6).tolist())
