import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=4,suppress=True)
def I(x):return np.c_[np.ones(len(x)),x]
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def pair(u):return np.column_stack([u[:,i]*u[:,j] for i in range(4) for j in range(i+1,4)])
print('T,telemetry',s['eval_times'],s['telemetry_vec'])
for k,vk in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[k]['u'];v=s[k][vk]; mu=I(u); print(k,vk,'std',u.std(0),v.std(0));
 for key,f in [('linear',lambda x:x),('poly2',lambda x:np.c_[x,x*x,pair(x)]),('univ3',lambda x:np.c_[x,x*x,x**3]),('univ5',lambda x:np.c_[x,x*x,x**3,x**4,x**5]),('roll2',lambda x:np.c_[x,x*x,x*np.roll(x,1,axis=1),x*np.roll(x,2,axis=1)])]:
  d=I(f(u)); mask=np.ones(len(u),bool)
  for _ in range(7 if k=='panel_a' else 1):
   c=ls(d[mask],v[mask]); r=np.sqrt(np.mean((d@c-v)**2,axis=1)); h=np.median(r[mask]); new=r<max(.02,4*h)
   if new.sum()<=d.shape[1] or np.array_equal(mask,new):break
   mask=new
  err=d@c-v; print(key,mask.sum(),np.sqrt(np.mean(err[mask]**2,axis=0)),np.median(r),np.max(r));
  if key in ('linear','univ3'): print(c.T)
t=s['eval_times']; tr=s['train']['traj'];
for f,p in [(lambda a:np.c_[-np.expm1(-a[0]*t),-np.expm1(-a[1]*t)],[1,4]),(lambda a:np.c_[-np.expm1(-a[0]*t),np.sin(a[1]*t)],[1,1]),(lambda a:np.c_[-np.expm1(-a[0]*t),t],[1]),(lambda a:np.c_[-np.expm1(-a[0]*t),np.exp(-a[0]*t)*t],[1])]:
 r=least_squares(lambda a:(f(a)@ls(f(a),tr.T)-tr.T).ravel(),p);print('time',r.x,np.sqrt(2*r.cost/tr.size))
