import sys, pickle,numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/tmp/hegel_continual_archive')
import dev_m102_c03_batch_03_solver as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; n=52
ls=lambda a,b: np.linalg.lstsq(a,b,rcond=1e-10)[0]
z=m._calibrate(s['panel_a'],s['train']['x']); u=s['panel_a']['u'];v=s['panel_a']['v']; print('cal inversion reference errors',np.sort(np.linalg.norm(m._calibrate(s['panel_a'],v)-u,axis=1)))
t=s['eval_times']; Y=s['train']['traj']; modes=lambda r:-np.expm1(-t[:,None]*r)
for q in [np.arange(n),np.arange(len(z))]:
 def fun(r):
  f=modes(r); return (Y[q].T-f@ls(f,Y[q].T)).ravel()
 r=least_squares(fun,[.22,2.15],bounds=([0,.4],[1.5,10])).x; print('time rates',r,'rmse',np.mean(fun(r)**2)**.5)
a=ls(modes(r),Y.T).T
for pan in ['panel_b','panel_c']:
 u=s[pan]['u'];
 for k,v in s[pan].items():
  if k=='u': continue
  print(pan,k)
  for nm,fn in [('aff',lambda x,r:x),('soft',lambda x,r:x/(1+r*abs(x))),('gauss',lambda x,r:np.exp(-(r*x)**2)),('tanh',lambda x,r:np.tanh(r*x)),('sin',lambda x,r:np.sin(r*x)),('cos',lambda x,r:np.cos(r*x)),('cyc',lambda x,r:x*np.roll(x,-1,axis=1)),('cube',lambda x,r:x**3)]:
   def des(x,r):return np.column_stack([m._aff(x),fn(x,r)])
   def fun(r):d=des(u,r);return (d@ls(d,v)-v).ravel()
   rr=least_squares(fun,[.8],bounds=(.08,4)).x
   d=des(u,rr); b=ls(d,v); print(nm,np.round(rr,4),round(np.mean(fun(rr)**2)**.5,6),end='; ')
  print()
c=m._branch(s['panel_c'],z,'v1','soft'); d=m._branch(s['panel_c'],z,'v2','gauss'); b=m._branch(s['panel_b'],z,'v','soft')
y=s['train']['y']; T=np.column_stack([y,a]); yz=np.column_stack([y,Y[:,-1]]); cross=lambda c,d:(c[:,:,None]*d[:,None,:]).reshape(len(c),-1)
for name,F in [('z',z),('c,d',np.c_[c,d]),('z,c,d',np.c_[z,c,d]),('z,c,d,diag',np.c_[z,c,d,c*d]),('z,c,d,dot',np.c_[z,c,d,(c*d).sum(1)]),('z,c,d,all',np.c_[z,c,d,cross(c,d)]),('z,c,d,b',np.c_[z,c,d,b]),('z,c,d,b,all',np.c_[z,c,d,b,cross(c,d)]),('z,c,d,all,zd',np.c_[z,c,d,cross(c,d),cross(z,d)])]:
 F=m._aff(F); pred=F@ls(F[:n],T[:n]); print(name,'train',np.round(abs(pred[:n]-T[:n]).mean(0),5),'test',np.round(abs(pred[n:]-T[n:]).mean(0),5),'traj',abs(pred[n:,1:]@modes(r).T-Y[n:]).mean())
