import numpy as np,pickle,runpy,itertools
from scipy.optimize import least_squares
from scipy.special import erf
m=runpy.run_path('/holdout_task/node_00.py'); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; y=s['train']['y']; z=m['_sensor'](s['panel_a'],s['train']['x']); n=len(y)
forms=[np.sin,np.tanh,np.arctan,erf,lambda x:x/(1+x*x)]
def fit(A,y):return np.linalg.lstsq(A,y,rcond=1e-9)[0]
def design(z,F,shift,typ):
 p=np.roll(z,shift,axis=1); g=np.roll(F,shift,axis=1); L=np.c_[np.ones(len(z)),z,F]
 inter=[z*g,z*p,F*g,z*z,F*F,z*F,z*g*g,z*z*g]
 return np.column_stack([L]+[inter[j] for j in typ])
records=[]
for j,fun in enumerate(forms):
 for shift in (1,-1,2):
  for typ in [(0,),(0,1),(0,2),(0,3),(0,4),(0,5),(0,1,2),(0,1,3),(0,3,4),(0,1,2,3),(0,3,5),(0,6),(0,7),(0,1,5),(0,1,6),(0,1,7),(0,1,3,5),(0,1,2,4),(0,1,3,4,5)]:
   def A(z,h):return design(z,fun(z*h),shift,typ)
   zt=z[:52]; yt=y[:52]
   def res(h):
    D=A(zt,h);return D@fit(D,yt)-yt
   for ini in ([2.,3.] if j==0 else [1.,2.]):
    rr=least_squares(res,[ini],bounds=(.18,5),max_nfev=40); D=A(z,rr.x); b=fit(D[:52],yt); e=D@b-y
    H=np.einsum('ij,ji->i',D[:52],np.linalg.pinv(D[:52])); loo=np.mean(abs(e[:52]/(1-H)))
    records.append((loo,np.mean(abs(e[:52])),np.mean(abs(e[52:])),j,shift,typ,float(rr.x[0]),np.round(b,3)))
for r in sorted(records)[:20]:print(r)
print('Calibrated training range',np.min(z,axis=0),np.max(z,axis=0))
for j,f in enumerate(forms):
 u=s['panel_b']['u']; v=s['panel_b']['v'];
 for shift in [1,-1]:
  for typ in [(0,), (1,), (3,), (0,1,3)]:
   def des(u,h):return design(u,f(h*u),shift,typ)
   def res(h):
    D=des(u,h);return (D@fit(D,v)-v).ravel()
   e=min([least_squares(res,[i],bounds=(.1,5),max_nfev=60) for i in (1.,2.,3.)],key=lambda a:a.cost); D=des(u,e.x); b=fit(D,v); print('panel B',j,shift,typ,round(e.x[0],4),round(np.sqrt(np.mean(e.fun**2)),6));
   if np.sqrt(np.mean(e.fun**2))<.0015:print(np.round(b,3))
