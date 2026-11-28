import sys,glob,pickle,importlib.util
import numpy as np
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']
for i in [1,4]:
 p=glob.glob('/tmp/hegel_continual_archive/*_%02d_solver.py'%i)[0]; sp=importlib.util.spec_from_file_location('m',p); m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);globals()['m'+str(i)]=m
A=lambda x:np.c_[np.ones(len(x)),x]
F=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
x=tr['x']; qa=s['test']['x']; y=tr['y']; yt=tr['traj']; p=s['panel_a']; u,v=p['u'],p['v']; b,g=m1._calibrate(p)
print('cal diag residual sorted',np.sort(np.linalg.norm(v-(u*g+b),axis=1))); z=m4._calibration(p,np.r_[x,qa]); za=m4._calibration(p,v);print('cal nonlinear approximate delta',np.median(abs(z[:72]-(x-b)/g)),np.sort(np.linalg.norm(za-u,axis=1)))
np.set_printoptions(precision=4,suppress=True)
for k in ['panel_b','panel_c']:
 p=s[k];u=p['u'];print(k,'u ranges',u.min(0),u.max(0))
 for vn,v in p.items():
  if vn=='u':continue
  fs={'linear':(lambda u,p:u,[],[],[]),'power':(lambda u,p:np.sign(u)*abs(u)**p[0],[2],[.1],[7]),'powrat':(lambda u,p: np.sign(u)*abs(u)**p[0]/(1+p[1]*abs(u)**p[2]),[2,.22,2],[.1,0,.1],[7,20,16]), 'cubicrat':(lambda u,p:(u+p[0]*u**3)/(1+p[1]*u*u),[1,.22],[-20,0],[20,15]),'tanh':(lambda u,p:np.tanh(p[0]*u),[1],[.01],[20]),'sin':(lambda u,p:np.sin(p[0]*u),[1],[.01],[8]),'tanhmix':(lambda u,p:u+p[0]*np.tanh(p[1]*u),[-1,1],[-10,.01],[10,10]),'poly3':(lambda u,p:np.c_[u,u*u,u**3],[],[],[]),'ratmix':(lambda u,p:u+p[0]*u/(1+p[1]*u*u),[-1,.8],[-5,.001],[5,12])}
  for tag,(fn,ini,lo,hi) in fs.items():
   def err(p):
    h=A(fn(u,p));return (h@F(h,v)-v).ravel()
   op=least_squares(err,ini,bounds=(lo,hi),max_nfev=250,gtol=1e-12).x if ini else []
   print(vn,tag,op,'rms',np.std(err(op)))
  for kind in ['powered','cubic']:
   w=m1._panel_operator({'u':u,'v':v},kind);print(vn,kind,np.std(w(u)-v))
