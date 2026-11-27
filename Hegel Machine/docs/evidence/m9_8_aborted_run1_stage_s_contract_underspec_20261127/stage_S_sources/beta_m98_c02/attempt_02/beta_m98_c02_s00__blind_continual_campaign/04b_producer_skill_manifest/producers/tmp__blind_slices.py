import sys, pickle, itertools
sys.path.insert(0,'/holdout_task')
import numpy as np
import batch_solver as b
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=b._Chamber(s); cal=b._calibration(s['sensor_calibration_sweeps']); y=cal(s['y_obs_raw']); E=c.edges; Z=s['Z_nuisance']; m=c.rhs.shape[1]; w0=np.full(m,s['total_budget']/m); Q=np.eye(m)[:,:-1]-np.eye(m)[:,[-1]]; base=E@c.nominal; diag=b._fit_boundary(c,s,y,cal); S0,U0=c.response(diag,c.rhs); ridge=s['nuisance_ridge']; Pz=np.linalg.solve(Z.T@Z+ridge*np.eye(Z.shape[1]),Z.T); fit=lambda S,v: b._quadratic(S-Z@(Pz@S),v-Z@(Pz@v),s['source_ridge'],s['total_budget'],s['u_box'])['weights']; rms=lambda v: np.sqrt(np.mean(v*v));
D0=np.c_[S0@Q,Z]; rr=y-S0@w0; noise=rr-D0@np.linalg.lstsq(D0,rr,rcond=None)[0];
cases=[('public',diag,y)]
patterns=[[0,0,0,0],[.7,.7,-.5,-.5],[.7,-.7,-.5,.5],[1.4,-.8,.4,1.0],[-.3,-.65,.75,-.65]]
for i,p in enumerate(patterns):
 d=base+E@p+c.slopes@(np.array([.20,-.07,.15,0.]) if i%2 else np.zeros(4)); S,_=c.response(d,c.rhs)
 for no in [0,1]:
  w=np.full(m,s['total_budget']/m) if i%2 else np.array([.42,.24,0,.03,.42,.14,.08,.07]); w*=s['total_budget']/w.sum(); cases.append((f'{i}/noise{no}',d,S@w+Z@np.array([.005,-.01,.015])+no*noise))
for label,basis in [('axes',E@np.repeat(np.eye(2),2,axis=0)),('walls',E),('walls+slopes',np.c_[E,c.slopes])]:
 for pen in [.0015,.005,.010,.015,.025,np.inf]:
  vals=[]
  for tag,d,v in cases:
   def fun(p):
    S,_=c.response(base+basis@p,c.rhs); D=np.c_[S@Q,Z]; r=v-S@w0
    return np.r_[r-D@np.linalg.lstsq(D,r,rcond=None)[0],pen*p]
   p=np.zeros(basis.shape[1]) if not np.isfinite(pen) else least_squares(fun,np.zeros(basis.shape[1]),bounds=(-2.5,3),gtol=1e-10).x
   S,U=c.response(base+basis@p,c.rhs); St,Ut=c.response(d,c.rhs); w,wt=fit(S,v),fit(St,v); vals.append((np.linalg.norm(U@w-Ut@wt)/np.linalg.norm(Ut@wt),np.linalg.norm(w-wt),tag))
  print(label,pen,'public %.4g/%.4g'%vals[0][:2], 'worst clean',max(vals[1::2]),'worst noisy',max(vals[2::2]))
