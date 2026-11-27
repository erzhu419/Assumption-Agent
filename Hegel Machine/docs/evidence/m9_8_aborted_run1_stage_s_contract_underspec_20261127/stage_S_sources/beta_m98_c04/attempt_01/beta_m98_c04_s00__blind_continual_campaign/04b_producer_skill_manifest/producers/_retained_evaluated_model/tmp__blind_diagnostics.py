import pickle, sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import batch_solver as b
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=b._calibrator(s['sensor_calibration_sweeps']); ch=b._Chamber(s); y=c(s['y_obs_raw']); wall=b._fit_walls(ch,s,c,y); S=ch.response(wall)[0]; Z=s['Z_nuisance']; T=s['aux_manifold_telemetry']; ij=s['sensor_ij']; ns=len(ij); N=len(y); W=np.column_stack([p['w_pulse'] for p in s['reference_tracer_pulses']]); Y=np.column_stack([p['y_pulse_raw'] for p in s['reference_tracer_pulses']]); X=np.c_[S,Z]; e=y-X@np.linalg.lstsq(X,y,rcond=None)[0]; E=e.reshape(-1,ns); print('residual-shift correlation',np.corrcoef(E)); print('residual RMS by bank',np.sqrt(np.mean(e.reshape(-1,4)**2,axis=0))); print('structured telemetry correlations',np.corrcoef(np.c_[e,T.ravel(),np.tile(T,4).ravel()[:N]].T)[0]); print('table sensor x,y, shift residual'); print(np.round(np.c_[ij,E.T],5)); print('item keys',list(pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]));
P=b._projector(Z,s['nuisance_ridge']); upper=np.broadcast_to(s['u_box'], S.shape[1]); M=len(upper)
def q(wall):
 m,u,_=ch.response(wall); x=P(m); return b._box_qp(x.T@x+s['source_ridge']*np.eye(M),-x.T@P(y),upper,s['total_budget'])[0],u
wr,ur=q(wall); rms=lambda a:np.sqrt(np.mean(a*a)); shifts=np.arange(N)//ns; Q=np.linalg.qr(Z)[0];
for strength in (0.,.003,.01,.025,.07,.2,1.,np.inf):
 scale=np.r_[ch.prior[:4],ch.scales]; lo=np.r_[np.full(4,.02),-ch.scales*.65]; hi=np.r_[np.maximum(20,ch.prior[:4]*4),-ch.inward*.65]; oof=np.empty(N)
 def fit(keep):
  zz=Z[keep]; proj=b._projector(zz,0.); yy=proj(y[keep]);
  def profile(w):
   x=proj(ch.response(w)[0][keep]); a=b._box_qp(x.T@x+1e-10*np.eye(M),-x.T@yy,upper,s['total_budget'])[0]; return x@a-yy,a
  w=ch.prior if np.isinf(strength) else least_squares(lambda w:np.r_[profile(w)[0],strength*(w-ch.prior)/scale],ch.prior,bounds=(lo,hi),x_scale=scale,max_nfev=42).x
  a=profile(w)[1]; m=ch.response(w)[0]; g=np.linalg.lstsq(zz,y[keep]-m[keep]@a,rcond=None)[0]; return w,m@a+Z@g
 w,_=fit(np.ones(N,dtype=bool)); a,u=q(w)
 for j in range(len(ch.shifts)):
  test=shifts==j; _,pred=fit(~test); oof[test]=pred[test]-y[test]
 print(strength,'held observation',max(rms(r) for r in np.split(oof,len(ch.shifts))),'held pulse',max(rms(r) for r in np.split(ch.response(w,ch.F@W)[0]-Y,len(ch.shifts))),'relative weights and field',np.linalg.norm(a-wr)/np.linalg.norm(wr),np.linalg.norm(u@a-ur@wr)/np.linalg.norm(ur@wr),'walls',np.round(w,3))
