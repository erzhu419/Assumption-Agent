import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/chamber98.py').read())
from scipy.optimize import least_squares
one=np.ones(len(H)); X=m._features(a,c)[n:]; drift=th[2]+X@th[3:]; forward=Z; reverse=P[:,::-1]@W
layouts={'hold':np.c_[one,H],'forward':np.c_[one,H,Z],'reverse':np.c_[one,H,reverse],'both':np.c_[one,H,Z,reverse],'all_history':np.c_[one,H,P],'forward_sweep':np.c_[one,H,Z,M[lab,0]*H,M[lab,0,None]*Z,M[lab,1]],'history_sweep':np.c_[one,H,P,M[lab,0]*H,M[lab,0,None]*P,M[lab,1]],'drift':np.c_[one,drift],'drift_x':np.c_[one,X]}
C=np.c_[coeff[:,:2]/th[:2],coeff[:,2]]
for name,F in layouts.items():
 Q=np.linalg.lstsq(F,C,rcond=None)[0]; e=[]
 for l in np.unique(lab):
  v=lab==l; e.append(np.mean((F[v]@np.linalg.lstsq(F[~v],C[~v],rcond=None)[0]-C[v])**2,0)**.5)
 print(name,'rank',np.linalg.matrix_rank(F),'fit',np.round(np.mean((F@Q-C)**2,0)**.5,5),'LOLO',np.round(np.max(e,0),5),'coeff',np.round(Q,4).T.tolist())
print('per coupon projected residuals:'); print(np.round(target-coeff@A.T,4))
print('past',np.round(P,3))
print('archive endpoints')
import json
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 f=np.array(row['facility_telemetry_trace']); print(np.round(f[[0,-1]],4).tolist())
# Profile free coupon amplitudes to test whether shared hold decay rates differ from the benchtop measurements.
def prof(q,rt=2):
 ta=np.exp(q); T=np.vstack([np.c_[-np.expm1(-t[:,None]/ta),t],np.r_[1/ta,1]*rt]); V=np.c_[Y,rt*R]; Q=np.linalg.lstsq(T,V.T,rcond=None)[0].T; return V-Q@T.T
for rt in [.3,1,2]:
 opt=least_squares(lambda q:prof(q,rt).ravel(),np.log(tau)); print('temporal',rt,'opt tau',np.exp(opt.x),'rms fixed/free',np.sqrt(np.mean(prof(np.log(tau),rt)**2)),np.sqrt(np.mean(opt.fun**2)))
