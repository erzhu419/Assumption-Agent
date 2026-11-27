import contextlib,io,runpy,numpy as np,json
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): v=runpy.run_path('/tmp/chamber_structure.py')
p,l,a,h,t,dt,Y,g,tr,te,align=(v[k] for k in ('p','l','a','h','t','dt','Y','g','tr','te','align'))
np.set_printoptions(precision=6,suppress=True,linewidth=155)
X=align([g]); r=h[2]+X@h[3:]; n=len(l); M=np.isnan(g['coupon_features_raw']); lab=np.eye(4)[g['lab_id']][:,1:]; q=np.arange(n); B=np.c_[np.ones(n),r,lab,M[:,M.any(0)]]; lin=np.r_[t,1.];
def f(q):return np.r_[-np.expm1(-t/q),1/q]
def m(q):return np.exp(-np.r_[np.cumsum(dt[:0:-1])[::-1],0]/q)*-np.expm1(-dt/q)
P=np.c_[l,p]; D=np.c_[P,B]; C=np.linalg.lstsq(D,Y,rcond=None)[0];print('Full load coefficients');print(C[:7]); print('Feature coefficient',C[8]); print('lab missing coeffs last/rate',C[9:,[-2,-1]])
outer=lambda x,y:x[:,None]*y

def fit(mode, ids):
 def calc(u,out=False):
  z=np.exp(u); F,G=map(f,z[:2]); mem=z[2:4] if 'separate' in mode else z[:2]; H,J=(l-p@m(k) for k in mem); cols=[outer(H,F),outer(J,G),outer(l,lin)];
  if 'memoryfree' in mode:cols=[outer(x,w) for x in P.T for w in [F,G]]+[outer(l,lin)];
  if 'holdfree' in mode:cols.extend([outer(l,F),outer(l,G)])
  if 'featurefree' in mode: cols.extend(outer(x,w) for x in B.T for w in (F,G,lin));
  elif 'fast' in mode: cols.extend(outer(x,F) for x in B.T)
  else:cols.extend(outer(x,G) for x in B.T)
  A=np.stack(cols,axis=-1); c=np.linalg.lstsq(A[ids].reshape(-1,A.shape[-1]),Y[ids].ravel(),rcond=1e-10)[0]; e=A@c-Y; return (z,c,e) if out else e[ids].ravel()
 starts=[[.65,1.45],[.7,4.5],[.55,1.05]]; fits=[least_squares(calc,np.log(z+(list(a) if 'separate' in mode else [])),bounds=(np.log(.08),np.log(25)),max_nfev=125) for z in starts]; z=min(fits,key=lambda s:s.cost);return calc(z.x,True)
for mode in ('shared','shared-fast','shared-featurefree','separate','separate-holdfree','separate-featurefree','memoryfree','memoryfree-featurefree'):
 u,c,e=fit(mode,q); E=np.array([fit(mode,q[q!=i])[2][i] for i in q]); clean=~M.any(1); print(mode,'taus',u,'coeff',c[:min(len(c),15)],'train',np.mean(e**2)**.5,'clean loo',np.max(np.mean(E[clean]**2,axis=0)**.5),'clean rate',np.mean(E[clean,-1]**2)**.5,'all loo',np.max(np.mean(E**2,axis=0)**.5));
 if mode=='memoryfree':print('temporal load coeff',c[:14].reshape(-1,2))
for row in json.load(open('/tmp/hegel_continual_archive/archive_apparatus_log.json')):
 w=np.array(row['facility_telemetry_trace']); print('log',list(row),{k:v for k,v in row.items() if k!='facility_telemetry_trace'}, w.mean(0), w.std(0))
