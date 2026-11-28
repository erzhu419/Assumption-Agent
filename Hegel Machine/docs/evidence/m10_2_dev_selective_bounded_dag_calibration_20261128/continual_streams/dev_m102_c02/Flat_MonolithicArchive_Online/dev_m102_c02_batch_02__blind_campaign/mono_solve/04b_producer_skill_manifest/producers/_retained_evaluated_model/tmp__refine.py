exec(open('/tmp/explore.py').read().split('ops={}')[0])
from scipy.optimize import least_squares
from scipy.special import erf
p=s['panel_b']; w=aff(z)@lstsq(aff(p['u']),p['v'])[0]; W=aff(w); P=w[:,:2]*w[:,2:]; ix=np.arange(52); jx=np.arange(52,72)
fs={'rational':lambda h:h**3/(1+h*h),'tanh':lambda h:h-np.tanh(h),'atan':lambda h:h-np.arctan(h),'erf':lambda h:h-np.sqrt(np.pi)/2*erf(h),'sqrt':lambda h:h-h/np.sqrt(1+h*h),'sinhinv':lambda h:h-np.arcsinh(h)}
rng=np.random.default_rng(780); splits=[(ix,jx)]+[(r[:52],r[52:]) for r in (rng.permutation(len(y)) for _ in range(16))]
for name,f in fs.items():
 for pair in [P,P.sum(axis=1,keepdims=True)]:
  for robust in [0,.008]:
   def fit(ii,jj):
    def design(q):return np.c_[W,pair,f(W@q)]
    def fun(q):
     A=design(q)[ii];return A@lstsq(A,y[ii])[0]-y[ii]
    a=lstsq(W[ii],y[ii])[0]; a[0]=0
    res=min([least_squares(fun,a*h,loss='soft_l1' if robust else 'linear',f_scale=robust or 1,max_nfev=120) for h in (.25,.55,.95)],key=lambda r:r.cost)
    A=design(res.x); b=lstsq(A[ii],y[ii])[0]; e=A@b-y
    return [np.mean(abs(e[ii])),np.mean(abs(e[jj])),max(abs(e[jj]))],res.x,b,e
   rr, q,b,e=fit(ix,jx); rr2=np.array([fit(ii,jj)[0] for ii,jj in splits[1:]]);print(name,pair.shape[1],robust,np.round(rr,5),'rand mean/max',np.round(rr2.mean(0),5),np.round(rr2.max(0),5),'params',q,b)
   if name=='rational' and pair.shape[1]==2 and not robust:
    I=np.argsort(abs(e))[-15:];print('largest residuals: row, observed, residual, projection',np.c_[I,y[I],e[I],(W@q)[I]])
T=s['eval_times'];print('traj bases')
for kind in ('free2','free3','linear1','linear2','telemetry'):
 def phi(q):
  if kind=='telemetry': return np.stack([-np.expm1(-r*T) for r in [.22,1.25]])
  F=-np.expm1(-np.exp(q)[:,None]*T)
  return np.vstack([T,F]) if kind.startswith('linear') else F
 A=w
 def prof(q):
  F=phi(q); return A@lstsq(A,t[ix]@np.linalg.pinv(F) if len(A)==len(ix) else t@np.linalg.pinv(F))[0]@F-t
 n=int(kind[-1]) if kind!='telemetry' else 0
 def run(ii,jj):
  def fit(q):
   F=phi(q); C=lstsq(A[ii],t[ii]@np.linalg.pinv(F))[0];return A@C@F-t,C
  q=least_squares(lambda q:fit(q)[0][ii].ravel(),np.log(np.geomspace(.35,1.5,n)),bounds=(-6,3),max_nfev=100).x if n else []
  e,c=fit(q);return np.mean(abs(e[ii])),np.mean(abs(e[jj])),np.exp(q),c
 r=run(ix,jx);print(kind,r,'rand worst',max(run(ii,jj)[1] for ii,jj in splits[1:]))
