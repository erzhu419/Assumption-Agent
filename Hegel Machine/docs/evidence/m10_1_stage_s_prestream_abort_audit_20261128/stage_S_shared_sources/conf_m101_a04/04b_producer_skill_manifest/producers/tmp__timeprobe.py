import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): exec(open('/tmp/probe.py').read())
from scipy.optimize import least_squares
P=fit(aff(T),y); print('y on traj',P,np.mean(abs(aff(T)@P-y)))
for k in ['panel_b','panel_c']:
 p=s[k];U=p['u']; D=poly(U,2)
 for j,V in p.items():
  if j=='u':continue
  B=fit(D,V); print(k,j,'quadratic coefficients',B,'res',np.std(D@B-V,0));print('resid rms for linear',np.std(aff(U)@fit(aff(U),V)-V,0))
for rank in range(1,5):
 u,d,v=np.linalg.svd(T,full_matrices=False);print('rank',rank,'MAE by time',np.mean(abs(T-(u[:,:rank]*d[:rank])@v[:rank]),0));print('right singular',v[rank-1])
for typ in ['exp','power','exp2','expt','exp3']:
 def D(p):
  e=[-np.expm1(-a*ts) for a in p];
  if typ=='power':return np.array([ts,ts**2,ts**3]).T
  if typ=='expt':return np.array(e+[ts]).T
  return np.array(e).T
 n={'exp':1,'power':1,'exp2':2,'expt':1,'exp3':3}[typ];r=least_squares(lambda p:(D(p)@fit(D(p),T.T)-T.T).ravel(),np.linspace(.3,2,n),bounds=(.005,15)); print('time basis',typ,r.x,np.mean(abs(r.fun)),np.std(r.fun))
