exec(open('/tmp/scalar.py').read().split('for name,u')[0]); from scipy.optimize import least_squares
from itertools import combinations
pairs=np.column_stack([w[:,i]*w[:,j]+w[:,k]*w[:,l] for i,j,k,l in ((0,2,1,3),(0,1,2,3),(0,3,1,2))]); pair=pairs[:,0];
links={'atan':np.arctan,'tanh':np.tanh,'sin':np.sin,'asinh':np.arcsinh,'sqrt':lambda q:q/np.sqrt(1+q*q),'rational':lambda q:q/(1+q*q),'cubic':lambda q:q-q**3/3,'erf':__import__('scipy.special',fromlist=['erf']).erf,'log':lambda q:np.sign(q)*np.log1p(abs(q))}
L=np.c_[np.ones(len(w)),w]; keep=np.arange(52); results=[]
for lname,f in links.items():
 for kind in ('inner','outer','both','splitden','denlinear','innerlinear','linklinear','denpair','anglepair','fixedbias','atan_pairden'):
  for j in range(3) if kind in ('inner','outer','both','denpair') else (0,):
   q=pairs[:,j]; F=np.c_[L,q]; c=fit(F)
   def pred(p):
    s=L@p[:5]; b=p[5]*q; amp=p[6]; off=p[7]; h=p[8] if len(p)>8 else 0
    if kind=='inner': return off+amp*f(s+b)
    if kind=='outer': return off+amp*f(s)+b
    if kind=='both': return off+amp*f(s+b)+h*q
    if kind=='splitden': return off+amp*(s+b)/(1+(s+h*q)**2)
    if kind=='denlinear': return off+amp*(s+b)/(1+s*s)
    if kind=='innerlinear': return off+amp*f(s+b)+h*(s+b)
    if kind=='linklinear': return off+amp*f(s)+h*s+b
    if kind=='denpair': return off+amp*s/(1+b*b)+h*q
    if kind=='anglepair': return off+amp*f(s)/(1+b*b)
    if kind=='fixedbias': return amp*f(s+b)
    if kind=='atan_pairden': return off+amp*f((s+b)/(1+h*h*q*q))
   best=None
   for sc in (.25,.5,.9):
    x0=np.r_[c*sc,1/sc,0,.15][:9 if kind in ('both','splitden','innerlinear','linklinear','denpair','atan_pairden') else 8]; m=least_squares(lambda p:(pred(p)-y)[:52],x0,max_nfev=280,ftol=2e-8); P=pred(m.x); rmse=np.sqrt(np.mean((P[:52]-y[:52])**2));
    if best is None or rmse<best[0]: best=(rmse,lname,kind,j,score(P),m.x)
   results.append(best)
for row in sorted(results,key=lambda r:r[0])[:36]: print(row)
print('trajectory sv against residual from archive')
t=runpy.run_path('/tmp/hegel_continual_archive/conf_m102_a03_batch_09_solver.py')['solve_batch']; o=t(dict(S,train={k:(v[:52] if k!='summary' else v) for k,v in S['train'].items()},test={'x':x[52:]})); print(np.mean(abs(o['state_pred']-T[52:]),axis=0))
print('calibration',a['_calibration'](S['panel_a'])); print('w range',w.min(0),w.max(0)); print('readout residual vs panel and trajectory')
