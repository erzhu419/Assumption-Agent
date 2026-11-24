exec(open('/tmp/probe.py').read().split('for k in gt:')[0])
from scipy.optimize import least_squares
# diagnostics per compound, direct robust non-linear fits, compare dropping residual outliers and Huber losses
for k in gt:
 y=np.array(allv[k]); c=np.array(allc[k]); ok=np.isfinite(y)&(y>-0.3)&(y<1.5)
 for lo,hi in [(-.3,1.5),(0,1),(0.03,.97)]:
  use=ok&(y>lo)&(y<hi)
  def f(q): return np.clip(y[use],0,1)-1/(1+10**(q[1]*(c[use]-q[0])))
  r=least_squares(f,[-.3,1.1],bounds=([-4,.2],[3,3]),loss='soft_l1',f_scale=.04)
  print(k,(lo,hi),len(c[use]),np.round(r.x,3),np.round(r.x-[gt[k]['log10_ic50_uM'],gt[k]['hill_slope']],3))
