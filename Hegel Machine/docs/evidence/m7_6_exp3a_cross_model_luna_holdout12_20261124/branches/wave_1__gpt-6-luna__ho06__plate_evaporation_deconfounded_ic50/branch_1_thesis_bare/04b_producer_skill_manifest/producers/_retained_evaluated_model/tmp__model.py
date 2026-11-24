import pickle, numpy as np
from scipy.interpolate import Rbf
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb'))
for p in x['plates']:
 w=p['wells']; dm=[a for a in w if a['well_type']=='DMSO']; pk=[a for a in w if a['well_type']=='POS_KILL']
 # interpolate control surfaces separately with smooth thin plate spline
 def surf(arr):
  r=np.array([a['row'] for a in arr]); c=np.array([a['col'] for a in arr]); v=np.array([a['fluorescence'] for a in arr]);
  return Rbf(r,c,v,function='thin_plate',smooth=100)
 ds=surf(dm); ps=surf(pk)
 # fit compound response via local-control normalization
 groups={}
 for a in w:
  if a['well_type']=='COMPOUND':
   d=float(ds(a['row'],a['col'])); b=float(ps(a['row'],a['col'])); y=(a['fluorescence']-b)/(d-b)
   groups.setdefault(a['compound_id'],[]).append((a['log10_conc_uM'],y))
 for k,z in groups.items():
  arr=np.array(z); xx,yy=arr.T
  def f(q): return 1/(1+10**(q[1]*(xx-q[0])))
  res=least_squares(lambda q: f(q)-yy,[-1,1],bounds=([-4,.2],[3,3]),loss='soft_l1')
  gt=x['gt'][k]
  print(p['plate_id'],k, np.round(res.x,3), 'gt',gt)
