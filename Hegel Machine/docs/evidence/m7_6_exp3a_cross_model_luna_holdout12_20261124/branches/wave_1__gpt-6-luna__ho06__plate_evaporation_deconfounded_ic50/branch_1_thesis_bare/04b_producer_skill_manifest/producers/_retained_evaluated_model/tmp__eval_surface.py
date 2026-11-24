import pickle, numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb'))
def des(items):
 r=np.array([w['row'] for w in items],float)/15;c=np.array([w['col'] for w in items],float)/23
 return np.column_stack((np.ones(len(r)),r,c,r*r,c*c,r*c))
obs={}
for p in x['plates']:
 ws=p['wells']; dm=[w for w in ws if w['well_type']=='DMSO']; kill=[w for w in ws if w['well_type']=='POS_KILL']
 b=np.median([w['fluorescence'] for w in kill]); X=des(dm); y=np.array([w['fluorescence'] for w in dm])-b
 coef=np.linalg.lstsq(X,y,rcond=None)[0]
 for w in ws:
  if w['well_type']=='COMPOUND':
   g=des([w])[0]@coef
   obs.setdefault(w['compound_id'],[]).append((w['log10_conc_uM'],(w['fluorescence']-b)/g))
res={}
for cid,z in obs.items():
 ar=np.array(z); c,v=ar.T
 def f(q): return 1/(1+np.exp(np.clip(q[1]*(c-q[0])*np.log(10),-35,35)))
 q=least_squares(lambda q:f(q)-v,[-.5,1],bounds=([-4,.25],[3,3]),loss='soft_l1',f_scale=.04).x; res[cid]=q
errs=[(abs(res[k][0]-v['log10_ic50_uM']),abs(res[k][1]-v['hill_slope'])) for k,v in x['gt'].items()]
print('max errors',np.max(errs,axis=0)); print({k:np.round(v,3).tolist() for k,v in res.items()})
