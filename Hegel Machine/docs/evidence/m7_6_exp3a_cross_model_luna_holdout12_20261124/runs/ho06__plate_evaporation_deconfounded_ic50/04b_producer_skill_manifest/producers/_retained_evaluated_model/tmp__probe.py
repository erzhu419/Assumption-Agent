import pickle, numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb')); gt=x['gt']
# Fit a smooth quadratic surface to DMSO controls, subtract plate background from robust POS median.
def surf(r,c):
 r=np.asarray(r)/15*2-1; c=np.asarray(c)/23*2-1
 return np.column_stack([np.ones(len(r)),r,c,r*r,c*c,r*c])
allv={k:[] for k in gt}; allc={k:[] for k in gt}
for p in x['plates']:
 w=p['wells']; pos=[z for z in w if z['well_type']=='POS_KILL']; bg=np.median([z['fluorescence'] for z in pos])
 dm=[z for z in w if z['well_type']=='DMSO']; A=surf([z['row'] for z in dm],[z['col'] for z in dm]); y=np.array([z['fluorescence']-bg for z in dm]);
 # robust-ish fit by iterative residual clipping
 keep=np.ones(len(y),bool)
 for _ in range(4):
  coef=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; resid=y-A@coef; med=np.median(resid); mad=1.4826*np.median(abs(resid-med))+1e-8; keep=abs(resid-med)<3*mad
 for z in w:
  if z['well_type']!='COMPOUND': continue
  pred=float(surf([z['row']],[z['col']])@coef); v=(z['fluorescence']-bg)/pred
  allv[z['compound_id']].append(v); allc[z['compound_id']].append(z['log10_conc_uM'])
def hill(q,c):
 ic,s=q; return 1/(1+10**(s*(c-ic)))
for k in gt:
 y=np.array(allv[k]); c=np.array(allc[k]); ok=np.isfinite(y)&(y>-0.3)&(y<1.5)
 res=least_squares(lambda q: np.clip(y[ok],0,1)-hill(q,c[ok]),[-.5,1],bounds=([-4,.2],[3,3]),loss='soft_l1',f_scale=.05)
 print(k,res.x,'err',res.x-[gt[k]['log10_ic50_uM'],gt[k]['hill_slope']])
