import pickle,numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb')); gt=x['gt']
def basis(r,c,deg):
 r=np.asarray(r)/15*2-1;c=np.asarray(c)/23*2-1
 cols=[np.ones(len(r)),r,c,r*r,c*c,r*c]
 if deg>=3: cols += [r**3,c**3,r*r*c,r*c*c]
 if deg>=4: cols += [r**4,c**4,r**3*c,r*c**3,r*r*c*c]
 return np.column_stack(cols)
for deg in (2,3,4):
 allv={k:[] for k in gt};allc={k:[] for k in gt}
 for p in x['plates']:
  w=p['wells']; pos=[z for z in w if z['well_type']=='POS_KILL']; bg=np.median([z['fluorescence'] for z in pos]); dm=[z for z in w if z['well_type']=='DMSO']
  A=basis([z['row'] for z in dm],[z['col'] for z in dm],deg); y=np.array([z['fluorescence']-bg for z in dm]); keep=np.ones(len(y),bool)
  for _ in range(5):
   coef=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0];res=y-A@coef; mad=1.4826*np.median(abs(res-np.median(res)))+1e-9; keep=abs(res-np.median(res))<2.8*mad
  for z in w:
   if z['well_type']=='COMPOUND':
    g=(basis([z['row']],[z['col']],deg)@coef)[0];allv[z['compound_id']].append((z['fluorescence']-bg)/g);allc[z['compound_id']].append(z['log10_conc_uM'])
 errs=[]
 for k in gt:
  y=np.array(allv[k]);c=np.array(allc[k]);ok=np.isfinite(y)&(y>-.3)&(y<1.5)
  fit=least_squares(lambda q: y[ok]-1/(1+10**(q[1]*(c[ok]-q[0]))),[-.5,1],bounds=([-4,.2],[3,3]),loss='soft_l1',f_scale=.08)
  err=fit.x-np.array([gt[k]['log10_ic50_uM'],gt[k]['hill_slope']]);errs.append(err)
 print('degree',deg,'maxabs',np.max(np.abs(errs),axis=0));print(np.round(errs,3))
