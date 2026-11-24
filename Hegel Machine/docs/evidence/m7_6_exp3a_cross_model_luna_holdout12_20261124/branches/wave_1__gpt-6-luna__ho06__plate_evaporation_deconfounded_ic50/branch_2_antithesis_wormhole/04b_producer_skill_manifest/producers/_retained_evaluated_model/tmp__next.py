import pickle,numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb')); gt=x['gt']
def b(r,c):
 r=np.asarray(r)/15*2-1;c=np.asarray(c)/23*2-1
 return np.column_stack([np.ones(len(r)),r,c,r*r,c*c,r*c,r**4,c**4,r**3*c,r*c**3,r*r*c*c])
# fit Hill to control-normalized responses; compare background estimation and robust loss / weight by plate
for bgmode in ['median','min','fit']:
 V={k:[] for k in gt}; C={k:[] for k in gt}; P={k:[] for k in gt}
 for pi,p in enumerate(x['plates']):
  w=p['wells']; pos=[z for z in w if z['well_type']=='POS_KILL']; dm=[z for z in w if z['well_type']=='DMSO']
  bg={'median':np.median,'min':lambda a:np.percentile(a,10),'fit':np.mean}[bgmode](np.array([z['fluorescence'] for z in pos]))
  A=b([z['row'] for z in dm],[z['col'] for z in dm]); y=np.array([z['fluorescence']-bg for z in dm]); keep=np.ones(len(y),bool)
  for _ in range(5):
   coef=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; rr=y-A@coef; mad=1.4826*np.median(abs(rr-np.median(rr)))+1e-9;keep=abs(rr-np.median(rr))<2.8*mad
  for z in w:
   if z['well_type']=='COMPOUND':
    k=z['compound_id']; V[k].append((z['fluorescence']-bg)/(b([z['row']],[z['col']])@coef)[0]); C[k].append(z['log10_conc_uM']);P[k].append(pi)
 errs=[]
 for k in gt:
  y=np.array(V[k]);c=np.array(C[k]); pi=np.array(P[k]); ok=np.isfinite(y)&(y>-.3)&(y<1.5)
  def fun(q):return y[ok]-1/(1+10**(q[1]*(c[ok]-q[0])))
  r=least_squares(fun,[-.5,1],bounds=([-4,.2],[3,3]),loss='soft_l1',f_scale=.08)
  errs.append(r.x-[gt[k]['log10_ic50_uM'],gt[k]['hill_slope']])
 print(bgmode,'max',np.max(np.abs(errs),axis=0));print(np.round(errs,3))
