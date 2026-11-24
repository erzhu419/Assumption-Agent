import pickle,numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb')); gt=x['gt']
def b(r,c):
 r=np.asarray(r)/15*2-1;c=np.asarray(c)/23*2-1
 return np.column_stack([np.ones(len(r)),r,c,r*r,c*c,r*c,r**4,c**4,r**3*c,r*c**3,r*r*c*c])
# DMSO-derived smooth gain map normalized to mean 1; fit baseline/surface using robust control residuals.
plates=[]
for p in x['plates']:
 w=p['wells']; dm=[z for z in w if z['well_type']=='DMSO']; pos=[z for z in w if z['well_type']=='POS_KILL']
 bg=np.median([z['fluorescence'] for z in pos]); A=b([z['row'] for z in dm],[z['col'] for z in dm]); y=np.array([z['fluorescence']-bg for z in dm]); keep=np.ones(len(y),bool)
 for _ in range(5):
  co=np.linalg.lstsq(A[keep],y[keep],rcond=None)[0]; rr=y-A@co; mad=1.4826*np.median(abs(rr-np.median(rr)))+1e-8;keep=abs(rr-np.median(rr))<2.8*mad
 co=co/np.mean(A@co); plates.append((w,bg,co))
# Assemble individually fit raw response, nuisance plate amplitudes and optionally backgrounds (bounded near POS control).
for bgshift in [-20,0,20]:
 errs=[]
 for k in gt:
  cs=[]; ys=[]; pis=[]; gs=[]
  for pi,(w,bg,co) in enumerate(plates):
   for z in w:
    if z['well_type']=='COMPOUND' and z['compound_id']==k:
     cs.append(z['log10_conc_uM']); ys.append(z['fluorescence']-bg-bgshift); pis.append(pi); gs.append((b([z['row']],[z['col']])@co)[0])
  c=np.array(cs); y=np.array(ys); pi=np.array(pis); g=np.array(gs)
  # Fit per-plate top amplitudes alongside Hill parameters on fluorescence scale; downweight pipetting outliers robustly.
  def pred(q):
   ic,s=q[:2]; amps=q[2:]
   return g*amps[pi]/(1+10**(s*(c-ic)))
  q0=np.r_[[-.3,1.05],[2200,2500,2800,3100]]
  fit=least_squares(lambda q:y-pred(q),q0,bounds=([-4,.2,*([500]*4)],[3,3,*([6000]*4)]),loss='soft_l1',f_scale=35,max_nfev=3000)
  errs.append(fit.x[:2]-[gt[k]['log10_ic50_uM'],gt[k]['hill_slope']])
 print('bgshift',bgshift,'maxabs',np.max(np.abs(errs),axis=0),'errors',np.round(errs,3))
