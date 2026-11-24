import pickle,numpy as np
from scipy.optimize import least_squares
x=pickle.load(open('/holdout_task/public_screen.pkl','rb')); gt=x['gt']
def b(r,c,terms):
 r=np.asarray(r)/15*2-1;c=np.asarray(c)/23*2-1
 cols=[np.ones(len(r)),r,c,r*r,c*c,r*c]
 if terms>=7: cols += [r**4,c**4,r**3*c,r*c**3,r*r*c*c]
 return np.column_stack(cols)
for deg in [2,4]:
 for loss in ['linear','soft_l1','huber','cauchy']:
  for scale in [.04,.08,.15]:
   V={k:[] for k in gt};C={k:[] for k in gt}
   for p in x['plates']:
    w=p['wells'];pos=[z for z in w if z['well_type']=='POS_KILL'];dm=[z for z in w if z['well_type']=='DMSO'];bg=np.median([z['fluorescence'] for z in pos]); A=b([z['row'] for z in dm],[z['col'] for z in dm],6 if deg==2 else 11); yy=np.array([z['fluorescence']-bg for z in dm]); keep=np.ones(len(yy),bool)
    for _ in range(5):
     co=np.linalg.lstsq(A[keep],yy[keep],rcond=None)[0];rr=yy-A@co; mad=1.4826*np.median(abs(rr-np.median(rr)))+1e-8;keep=abs(rr-np.median(rr))<2.8*mad
    for z in w:
     if z['well_type']=='COMPOUND':
      V[z['compound_id']].append((z['fluorescence']-bg)/(b([z['row']],[z['col']],6 if deg==2 else 11)@co)[0]);C[z['compound_id']].append(z['log10_conc_uM'])
   errs=[]
   for k in gt:
    y=np.array(V[k]);c=np.array(C[k]); ok=np.isfinite(y)&(y>-.3)&(y<1.5)
    fit=least_squares(lambda q:y[ok]-1/(1+10**(q[1]*(c[ok]-q[0]))),[-.5,1],bounds=([-4,.2],[3,3]),loss=loss,f_scale=scale)
    errs.append(fit.x-[gt[k]['log10_ic50_uM'],gt[k]['hill_slope']])
   e=np.abs(errs); print(deg,loss,scale,'max',np.max(e,axis=0),'worst',np.argmax(e[:,0]),np.argmax(e[:,1]))
