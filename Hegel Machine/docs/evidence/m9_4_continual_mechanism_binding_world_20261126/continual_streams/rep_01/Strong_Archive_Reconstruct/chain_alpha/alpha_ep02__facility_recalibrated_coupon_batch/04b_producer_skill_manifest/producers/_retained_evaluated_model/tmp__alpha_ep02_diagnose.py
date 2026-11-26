exec(open('/tmp/alpha_ep02_cv.py').read().split('for lab in [-1,')[0])
from scipy.optimize import least_squares
cal=d['calibration_load_cells']
for l in np.unique(labs):
 g=cal['lab_id']==l; x=cal['cmd_stress'][g]; z=cal['true_stress'][g]; A=np.c_[x,np.ones(len(x))]; q=np.linalg.lstsq(A,z,rcond=None)[0]; print('load cell',l,'gain,bias',q,'residual RMS',np.std(z-A@q))
# Estimate memory from a contrast that exactly annihilates any linear drift.
_,_,vh=np.linalg.svd(times[None,:],full_matrices=True); Q=vh[1:].T

def curvature(keep, tau, w):
 target=y[keep,:-1]@Q
 def r(p):
  M=a.matrix(p[:2])[keep,:-1,:2]; return ((M@p[2:]-y[keep,:-1])@Q).ravel()
 p=np.r_[np.log(tau),w[:2]]
 return least_squares(r,p,bounds=(np.r_[np.log([.2,.2]),[-np.inf]*2],np.r_[np.log([8.,8.]),[np.inf]*2]),loss='cauchy',f_scale=.017,x_scale='jac',max_nfev=100).x

for lab in np.unique(labs):
 keep=labs!=lab; mask=~keep
 tau,w=m._fit(a,y,complete & keep); pred=a.matrix(np.log(tau))@w
 pc=curvature(keep,tau,w)
 print('LAB',lab,'standard tau',tau,'curvature tau',np.exp(pc[:2]),'curvature weights',pc[2:])
 for label,g in [('complete',mask&complete),('imputed',mask&~complete)]:
  e=(y-pred)[g]; delta=e[:,-1]; t=times; slope=np.median(e[:,3:-1]/t[3:],axis=1)
  print(label,'N',len(delta),'median slope/rate innovation/slope-minus-rate',np.round([np.median(slope),np.median(delta),np.median(slope-delta)],5))
  for k in ([0,.25,.4,.5,1] if label=='complete' else [.8,1]):
   ec=e[:,:-1]-k*delta[:,None]*t
   print('update',k,'horizon-worst median',round(np.max(np.median(abs(ec),axis=0)),5),'abs90',round(np.quantile(abs(ec),.9),5),'rate MAE',round(np.mean(abs((1-k)*delta)),5))
 for s in np.unique(tr['stratum']):
  g=mask & complete & (tr['stratum']==s); e=(y-pred)[g]; late=e[:,-2]-.25*times[-1]*e[:,-1]; print('stratum',s,'N',g.sum(),'median abs late',round(np.median(abs(late)),5))
