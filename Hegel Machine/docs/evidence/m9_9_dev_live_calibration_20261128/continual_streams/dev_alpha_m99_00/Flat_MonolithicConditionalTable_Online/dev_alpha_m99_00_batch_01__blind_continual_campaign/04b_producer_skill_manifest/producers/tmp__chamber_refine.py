import contextlib,io,runpy,numpy as np
from scipy.optimize import least_squares
with contextlib.redirect_stdout(io.StringIO()): p=runpy.run_path('/tmp/chamber_probe.py')
globals().update({k:p[k] for k in ('m','d','tr','te','di','dt','t','tau','theta','bridge','n','h','v','vr')})
x=m._features(di); _,hold,_=m._coupon(di,x,bridge); labs=di['lab_id']; z=np.c_[np.ones(len(x)),x]; L=np.r_[t,1.]; E=lambda a:np.r_[-np.expm1(-t/a),1/a]; S=sum(w*E(a) for w,a in zip(theta[:2],tau)); Y=np.c_[v,vr]-hold[:,None]*S; W=np.r_[np.ones(len(t)),1.4]; f=z@theta[2:]; print('benchtop drift direction',theta[2:].round(6));
for mode in ['two','rational','fixed','tied-pole','two-estimate-direction','rational-estimate-direction']:
 def shapes(q):
  a=np.exp(q); return {'two':lambda:[E(a[0]),E(a[1]),L], 'rational':lambda:[np.r_[t/(a[0]+t),1/a[0]],L], 'fixed':lambda:[E(tau[0]),E(tau[1]),L], 'tied-pole':lambda:[E(tau[0]),E(a[0]),L]}[mode.split('-estimate')[0]]()
 ini={'two':[.32,1.5],'rational':[1.92],'fixed':[],'tied-pole':[1.5],'two-estimate-direction':[.32,1.5],'rational-estimate-direction':[1.92]}[mode]
 def fit(rows):
  def fun(q,ret=False):
   B=np.array(shapes(q[:len(ini)])); f1=z@np.r_[theta[2],q[len(ini):]] if mode.endswith('direction') else f
   A=np.concatenate([f1[:,None,None]*B.T[None,:,:],np.tile(L[:,None],(len(f),1,1))],axis=-1); a=(A[rows]*W[:,None]).reshape(-1,A.shape[-1]); c=np.linalg.lstsq(a,(Y[rows]*W).ravel(),rcond=None)[0]; r=A@c-Y
   return (r,c,f1) if ret else (r[rows]*W).ravel()
  q=np.r_[np.log(ini),theta[3:] if mode.endswith('direction') else []]; q=least_squares(fun,q,max_nfev=60).x if len(q) else q; r,c,f1=fun(q,True); return r,c,q
 r,c,q=fit(np.ones(len(x),bool)); scores=[]
 for lab in np.unique(labs):
  test=labs==lab; e=fit(~test)[0][test]; scores.append([np.sqrt(np.mean(e[:,:-1]**2)),np.sqrt(np.mean(e[:,-2]**2)),np.sqrt(np.mean(e[:,-1]**2))])
 print(mode,'worst-lab all-strain, final-strain, rate RMS',np.max(scores,axis=0).round(6),'fit r rms',np.sqrt(np.mean(r**2)).round(6),'taus',np.exp(q[:len(ini)]).round(6),'coeff',c.round(6),'direction',q[len(ini):].round(6))
 print('per-lab mean residual at final/rate',np.array([r[labs==lab,-2:].mean(0) for lab in np.unique(labs)]).round(5))
