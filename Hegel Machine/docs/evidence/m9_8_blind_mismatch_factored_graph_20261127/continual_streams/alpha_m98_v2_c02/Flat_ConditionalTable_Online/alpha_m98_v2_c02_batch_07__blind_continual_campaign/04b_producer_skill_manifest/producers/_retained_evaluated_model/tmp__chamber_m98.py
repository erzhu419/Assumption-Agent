exec(open('/tmp/probe_m98.py').read().split('for name,C')[0])
from scipy.optimize import least_squares
for mode in ['joint','train-test','diag-only']:
 if mode=='joint': X=xd
 elif mode=='diag-only': X=m._standardize(di,di)[0]
 else:
  X=di['coupon_features_raw'].copy()
  for lab in np.unique(di['lab_id']):
   refs=combined['ref_coupons'][combined['lab_id']==lab]; ii=di['lab_id']==lab
   X[ii]=(X[ii]-np.median(refs.mean(0)))/np.median(refs.std(0))
  X=np.where(np.isfinite(X),X,0.)
 D=th[2]+X@th[3:]; F=np.c_[np.ones(len(X)),X]; base=h[:,None]-z
 def predict(p,model):
  G=base.copy(); C=-np.expm1(-t); v=D.copy()
  for name,val in zip(model,p):
   if name=='gain': v+=val*D
   elif name=='bias': v+=val
   elif name=='square': v+=val*D**2
   elif name=='hold': v+=val*h
   elif name=='rate': C=-np.expm1(-(1+val)*t)/(1+val)
   elif name=='w0': G[:,0]-=val*z[:,0]
   elif name=='w1': G[:,1]-=val*z[:,1]
   elif name.startswith('x'): v+=val*F[:,int(name[1:])]
  return G@R.T+v[:,None]*C,G@g+v
 for model in [(),('gain',),('gain','bias'),('gain','bias','square'),('gain','hold'),('gain','w0','w1','rate'),('gain','bias','rate'),('x0','x1','x2','x3','x4'),('x0','x1','x2','x3','x4','rate','w0','w1')]:
  def fit(idx):
   if not model:return []
   def fun(p):
    a,b=predict(p,model);return np.r_[((a-yd)[idx]/.010).ravel(),(b-rd)[idx]/.012]
   return least_squares(fun,np.zeros(len(model)),loss='cauchy',f_scale=2,bounds=(-.8,.8)).x
  ea=np.empty_like(yd); eb=np.empty_like(rd)
  for lab in np.unique(di['lab_id']):
   mask=di['lab_id']==lab; a1,b1=predict(fit(~mask),model);ea[mask]=(a1-yd)[mask];eb[mask]=(b1-rd)[mask]
  worst=[max(np.median(np.abs(e[di['lab_id']==lab])) for lab in np.unique(di['lab_id'])) for e in [ea,eb,ea[:,-3:]]]
  print(mode,','.join(model) or 'fixed','params',np.round(fit(slice(None)),5),'OOF med',np.round([np.median(abs(ea)),np.median(abs(eb))],5),'worst lab [strain,rate,late]',np.round(worst,5))
 print('drift versus empirical correction',np.c_[D,np.median((yd-base@R.T)/(-np.expm1(-t)),axis=1)-D,rd-base@g-D].round(4))
