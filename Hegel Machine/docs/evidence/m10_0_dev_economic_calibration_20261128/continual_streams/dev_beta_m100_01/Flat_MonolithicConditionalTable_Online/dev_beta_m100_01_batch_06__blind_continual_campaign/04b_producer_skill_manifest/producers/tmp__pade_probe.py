import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; q=c['single_bank_sweeps']; i=q['bank_id']; t=q['ref_signal']; v=q['raw_readout']; clean=(np.max(abs(Y),axis=1)<4)&(np.count_nonzero(X,axis=1)==1)
for j in [0,1]:
 o=(j+2)%4; off=np.median(Y[clean&(X[:,j]==0)&(X[:,o]==0),j]); m=clean&(X[:,j]!=0); k=(i==j)&(abs(v)<4); a=np.r_[X[m,j],t[k]]; b=np.r_[Y[m,j],v[k]]-off
 for n,d in [(1,1),(2,1),(1,2),(2,2),(3,2),(2,3),(3,3)]:
  N=a[:,None]**(1+2*np.arange(n)); D=a[:,None]**(2+2*np.arange(d)); A=np.c_[N,-D*b[:,None]]; pred=lambda p:N@p[:n]/(1+D@p[n:]); errs=[]
  for fold in [np.ones(len(a),bool)]+[np.arange(len(a))%4!=i for i in range(4)]:
   p=np.linalg.lstsq(A[fold],b[fold],rcond=None)[0]; use=~fold if any(~fold) else fold;errs.append(np.max(abs(pred(p)[use]-b[use])))
   if all(fold): full=p
  print(j,n,d,np.round(errs,7),full.round(7))
 for name,f in [('sin',np.sin),('tanh',np.tanh),('erf',__import__('scipy').special.erf)]:
  for model in range(3):
   def pred(p):
    u=p[0]*a; v=p[1]*a
    return [p[2]*f(u)+p[3]*np.sin(v), p[2]*f(u)/(1+v*v)+p[3]*a, p[2]*f(u)+p[3]*a/(1+v*v)][model]
   best=min((least_squares(lambda p:pred(p)-b,[h,k,1,.2],max_nfev=600) for h in (.7,1.5,2.5) for k in (.6,1.7,3)), key=lambda z:np.linalg.norm(z.fun)); print(j,name,model,max(abs(best.fun)),best.x.round(6))
 print('axis ratios',np.c_[a,b,b/a].round(6))
