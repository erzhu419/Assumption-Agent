import numpy as np,pickle
from scipy.optimize import least_squares
from scipy.special import erf,expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; C=s['sensor_calibration_sweeps']; d=C['single_bank_sweeps']; t=d['ref_signal']; y=d['raw_readout']; bank=d['bank_id']; good=abs(y)<6
laws={
 'tanh+rational':(lambda x,a,b:[np.tanh(a*x),x/(1+b*x*x)],2),
 'tanh+atan':(lambda x,a,b:[np.tanh(a*x),np.arctan(b*x)],2),
 'tanh+sqrt':(lambda x,a,b:[np.tanh(a*x),x/np.sqrt(1+b*x*x)],2),
 'tanh+linear+cubic':(lambda x,a:[np.tanh(a*x),x,x**3],1),
 'tanh+cubic':(lambda x,a:[np.tanh(a*x),x**3],1),
 'atan+linear+cubic':(lambda x,a:[np.arctan(a*x),x,x**3],1),
 'erf+linear+cubic':(lambda x,a:[erf(a*x),x,x**3],1),
 'sqrt+linear+cubic':(lambda x,a:[x/np.sqrt(1+a*x*x),x,x**3],1),
 'rational+linear+cubic':(lambda x,a:[x/(1+a*x*x),x,x**3],1),
 'tanh_warp':(lambda x,a,b:[np.tanh(a*x+b*x**3)],2),
 'tanh_warp_rat':(lambda x,a,b:[np.tanh(a*x/(1+b*x*x))],2),
 'tanh_warp_add':(lambda x,a,b:[np.tanh(a*x+b*x/(1+x*x))],2),
 'tanh+exp':(lambda x,a,b:[np.tanh(a*x),x*np.exp(-b*x*x)],2),
 'erf+rat':(lambda x,a,b:[erf(a*x),x/(1+b*x*x)],2),
 'softpower':(lambda x,a,b:[x/(1+a*np.abs(x)**b)**(1/b)],2),
 'ratpoly':(lambda x,a,b:[x/(1+a*x*x+b*x**4),x**3/(1+a*x*x+b*x**4)],2),
 'sinh_rat':(lambda x,a,b:[np.sinh(a*x)/(1+b*np.sinh(a*x)**2)**.5],2),
 'tanh+sin':(lambda x,a,b:[np.tanh(a*x),np.sin(b*x)],2),
 'rat+sin':(lambda x,a,b:[x/(1+a*x*x),np.sin(b*x)],2),
 'tanh+softsign':(lambda x,a,b:[np.tanh(a*x),x/(1+b*abs(x))],2),
 'tanh_over_rat':(lambda x,a,b:[np.tanh(a*x)/(1+b*x*x)],2),
}
def check(name,f,n):
 def calc(p,detail=False):
  err=[]; co=[]
  for k in range(4):
   m=good&(bank==k);q=t[m]; h=np.column_stack([np.ones(len(q)),*f(q,*p)]); c=np.linalg.lstsq(h,y[m],rcond=None)[0];co.append(c);err.extend(h@c-y[m])
  return (np.asarray(err),np.asarray(co)) if detail else np.asarray(err)
 lo=[.01]*n;hi=[8.]*n
 if name in ('tanh_warp','tanh_warp_add','tanh_over_rat','tanh_warp_rat'):lo[1]=-.25
 r=min((least_squares(calc,p,bounds=(lo,hi),gtol=1e-12,ftol=1e-11,xtol=1e-11,max_nfev=400) for p in ([v]*n for v in [.2,.7,1.5,3.])),key=lambda r:r.cost)
 e,c=calc(r.x,True);return np.sqrt(np.mean(e*e)),name,r.x,c[:,1:],e[:11]
for e,n,p,b,r in sorted(check(n,*v) for n,v in laws.items()): print(n,'rms %.8g'%e,'params',p.round(6),'bank0',b[0].round(6),'scaled residual',np.rint(1e6*r).astype(int))
