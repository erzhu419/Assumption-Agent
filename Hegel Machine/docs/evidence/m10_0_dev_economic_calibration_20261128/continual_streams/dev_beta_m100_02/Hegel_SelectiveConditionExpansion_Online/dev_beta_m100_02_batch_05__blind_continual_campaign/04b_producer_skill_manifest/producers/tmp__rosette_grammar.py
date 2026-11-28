import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; b=c['single_bank_sweeps']; bias=np.array([-.06,.07,.06,-.05]); bank=b['bank_id']; u=b['ref_signal']; v=b['raw_readout']-bias[bank]; m=abs(v)<4; u,v,bank=u[m],v[m],bank[m]; m=(bank==0)&(u<1.5); t=u[m]; z=v[m]; axis=np.array([-1.2,-.4,.5,1.3]); cross=y[:4,2]-.06
T=np.tanh; A=np.arctan; R=lambda x:x/np.sqrt(1+x*x); S=lambda x:x/(1+x*x); F=lambda x:x/(1+abs(x)); P=lambda x:x; C=lambda x:x**3; E=lambda x:x*np.exp(-x*x); D=lambda x:x/(1+x**4); G=lambda x: np.sin(x); H=lambda x:x**3/(1+x*x); L=lambda x:x/(1+x*x)**1.5
basis={'tanh':T,'atan':A,'soft':R,'rational':S,'lin':P,'cubic':C,'exp':E,'r4':D,'sin':G,'rat3':H,'ratpow':L,'softsign':F}
vals=[]
for n1,f in basis.items():
 for n2,g in basis.items():
  def fun(p,x):return p[0]*f(np.exp(p[1])*x)+p[2]*g(np.exp(p[3])*x)
  try:q=min((least_squares(lambda p:fun(p,t)-z, p0,bounds=([-12,-2,-12,-2],[12,2,12,2]),max_nfev=140) for p0 in ([1,0,.3,-.5],[1,0,-.3,0])),key=lambda r:np.linalg.norm(r.fun))
  except ValueError:continue
  vals.append((np.linalg.norm(q.fun),n1,n2,q.x,fun(q.x,axis)))
for row in sorted(vals,key=lambda a:a[0])[:16]:print(row)
print('direct truth',y[:4,0]+.06)
print('cross',cross)
for n,f in basis.items():
 for nn,g in [('sq',lambda x:x*x),('sat_sq',lambda x:x*x/(1+x*x)),('tanh_sq',lambda x:T(x)**2),('abs',lambda x:abs(x)),('self_sq',lambda x:f(x)**2)]:
  mat=np.c_[f(axis),g(axis)]; p=np.linalg.lstsq(mat,cross,rcond=None)[0]; rm=np.linalg.norm(mat@p-cross)
  if rm<.026:print('cross guess',n,nn,rm,p)
print('context telemetry shape',np.shape(s['aux_manifold_telemetry']))
