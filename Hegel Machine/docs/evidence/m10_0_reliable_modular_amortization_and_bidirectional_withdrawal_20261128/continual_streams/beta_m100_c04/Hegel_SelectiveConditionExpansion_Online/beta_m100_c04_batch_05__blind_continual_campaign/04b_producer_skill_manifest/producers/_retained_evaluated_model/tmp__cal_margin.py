import sys,pickle
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0,'/holdout_task')
import upstream_transducer as m
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=(c[k] for k in ('ref_rosette','raw_rosette')); b=c['single_bank_sweeps']; j,u,v=(b[k] for k in ('bank_id','ref_signal','raw_readout')); clean=m._clean(y); bc=np.zeros(len(v),bool)
for h in range(4): bc[j==h]=m._clean(v[j==h,None])
edge=clean&(np.max(abs(x),axis=1)>1.4); mid=clean&(np.max(abs(x),axis=1)<=.75)
models={
'current':([.8,.9],[.3,.05],[2.8,3],lambda z,p,a,b:[np.tanh(a*z),z,z**3,p,np.tanh(b*p*p)]),
'mixed':([.8,.9,0],[.3,.05,-.5],[2.8,3,.5],lambda z,p,a,b,d:[np.tanh(a*(z+d*p)),z,z**3,p,np.tanh(b*p*p)]),
'pair_diff':([.8,.9],[.3,.05],[2.8,3],lambda z,p,a,b:[np.tanh(a*z),z,z**3,p,np.tanh(b*p*p),z*p,z*z*p,z*p*p]),
'rational':([.8,.9,.5],[.3,.05,0],[2.8,3,4],lambda z,p,a,b,d:[np.tanh(a*z),z,z**3/(1+d*z*z),p,np.tanh(b*p*p)]),
'cross_rational':([.8,.8],[.3,0],[2.8,3],lambda z,p,a,b:[np.tanh(a*z),z,z**3,p,p*p/(1+b*p*p)]),
'double_tanh':([1.2,.9,.5],[.3,.05,.12],[3,3,3],lambda z,p,a,b,d:[np.tanh(a*z),z,np.tanh(d*z),p,np.tanh(b*p*p)]),
'quintic':([.8,.9],[.3,.05],[2.8,3],lambda z,p,a,b:[np.tanh(a*z),z,z**3,z**5,p,np.tanh(b*p*p)]),
}
for power in [5,7,9,11,13]: models['power'+str(power)]=([1.2,.9],[.3,.05],[3,3],lambda z,p,a,b,n=power:[np.tanh(a*z),z**n,p,np.tanh(b*p*p)])
for name,(init,lo,hi,f) in models.items():
 e=np.zeros_like(y); inv=np.zeros(len(x)); eb=np.zeros(len(v));
 def basis(z,p,par): return np.column_stack((np.ones_like(z),*f(z,p,*par)))
 for fold in range(7):
  te=np.arange(len(x))%6==fold; be=np.arange(len(v))%6==fold; tr=~te; bt=~be
  tr[tr]=m._clean(y[tr]);
  for h in range(4):
   q=bt&(j==h); bt[q]=m._clean(v[q,None])
  dat=[(np.r_[x[tr,h],u[bt&(j==h)]], np.r_[x[tr,k],np.zeros(np.sum(bt&(j==h)))],np.r_[y[tr,h],v[bt&(j==h)]]) for h,k in enumerate(m._PAIR)]
  def profile(par,coef=False):
   out=[]
   for z,p,t in dat:
    D=basis(z,p,par); C=np.linalg.lstsq(D,t,rcond=None)[0]; out.append(C if coef else D@C-t)
   return np.array(out) if coef else np.concatenate(out)
  opt=least_squares(profile,init,bounds=(lo,hi),loss='soft_l1',f_scale=.008,max_nfev=100); par=opt.x; coef=profile(par,True)
  def forward(z): return np.column_stack([basis(z[:,h],z[:,k],par)@coef[h] for h,k in enumerate(m._PAIR)])
  valid=te&clean; e[valid]=forward(x[valid])-y[valid]
  for i in np.flatnonzero(valid):
   z=least_squares(lambda z:forward(z[None])[0]-y[i],x[i],max_nfev=30,gtol=1e-11).x; inv[i]=max(abs(z-x[i]))
  for h in range(4):
   q=be&bc&(j==h); eb[q]=abs(basis(u[q],np.zeros(sum(q)),par)@coef[h]-v[q]);
 print(name,'edge inverse p95/max',np.round(np.quantile(inv[edge],[.95,1]),6),'central inverse max',round(inv[mid].max(),6),'bank max',round(eb[bc].max(),6),'shape',np.round(par,5),'coef0',np.round(coef[0],5),flush=True)
