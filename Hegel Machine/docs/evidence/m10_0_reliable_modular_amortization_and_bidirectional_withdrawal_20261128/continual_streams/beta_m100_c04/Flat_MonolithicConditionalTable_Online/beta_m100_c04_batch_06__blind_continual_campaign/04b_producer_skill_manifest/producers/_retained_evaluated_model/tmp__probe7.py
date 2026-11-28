import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; x,y=c['ref_rosette'],c['raw_rosette']; m=np.max(abs(y),axis=1)<8; x,y=x[m],y[m]; b=c['single_bank_sweeps']; ids,t,z=[b[k] for k in ('bank_id','ref_signal','raw_readout')]; m=abs(z)<8; ids,t,z=ids[m],t[m],z[m]; j=np.arange(4); opp=(j+2)%4; q=x[:,opp]; pure=np.count_nonzero(x,axis=1)==1
X=np.r_[x[pure], np.eye(4)[ids]*t[:,None]]; Y=np.r_[y[pure],np.where(np.eye(4)[ids],z[:,None],np.nan)]; selfrows=(X!=0)&np.isfinite(Y); R,C=np.where(selfrows); V=X[R,C]; D=Y[R,C]
models={
 'tanh_lin':(lambda v,p:np.tanh(p[0]*v)+p[1]*v,[1.3,.16]),
 'tanh_poly_lin':(lambda v,p:np.tanh(p[0]*v+p[1]*v**3)+p[2]*v,[1.3,0,.16]),
 'tanh_rational_lin':(lambda v,p:np.tanh(p[0]*v/(1+p[1]*v*v))+p[2]*v,[1.3,0,.16]),
 'tanh_out_cubic':(lambda v,p:np.tanh(p[0]*v)+p[1]*v+p[2]*v**3,[1.3,.16,0]),
 'tanh_plus_rational':(lambda v,p:np.tanh(p[0]*v)+p[1]*v/(1+p[2]*v*v),[1.3,.16,0]),
 'rational':(lambda v,p:v*(1+p[0]*v*v+p[1]*v**4)/(1+p[2]*v*v+p[3]*v**4),[.1,.01,.5,.02]),
 'sqrt_lin':(lambda v,p:v/np.sqrt(1+p[0]*v*v)+p[1]*v,[1,.16]),
 'asinh_lin':(lambda v,p:np.arcsinh(p[0]*v)+p[1]*v,[1,-.1]),
 'poly':(lambda v,p:v*np.polynomial.polynomial.polyval(v*v,np.r_[1,p]),[-.3,.05,-.005,0,0,0]),
 'tanh_sin':(lambda v,p:np.tanh(p[0]*v)+p[1]*np.sin(p[2]*v),[1.3,.3,.6]),
}
base0=np.array([.07,-.08,-.06,.05]); fits={}
for name,(f,p0) in models.items():
 def r(p): return p[:4][C]+p[4:8][C]*f(V,p[8:])-D
 a=least_squares(r,np.r_[base0,[1.5,.9,.9,1.5],p0],max_nfev=2500,gtol=1e-13); fits[name]=(np.mean(r(a.x)**2),a.x,f); print(name,np.sqrt(fits[name][0]),max(abs(r(a.x))),a.x.round(7))
_,p,f=min(fits.values(),key=lambda a:a[0]); E=(y-p[:4])/p[4:8]-f(x,p[8:]); print('normalized cross isolated:',np.c_[q[pure].ravel(),E[pure].ravel()][q[pure].ravel()!=0].round(7).tolist());
# A held-out rosette group tests additive and mixed rational grammars.
a=x.ravel(); d=q.ravel(); e=E.ravel(); bank=np.tile(j,len(x)); hold=np.repeat((np.arange(len(x))%5==1)&~pure,4)
for name,F in {
 'q_poly':lambda a,d:np.array([d**k for k in range(1,6)]).T,
 'mixed_poly':lambda a,d:np.array([d**k*a**l for k in range(1,6) for l in range(4) if k+l<=6]).T,
 'q_tanh2':lambda a,d:np.array([np.tanh(d),d*d/(1+d*d),d]).T,
 'q_rational':lambda a,d:np.array([d/(1+d*d),d*d/(1+d*d),d,d*d]).T,
 }.items():
 A=F(a,d); h=np.linalg.lstsq(A[~hold],e[~hold],rcond=None)[0]; print(name,'train/hold',*(np.sqrt(np.mean((A[m]@h-e[m])**2)) for m in (~hold,hold)),h.round(5))
print('study scalar or shape',[(k,getattr(v,'shape',v) if not isinstance(v,dict) else list(v)) for k,v in s.items()]); print('gate open snippet');
