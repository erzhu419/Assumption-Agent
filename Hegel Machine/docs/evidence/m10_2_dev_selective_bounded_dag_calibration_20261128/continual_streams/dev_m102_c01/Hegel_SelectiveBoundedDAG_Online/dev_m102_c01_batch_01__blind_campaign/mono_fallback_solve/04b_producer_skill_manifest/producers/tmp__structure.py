exec(open('/tmp/models.py').read().split('rows=[]')[0])
from itertools import combinations_with_replacement as cr
from scipy.optimize import least_squares
Z={k:base[k] for k in ['X','A-','A+','BX','BA-','BA+']}
# Test separable odd and even nonlinearities in each calibrated coordinate system.
for name,z0 in Z.items():
 D=design(z0); Q=poly(z0,2)[:,4:]; pair=z0[:,np.triu_indices(4,1)[0]]*z0[:,np.triu_indices(4,1)[1]]
 for tar in range(2):
  vals=[]
  for kind in ['trig','tanh','rat','p3','p4','pairs','fulltrig']:
   def ff(h,z):
    dd=design(z); k=np.exp(h[0]); i,j=np.triu_indices(4,1)
    if kind=='p3':return np.c_[dd,poly(z,3)[:,4:]]
    if kind=='p4':return np.c_[dd,poly(z,4)[:,4:]]
    if kind=='pairs':return np.c_[dd, z*z, z[:,i]*z[:,j],z**3,z**4]
    if kind=='trig':return np.c_[dd,np.sin(k*z),np.cos(k*z),z[:,i]*z[:,j]]
    if kind=='fulltrig':return np.c_[dd,np.sin(k*z),np.cos(k*z),np.sin(k*(z[:,i]+z[:,j])),np.cos(k*(z[:,i]+z[:,j])),np.sin(k*(z[:,i]-z[:,j])),np.cos(k*(z[:,i]-z[:,j]))]
    if kind=='tanh':return np.c_[dd,np.tanh(k*z),np.tanh(k*z)**2,z[:,i]*z[:,j]]
    return np.c_[dd,z*z/(1+k*z*z),z*z*z/(1+k*z*z),z[:,i]*z[:,j]]
   out=[]
   for jj,ll in splits:
    def obj(h):
     F=ff(h,z0[jj]); return F@fit(F,Y[jj,tar])-Y[jj,tar]
    if kind in ('p3','p4','pairs'):h=[0.]
    else:h=least_squares(obj,[0.],bounds=(-2.5,1.6),max_nfev=25).x
    F=ff(h,z0); out.append([np.mean(abs(obj(h))), np.mean(abs(F[ll]@fit(F[jj],Y[jj,tar])-Y[ll,tar])),np.exp(h[0])]);
   out=np.array(out); vals.append((kind,*out[0],out[:,1].mean(),out[:,1].max()))
  print(name,tar,sorted(vals,key=lambda r:r[-2])[:3])
# Recover spatial coefficients of the first trajectory mode for several time laws.
for typ in range(5):
 def curves(h):
  k=np.exp(h[0]); e=-np.expm1(-k*t)
  return [e[:,None],(t/(t+k))[:,None],np.c_[e,t*np.exp(-k*t)],np.c_[e,t],np.c_[e,-np.expm1(-np.exp(h[1])*t)]][typ]
 def obj(h):
  D=curves(h); return (D@fit(D,T.T)-T.T).ravel()
 o=least_squares(obj,[-.1,.5],bounds=(-3,3)); print('time',typ,np.exp(o.x),np.std(o.fun))
