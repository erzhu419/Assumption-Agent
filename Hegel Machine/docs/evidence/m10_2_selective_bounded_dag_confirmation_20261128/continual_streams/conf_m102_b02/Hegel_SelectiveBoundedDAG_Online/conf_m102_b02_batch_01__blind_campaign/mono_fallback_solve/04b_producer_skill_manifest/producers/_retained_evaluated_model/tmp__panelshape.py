from ops import *
from scipy.special import erf, expit
from scipy.optimize import least_squares
fun={'erf':lambda z,k:erf(k*z),'arctan':lambda z,k:np.arctan(k*z),'asinh':lambda z,k:np.arcsinh(k*z),'tan':lambda z,k:np.tanh(k*z),'rat':lambda z,k:z/(1+k*z*z),'sin':lambda z,k:np.sin(k*z),'tancube':lambda z,k:np.tanh(k*z)**3,'tancubed':lambda z,k:np.tanh(k*z**3),'expcubed':lambda z,k:z*(-np.expm1(-k*z*z)),'ratcubed':lambda z,k:z**3/(1+k*z*z),'ratquad':lambda z,k:z**3/(1+k*z**4),'expcubic':lambda z,k:z**3*np.exp(-k*z*z),'poly3':lambda z,k:z**3,'poly5':lambda z,k:z**5,'poly7':lambda z,k:z**7,'softabs':lambda z,k:z*abs(z)/(1+k*abs(z)),'tanhabs':lambda z,k:z*np.tanh(k*abs(z)),'arctanabs':lambda z,k:z*np.arctan(k*abs(z)), 'sincube':lambda z,k:np.sin(k*z)**3}
for p,key in [('panel_b','v'),('panel_c','v1')]:
 u,v=s[p]['u'],s[p][key]; out=[]
 for name,f in fun.items():
  for option in ('bare','linear','cubic','shift','cross','radial'):
   def mat(par):
    k=np.exp(par[0]); z=u+par[1] if option=='shift' else u; v=f(z,k); lin=(u+par[1 if option!='shift' else 2]*u**3) if option=='cubic' else u
    if option in ('linear','shift','cubic') :v=v+par[-1]*lin if option!='cubic' else v+par[2]*u+par[1]*u**3
    if option=='cross':v=v+par[1]*u+par[2]*np.roll(u,1,axis=1)*u*u
    if option=='radial':v=v+par[1]*u+par[2]*u*(u*u).sum(1)[:,None]
    return np.c_[np.ones(len(u)),v]
   def res(par):
    a=mat(par);return (a@np.linalg.lstsq(a,v,rcond=None)[0]-v).ravel()
   k={'bare':1,'linear':2,'cubic':3,'shift':3,'cross':3,'radial':3}[option]; cand=[]
   for k0 in [-1,0,1,1.65]:
    z=np.zeros(k);z[0]=k0
    if k>1:z[-1]=-.7 if option in ('linear','shift','cubic') else 0
    sol=least_squares(res,z,bounds=(np.r_[-4,np.full(k-1,-200)],np.r_[3,np.full(k-1,200)]),max_nfev=110);cand.append(sol)
   sol=min(cand,key=lambda o:o.cost);a=mat(sol.x); R=sol.fun.reshape(v.shape)/(1-(a*np.linalg.pinv(a).T).sum(1))[:,None]; out.append((np.sqrt(np.mean(R**2)),name,option,sol.x))
 print(p, sorted(out,key=lambda a:a[0])[:15])
