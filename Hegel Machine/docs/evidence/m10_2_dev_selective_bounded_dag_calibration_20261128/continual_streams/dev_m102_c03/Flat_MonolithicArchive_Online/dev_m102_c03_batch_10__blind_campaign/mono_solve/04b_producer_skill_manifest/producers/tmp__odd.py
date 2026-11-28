import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf, expit
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v1']
fit=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
forms={
'atan':(lambda x,p:np.arctan(p[0]*x),1), 'tanh':(lambda x,p:np.tanh(p[0]*x),1),'erf':(lambda x,p:erf(p[0]*x),1),'rat':(lambda x,p:x/(1+(p[0]*x)**2),1),'smooth':(lambda x,p:x/np.sqrt(1+(p[0]*x)**2),1),'gaussian_odd':(lambda x,p:x*np.exp(-(p[0]*x)**2),1), 'sin':(lambda x,p:np.sin(p[0]*x),1),'asinh':(lambda x,p:np.arcsinh(p[0]*x),1), 'sinh':(lambda x,p:np.sinh(p[0]*x),1),
'rat4':(lambda x,p:x/(1+(p[0]*x)**2+p[1]*x**4),2),'tanh_cubic':(lambda x,p:np.tanh(p[0]*x+p[1]*x**3),2), 'atan_cubic':(lambda x,p:np.arctan(p[0]*x+p[1]*x**3),2),'erf_cubic':(lambda x,p:erf(p[0]*x+p[1]*x**3),2),'softabs':(lambda x,p:x/(1+p[0]*abs(x)+p[1]*x*x),2),'satpower':(lambda x,p:x/(1+(p[0]*abs(x))**p[1])**(1/p[1]),2),'ratpow':(lambda x,p:x/(1+(p[0]*x)**2)**p[1],2),'tanh_shift':(lambda x,p:np.tanh(p[0]*x+p[1]),2),'sin_shift':(lambda x,p:np.sin(p[0]*x+p[1]),2),'cos_shift':(lambda x,p:np.cos(p[0]*x+p[1]),2),'softsign_cubic':(lambda x,p:(x+p[1]*x**3)/(1+abs(p[0]*(x+p[1]*x**3))),2)}
rows=[]
for name,(fun,k) in forms.items():
 for extra in ((),(1,),(3,),(1,3),(1,3,5)):
  def D(x,p):return np.c_[np.ones(len(x)),fun(x,p),*[x**j for j in extra]]
  a0=1.0 if name=='satpower' else (0.5 if name=='ratpow' else .2 if name in ('softabs','rat4') else 0)
  lo=([.07,.2] if name in ('satpower','ratpow') else [.07,0.] if name in ('rat4','softabs') else [.07,-2])[:k]; hi=[6,6][:k]
  def res(p):
   a=D(u,p);return (a@fit(a,v)-v).ravel()
  opts=[least_squares(res,[i,a0][:k],bounds=(lo,hi),gtol=1e-11,max_nfev=150) for i in (.45,1.05,1.8,2.8,4.)]
  o=min(opts,key=lambda o:o.cost); rows.append((np.sqrt(2*o.cost/v.size),name,extra,o.x))
for row in sorted(rows,key=lambda r:r[0])[:28]:
 e,name,h,p=row; fun=forms[name][0]; a=np.c_[np.ones(len(u)),fun(u,p),*[u**j for j in h]]; b=fit(a,v); print(row,'scales',[np.linalg.norm(j) for j in np.array_split(b[1:],len(h)+1)],'mix similarity',[np.sum(b[1:5]*b[i:i+4])/np.sum(b[1:5]**2) for i in range(5,len(b),4)])
print('panel input range',u.min(),u.max(),'telemetry',s['telemetry_vec'])
