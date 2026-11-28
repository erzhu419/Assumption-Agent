import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
p=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']['panel_c']; u=p['u']
fun={'tanh':np.tanh,'atan':np.arctan,'erf':erf,'soft':lambda z:z/np.sqrt(1+z*z),'sin':np.sin,'cos':np.cos,'rational':lambda z:z/(1+z*z),'asinh':np.arcsinh}
def spec(k):
 f=fun[k]; return lambda u,a: u+a[1]*f(a[0]*u)
def model(u,v,f,a):
 def basis(a,u): return np.column_stack((np.ones(len(u)),f(u,a)))
 def res(a):
  X=basis(a,u); return (X@np.linalg.lstsq(X,v,rcond=None)[0]-v).ravel()
 opt=least_squares(res,a,max_nfev=180); X=basis(opt.x,u); return opt.x,np.linalg.lstsq(X,v,rcond=None)[0],np.sqrt(np.mean(opt.fun**2))
for key in ('v1','v2'):
 v=p[key]; ks=['tanh','atan','erf','soft','sin','rational','asinh'] if key=='v1' else ['cos']
 candidates=[(k,spec(k),[1.5,2.5 if key=='v1' else -.65]) for k in ks]
 candidates += [('power-soft',lambda u,a:u+a[1]*u/(1+a[0]**2*u*u)**a[2],[1.5,3.,.5]),('odd-rational',lambda u,a:u*(1+a[0]*u*u)/(1+a[1]**2*u*u+a[2]**2*u**4),[1.,1.,.1])] if key=='v1' else [('even-rational',lambda u,a:u+a[0]*u*u/(1+a[1]**2*u*u),[.5,.5]),('cos+square',lambda u,a:u+a[1]*np.cos(a[0]*u)+a[2]*u*u,[1.25,-.67,0.])]
 for k in ks:
  for pow in [2,3,5]:
   f=fun[k]; candidates.append((f'{k}+pow{pow}',lambda u,a,f=f,pow=pow:u+a[1]*f(a[0]*u)+a[2]*u**pow,[1.5,2.5 if key=='v1' else -.65,0.]))
 ranked=[]
 for k,f,a in candidates:
  q,g,e=model(u,v,f,a); ranked.append((e,k,f,q)); print(key,k,np.round(q,6),round(e,8))
 for _,k,f,q in sorted(ranked,key=lambda o:o[0])[:6]:
  errors=[]
  for i in range(len(u)):
   mask=np.arange(len(u))!=i; a,g,_=model(u[mask],v[mask],f,q); errors.append(np.abs(np.c_[np.ones(1),f(u[i:i+1],a)]@g-v[i]).mean())
  print('LOO',key,k,'mean/max',np.mean(errors),max(errors))
