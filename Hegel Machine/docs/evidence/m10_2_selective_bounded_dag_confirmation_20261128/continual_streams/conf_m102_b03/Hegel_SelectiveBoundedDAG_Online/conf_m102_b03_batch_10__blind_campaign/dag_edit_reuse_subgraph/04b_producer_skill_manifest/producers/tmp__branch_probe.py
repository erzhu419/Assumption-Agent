import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u,v=[s['panel_c'][k] for k in ('u','v1')];ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
forms={'rational':lambda x,a,b:x/(1+a*abs(x)**b),'cubic_rational':lambda x,a,b:x**3/(1+a*abs(x)**b),'satpower':lambda x,a,b:x/(1+a*x*x)**b,'power':lambda x,a,b:x*abs(x)**a,'exp':lambda x,a,b:x*np.exp(-a*abs(x)**b),'tanh_power':lambda x,a,b:np.tanh(a*x)*abs(x)**b,'softsign':lambda x,a,b:x/(1+a*abs(x)),'tanh':lambda x,a,b:np.tanh(a*x),'sin':lambda x,a,b:np.sin(a*x),'atan':lambda x,a,b:np.arctan(a*x),'cos':lambda x,a,b:np.cos(a*x)}
i,j=np.triu_indices(4,1); adj=u*np.roll(u,-1,axis=1)
for name,f in forms.items():
 for mode in [0,1,2,3]:
  def design(w,p):
   a,b,sh,cross=p; z=w+sh+cross*np.roll(w,-1,axis=1); h=f(z,a,b)
   extra=[[],[w*w],[w[:,i]*w[:,j]],[w**3]][mode]
   return np.column_stack([np.ones(len(w)),w,h]+extra)
  def fun(p):
   d=design(u,p);return (d@ls(d,v)-v).ravel()
  sol=min([least_squares(fun,[a,1.5,0,0],bounds=([.001,.02,-.5,-.5],[5,6,.5,.5]),max_nfev=100) for a in [.22,1.5,2.9]],key=lambda r:r.cost)
  sc=np.sqrt(np.mean(sol.fun**2));print('fit',round(sc,8),name,mode,np.round(sol.x,5),flush=True)
