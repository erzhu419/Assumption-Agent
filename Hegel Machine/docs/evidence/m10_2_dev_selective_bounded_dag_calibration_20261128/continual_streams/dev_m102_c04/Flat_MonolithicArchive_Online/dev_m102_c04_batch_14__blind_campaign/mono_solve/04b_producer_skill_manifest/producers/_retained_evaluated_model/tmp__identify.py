import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares,minimize_scalar
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-9)[0]
D=lambda *a:np.column_stack((np.ones(len(a[0])),*a))
pairs=list(it.combinations(range(4),2)); off=lambda x:np.array([x[:,i]*x[:,j] for i,j in pairs]).T
funcs={'tanh':lambda x,k:np.tanh(k*x),'atan':lambda x,k:np.arctan(k*x),'sin':lambda x,k:np.sin(k*x),'cos':lambda x,k:np.cos(k*x),'soft':lambda x,k:x/(1+k*abs(x)),'rat':lambda x,k:x/(1+k*x*x),'ratsq':lambda x,k:x*x/(1+k*x*x),'powrat':lambda x,k:x/np.sqrt(1+k*x*x),'log':lambda x,k:np.log1p(k*x*x),'ablog':lambda x,k:np.sign(x)*np.log1p(k*abs(x)),'exp':lambda x,k:np.exp(-k*x*x),'cosh':lambda x,k:np.cosh(k*x),'sinh':lambda x,k:np.sinh(k*x),'softsq':lambda x,k:x*x/(1+k*abs(x)),'cube':lambda x,k:x**3,'sq':lambda x,k:x*x,'abs':lambda x,k:abs(x),'signedsq':lambda x,k:x*abs(x)}
if __name__=='__main__':
 t=s['eval_times']; y=s['train']['traj'][:52]; basis=lambda k:-np.expm1(-np.outer(t,k)); err=lambda k:(y-y@ls(basis(k),np.eye(len(t))).T@basis(k).T).ravel(); r=least_squares(err,[1.8,.4],bounds=([.7,.02],[8,.69]));print('rates',r.x,'rms',np.mean(r.fun**2)**.5)
 for name in ['panel_b','panel_c']:
  pa=s[name]; x=pa['u']; p=off(x);
  for key,v in pa.items():
   if key=='u':continue
   found=[]
   def reg(tag,a):
    e=a@ls(a,v)-v; h=np.sum(a*np.linalg.pinv(a).T,axis=1); found.append((np.mean(e**2)**.5,np.mean((e/np.maximum(1-h,1e-6)[:,None])**2)**.5,tag))
   for name1,a in [('x',D(x)),('off',D(p)),('x+off',D(x,p)),('quad',D(x,x*x,p)),('sq',D(x,x*x))]:reg(name1,a)
   for label,func in funcs.items():
    for extra in [(),(x,),(p,),(x,p)]:
     def des(k):return D(*extra,func(x,k))
     f=lambda k:np.sum((des(k)@ls(des(k),v)-v)**2)
     g=minimize_scalar(f,bounds=(.005,3.5),method='bounded');reg((label,tuple(a.shape[1] for a in extra),round(g.x,5)),des(g.x))
   print(name,key,'best residual',sorted(found,key=lambda x:x[0])[:7]); print('best loo',sorted(found,key=lambda x:x[1])[:7])
