import pickle,numpy as np,itertools as it
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v1']; n=len(u)
np.set_printoptions(precision=4,suppress=True)
forms={'sin':lambda x,k:np.sin(k*x),'tanh':lambda x,k:np.tanh(k*x),'erf':lambda x,k:erf(k*x),'rat':lambda x,k:x/(1+k*x*x),'soft':lambda x,k:x/(1+k*abs(x)),'gauss':lambda x,k:np.exp(-k*x*x),'cos':lambda x,k:np.cos(k*x),'abs':lambda x,k:abs(x),'satcube':lambda x,k:np.tanh(k*x**3)}
# The rank of each mixing matrix distinguishes independent linear and nonlinear channels from a coupled basis.
for name,f in forms.items():
 def calc(p,ret=False):
  d=np.c_[np.ones(n),u,f(u,np.exp(p[0]))];w=np.linalg.lstsq(d,v,rcond=None)[0];r=d@w-v;return (w,r) if ret else r.ravel()
 p=min([least_squares(calc,[np.log(k)],bounds=(-4,2),max_nfev=70) for k in (.4,1.5,3.8)],key=lambda a:np.linalg.norm(a.fun));w,r=calc(p.x,True); print('one',name,np.exp(p.x),np.std(r),np.linalg.svd(w[1:5],compute_uv=False),np.linalg.svd(w[5:],compute_uv=False));
 if name in ('tanh','rat'):print('linear @ inv nonlinear',w[1:5]@np.linalg.inv(w[5:]))
rows=[]
for (a,f),(b,g) in it.combinations_with_replacement(forms.items(),2):
 for ext in ('linear','no-linear','pair','mixed'):
  def calc(p,ret=False):
   z=f(u,np.exp(p[0])); zz=g(u,np.exp(p[1]));
   ds={'linear':[u,z,zz], 'no-linear':[z,zz], 'pair':[u,z,u*np.roll(u,2,axis=1)], 'mixed':[u,z,z*np.roll(u,2,axis=1)]};d=np.c_[np.ones(n),*ds[ext]];w=np.linalg.lstsq(d,v,rcond=None)[0];r=d@w-v
   return (w,r,d) if ret else r.ravel()
  if ext in ('pair','mixed') and a!=b:continue
  ps=[least_squares(calc,np.log(ks),bounds=(-4,2),max_nfev=85) for ks in ((.6,1.6),(1.7,.7),(2,3.5),(3.4,1.3))];p=min(ps,key=lambda a:np.linalg.norm(a.fun));w,r,d=calc(p.x,True);h=np.sum(d*np.linalg.pinv(d).T,axis=1);err=np.sqrt(np.mean((r/np.maximum(1-h[:,None],1e-6))**2)); rows.append((np.std(r),err,a,b,ext,np.exp(p.x),w))
for row in sorted(rows,key=lambda r:r[1])[:18]:print('two',*row[:6])
for row in sorted(rows,key=lambda r:r[0])[:3]:print('best fit',*row)
