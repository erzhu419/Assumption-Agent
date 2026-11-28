import numpy as np,pickle,itertools
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];u=s['panel_c']['u'];v=s['panel_c']['v2']; i,j=np.array(list(itertools.combinations(range(4),2))).T
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]; P=u[:,i]*u[:,j]; O=np.c_[np.ones(len(u)),u]; out=[]
for cols in [np.arange(6)]+[np.array(c) for c in itertools.combinations(range(6),4)]:
 x=P[:,cols]
 for tag,fun in [('sin',lambda k:np.sin(k*x)),('sat',lambda k:x/(1+k*abs(x))),('rat',lambda k:x/(1+k*x*x)),('cos',lambda k:np.cos(k*x)),('coordR',lambda k:x/(1+k*u[:,j[cols]]**2)),('coordL',lambda k:x/(1+k*u[:,i[cols]]**2)),('sum',lambda k:x/(1+k*(u[:,i[cols]]**2+u[:,j[cols]]**2))),('diff',lambda k:x/(1+k*(u[:,i[cols]]-u[:,j[cols]])**2)),('sinmix',lambda k:np.sin(k[0]*x+k[1]*u[:,i[cols]]+k[2]*u[:,j[cols]]))]:
  def res(k):
   f=np.c_[O,fun(k)];return (f@ls(f,v)-v).ravel()
  fits=[least_squares(res,([k,0,0] if tag=='sinmix' else [k]), max_nfev=60) for k in [.4,1,2.15,3.5]]; p=min(fits,key=lambda p:np.linalg.norm(p.fun));out.append((np.mean(abs(p.fun)),tag,cols,p.x))
print(*sorted(out,key=lambda x:x[0])[:24],sep='\n')
