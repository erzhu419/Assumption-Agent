import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import product
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; P=s['panel_a'];u,v=P['u'],P['v'];rng=np.random.default_rng(11)
def ls(x,y):return np.linalg.lstsq(x,y,rcond=1e-11)[0]
def D(*x):return np.column_stack((np.ones(len(x[0])),*x))
a=u;b=np.roll(u,2,axis=1)
poly=[D(a[:,j],b[:,j],a[:,j]*b[:,j],a[:,j]**3,b[:,j]**3,a[:,j]**2,b[:,j]**2) for j in range(4)]
r=[]
for j,x in enumerate(poly):
 c=min((ls(x[i],v[i,j]) for i in (rng.choice(len(u),14,False) for _ in range(350))),key=lambda q:np.median(abs(x@q-v[:,j])));r.append(abs(x@c-v[:,j]))
r=np.array(r).T;g=r.max(1)<.15;print('mask',g.sum(),np.round(np.sort(r.max(1)),4))
F={'tan':lambda z,k:np.tanh(k*z),'sin':lambda z,k:np.sin(k*z),'rat':lambda z,k:z/(1+k*z*z),'soft':lambda z,k:z/(1+k*abs(z)),'cub':lambda z,k:z**3,'atan':lambda z,k:np.arctan(k*z),'erf':lambda z,k:__import__('scipy').special.erf(k*z)}
def matrix(a,b,k,f,h,extra):
 q=[a,b,F[f](a,k[0]),F[h](b,k[1])]
 if extra==1:q+=[a*b]
 if extra==2:q+=[a*b,a*a,b*b]
 if extra==3:q+=[a*b,a*a,b*b,a*a*b,a*b*b]
 if extra==4:q+=[a*b,a*a,b*b,a*a*b,a*b*b, np.sin(k[0]*a)*b]
 return D(*q)
results=[]
for f,h in product(F,F):
 for e in (0,1,2,3):
  def resid(k,mask=g):
   mats=[matrix(a[mask,j],b[mask,j],k,f,h,e) for j in range(4)];return np.column_stack([x@ls(x,v[mask,j])-v[mask,j] for j,x in enumerate(mats)]).ravel()
  fit=least_squares(resid,[1.3,1.6],bounds=(.05,5),max_nfev=45);results.append((np.sqrt(np.mean(fit.fun**2)),f,h,e,fit.x))
for it in sorted(results,key=lambda q:q[0])[:28]:
 er,f,h,e,k=it; cs=[];rs=[]
 for j in range(4):
  x=matrix(a[:,j],b[:,j],k,f,h,e);c=ls(x[g],v[g,j]);cs.append(c);rs.append(x@c-v[:,j]);
 r=np.max(abs(np.array(rs)),axis=0); print(round(er,7),f,h,e,np.round(k,5),np.round(np.quantile(r,[.5,.6,.7,.75]),5),np.round(cs[0],4))
print('B',ls(D(s['panel_b']['u']),s['panel_b']['v']))
