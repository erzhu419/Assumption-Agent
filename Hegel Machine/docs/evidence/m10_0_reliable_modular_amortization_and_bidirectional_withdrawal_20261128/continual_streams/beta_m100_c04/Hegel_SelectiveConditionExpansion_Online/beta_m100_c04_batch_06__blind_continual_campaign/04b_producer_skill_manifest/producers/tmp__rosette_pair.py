import pickle,numpy as np
from scipy.optimize import least_squares
from numpy.polynomial.chebyshev import chebvander
s=pickle.load(open('public_batch.pkl','rb'))[0]['study']; c=s['sensor_calibration_sweeps']; X,Y=c['ref_rosette'],c['raw_rosette']; keep=np.max(abs(Y),axis=1)<4; X,Y=X[keep],Y[keep]; F=np.random.default_rng(193).permutation(len(X))%5
print('raw observation range',np.quantile(np.asarray(s['y_obs_raw']).reshape(-1,4),[0,.1,.9,1],axis=0))
for label in ['poly','crossCheb','sines','ratAsym','ratEven','mixed','stretch']:
 E=np.empty_like(Y); params=[]
 for i,j in enumerate([2,3,0,1]):
  x,z,y=X[:,i],X[:,j],Y[:,i]; anchor=(x==0)&(z==0)
  def design(p):
   k,a,b=p; u=np.sin(k*x)
   if label=='poly': return np.c_[np.ones(len(x)),chebvander(x/1.8,11)[:,1:],chebvander(z/1.8,11)[:,1:]]
   if label=='crossCheb': return np.c_[np.ones(len(x)),x,u,chebvander(z/1.8,12)[:,1:]]
   if label=='sines': return np.c_[np.ones(len(x)),x,u,z,np.sin(a*z),1-np.cos(b*z)]
   if label=='ratAsym':
    den=1+a*z*z+b*z; return np.c_[np.ones(len(x)),x,u,z,z/den,z*z/den]
   if label=='ratEven':
    den=1+a*z*z+b*z**4; return np.c_[np.ones(len(x)),x,u,z,z/den,z*z/den,z**4/den]
   if label=='mixed': return np.c_[np.ones(len(x)),x,u,chebvander(z/1.8,9)[:,1:],x*z,x*z*z,x*x*z,x*x*z*z,u*z,u*z*z]
   if label=='stretch': return np.c_[np.ones(len(x)),x,np.sin(k*x+a*x**3+b*x**2),chebvander(z/1.8,11)[:,1:]]
  def fit(m):
   def solve(p):
    A=design(p)[m]; q=np.linalg.lstsq(A,y[m],rcond=1e-10)[0]; return A@q-y[m],q
   if label=='poly': p=np.zeros(3)
   else:
    bounds={'sines':([1.8,.02,.02],[3.3,5,7]),'ratAsym':([1.8,.1,-.5],[3.3,4,.5]),'ratEven':([1.8,.001,.0],[3.3,4,4]),'stretch':([1.8,-.5,-.2],[3.3,.5,.2])}.get(label,([1.8,-1,-1],[3.3,1,1])); starts={'sines':[(2.5,.7,1),(2.5,2.5,2.5)],'ratAsym':[(2.5,.8,0)],'ratEven':[(2.5,.7,.05)],'stretch':[(2.5,0,0)]}.get(label,[(2.5,0,0)]); p=min((least_squares(lambda p:solve(p)[0],p,bounds=bounds,max_nfev=250,gtol=1e-11) for p in starts),key=lambda o:o.cost).x
   return p,solve(p)[1],np.sqrt(np.mean(solve(p)[0]**2))
  for f in range(5):
   m=F==f; p,q,_=fit(~m); E[m,i]=design(p)[m]@q-y[m]
  p,q,r=fit(F>=0); params.append((np.round(p,4).tolist(),np.round(q[:7],5).tolist(),round(r,8)))
  if label=='crossCheb' and i in (0,3):
   t=np.arange(-1.4,1.81,.2); v=chebvander(t/1.8,12)[:,1:]@q[3:]; v-=float(chebvander([0],12)[:,1:]@q[3:]); print('cross curve',i,np.c_[t,v].round(6).tolist())
 print(label,'OOF row p90/max',np.quantile(np.linalg.norm(E,axis=1),[.9,1]).round(7),params)
