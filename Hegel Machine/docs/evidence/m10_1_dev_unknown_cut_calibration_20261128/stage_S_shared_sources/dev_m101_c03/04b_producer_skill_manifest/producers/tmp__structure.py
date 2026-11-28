import pickle,itertools as it,numpy as np
from scipy.optimize import minimize_scalar
from sklearn.preprocessing import PolynomialFeatures
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x,y,q=[s['train'][k] for k in ('x','y','traj')];t=s['eval_times'];np.set_printoptions(precision=4,suppress=True,linewidth=140)
u=s['panel_c']['u']; L=np.c_[np.ones(len(u)),u];R=np.eye(len(u))-L@np.linalg.pinv(L)
for k in ('v1','v2'):
 v=s['panel_c'][k]; print(k,'linear residual RMS',np.std(R@v,0));D={'sq':u**2,'cube':u**3,'abs':abs(u),'sign':np.sign(u),'sin2':np.sin(2*u),'sin3':np.sin(3*u),'sin5':np.sin(5*u),'tanh3':np.tanh(3*u),'cross':np.array([u[:,i]*u[:,j] for i,j in it.combinations(range(4),2)]).T,'cos2':np.cos(2*u),'cos3':np.cos(3*u),'cos5':np.cos(5*u)}
 for j,e in enumerate((R@v).T):
  rr=[]
  for n,a in D.items():
   f=R@a;co=f.T@e/np.sum(f*f,0);rm=np.sqrt(np.mean((e[:,None]-f*co)**2,0));rr.extend((round(vv,5),n,i,round(co[i],4)) for i,vv in enumerate(rm))
  print(j,sorted(rr)[:5]);
  for name,func in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:
   rr=[]
   for i in range(4):
    def err(p):
     f=R@func(p*u[:,i]);return np.sqrt(np.mean((e-f*(f@e)/(f@f))**2))
    for a,b in [(0.01,2),(2,4),(4,7),(7,11)]:
     z=minimize_scalar(err,bounds=(a,b),method='bounded');rr.append((round(z.fun,5),i,round(z.x,4)))
   print(name,min(rr),end='; ')
  print()
A=lambda u,v:np.linalg.lstsq(np.c_[np.ones(len(u)),u],v,rcond=None)[0]
a=s['panel_a']; co=[]
for i in range(4):
 u,v=a['u'][:,i],a['v'][:,i]; X=np.c_[np.ones(len(u)),u];p=np.median((v[:,None]-v)[np.triu_indices(len(u),1)]/(u[:,None]-u)[np.triu_indices(len(u),1)]);b=np.median(v-p*u);j=abs(v-b-p*u)<.02;co.append(np.linalg.lstsq(X[j],v[j],rcond=None)[0])
b=A(**s['panel_b']);a=np.array(co);cal=(x-a[:,0])/a[:,1]; bb=cal@b[1:]+b[0];bc=(cal-b[0])@np.linalg.inv(b[1:]);F=-np.expm1(-t[:,None]*[1.8131,.40838]);amp=q@np.linalg.pinv(F.T);print('amps ranges',amp.min(0),amp.max(0),'RMS time error',np.std(q-amp@F.T));print('amp quantiles and corr',np.quantile(amp,[0,.1,.5,.9,1],axis=0),np.corrcoef(np.c_[amp,y,cal,bb].T)[:3]);
for n,z in [('cal',cal),('B',bb),('Bi',bc)]:
 print(n)
 for j,Y in enumerate(np.c_[y,amp].T):
  D=[];N=[]
  def add(f,n):D.append(f);N.append(n)
  for i in range(4):
   for w in [.5,1,1.5,2,2.5,3,3.5,4,5,6]:
    for fn in (np.sin,np.cos,np.tanh):add(fn(w*z[:,i]),f'{fn.__name__}{w} z{i}')
  for i,k in it.combinations(range(4),2):
   for v in [z[:,i]+z[:,k],z[:,i]-z[:,k],z[:,i]*z[:,k]]:
    for w in [1,2,3]:
     for fn in (np.sin,np.cos,np.tanh):add(fn(w*v),f'{fn.__name__}{w} pair{i}{k}_{len(D)}')
  l=np.c_[np.ones(len(z)),z];r=Y-l@np.linalg.lstsq(l,Y,rcond=None)[0];d=np.array(D).T;d-=l@np.linalg.lstsq(l,d,rcond=None)[0];rho=(d.T@r)/np.linalg.norm(d,axis=0)/np.linalg.norm(r);print(j,'rms lin',np.std(r),[(N[k],round(rho[k],3)) for k in np.argsort(abs(rho))[-6:][::-1]])
