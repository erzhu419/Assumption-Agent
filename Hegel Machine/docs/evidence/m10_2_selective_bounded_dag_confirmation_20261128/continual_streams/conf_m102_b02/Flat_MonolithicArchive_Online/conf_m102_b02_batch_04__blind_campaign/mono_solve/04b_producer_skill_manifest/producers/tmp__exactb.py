from probe import *
u,v=s['panel_b']['u'],s['panel_b']['v']; print('ranges',u.min(0),u.max(0),x.min(0),x.max(0),'times',t)
shapes={}
for p in [1,3,5]:
 for k in [1,2,3,4,6]:
  shapes[f'rat{p}/{k}']=lambda x,a,p=p,k=k:x**p/(1+a*abs(x)**k)
  shapes[f'root{p}/{k}']=lambda x,a,p=p,k=k:x**p/np.sqrt(1+a*abs(x)**k)
  shapes[f'exp{p}/{k}']=lambda x,a,p=p,k=k:x**p*np.exp(-a*abs(x)**k)
for k in [1,2,3]:
 for nam in ['sin','tanh','atan','asinh','erf']:
  shapes[f'{nam}{k}']=lambda x,a,k=k,f=fs[nam]:f(a*x**k)
res=[]
for name,f in shapes.items():
 for power in [1,3,5]:
  def d(x,q):return ic(np.c_[x**power,f(x,np.exp(q[0]))]);
  def err(q):
   a=d(u,q);return (a@ls(a,v)-v).ravel()
  fit=min((least_squares(err,[b],bounds=(-5,2),max_nfev=70) for b in [-2,-.5,1]),key=lambda fit:fit.cost)
  a=d(u,fit.x); h=np.sum(a*np.linalg.pinv(a).T,axis=1); e=fit.fun.reshape(v.shape); res.append((np.mean(e**2)**.5,np.mean((e/(1-h[:,None]))**2)**.5,name,power,np.exp(fit.x[0]),ls(a,v)))
for r in sorted(res,key=lambda r:r[0])[:20]: print(r[:5]);
for r in sorted(res,key=lambda r:r[0])[:3]: print(r[2],r[5].round(5))
