import pickle,numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v=[s['panel_b'][k] for k in ('u','v')]; print({k:{j:np.shape(a) for j,a in d.items()} for k,d in s.items() if isinstance(d,dict)})
ls=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
fun={'s':lambda z,k:np.sin(k*z),'r':lambda z,k:z/(1+k*z*z),'t':lambda z,k:np.tanh(k*z),'e':lambda z,k:erf(k*z),'a':lambda z,k:np.arctan(k*z)}
init={'s':2.9,'r':.8,'t':1.45,'e':1.46,'a':1.35}
def fit(name,u,v):
 typ,mode=name[0],name[1:]; k0=init[typ]; f=fun[typ]
 if mode in ('even','odd','third','cross','shift','plain'):
  def basis(z,p):
   k=p[0]; a=[np.ones(len(z)),z,f(z,k)];
   if mode=='even':a.append(np.cos(p[1]*z))
   if mode=='odd':a.append(np.sin(p[1]*z))
   if mode=='third':a.append(z**3)
   if mode=='cross':a.append(z*np.roll(z,-1,axis=1))
   if mode=='shift':a[2]=f(z+p[1],k)
   return np.column_stack(a)
  pi=[k0]+({'even':[1.9],'odd':[3.9],'shift':[0.]}.get(mode,[]))
 else:
  def basis(z,p):
   k,h=p[:2]; q=f(z,k)-h*z
   if mode=='com': pass
   if mode=='com3':q=q+p[2]*z**3
   if mode=='com2':q=q+p[2]*z*z
   if mode=='com5':q=q+p[2]*z**3+p[3]*z**5
   if mode=='comr':q=q+p[2]*z/(1+p[3]*z*z)
   if mode=='comx':q=q+p[2]*z*np.roll(z,-1,axis=1)
   if mode=='comn':q=q+p[2]*np.sin(p[3]*z)
   if mode=='com4':q=q+p[2]*np.cos(p[3]*z)
   if mode=='coml':return np.column_stack([np.ones(len(z)),q,z])
   return np.column_stack([np.ones(len(z)),q])
  pi=[k0,{'s':.1,'r':.9,'t':1.1,'e':1.54,'a':1.}[typ]]+({'com3':[0.],'com2':[0.],'com5':[0.,0.],'comx':[0.],'comr':[0.,.4],'comn':[0.,3.7],'com4':[0.,1.9]}.get(mode,[]))
 def res(p):
  x=basis(u,p);return (x@ls(x,v)-v).ravel()
 p=least_squares(res,pi,max_nfev=100).x; c=ls(basis(u,p),v)
 return lambda z:basis(z,p)@c,p
if __name__=='__main__':
 names=[a+b for a in fun for b in ('plain','even','odd','third','cross','shift','com','com3','com2','com5','comr','comx','comn','com4')]; scores=[]
 for name in names:
  out=[]
  for j in range(5):
   keep=np.arange(len(u))%4!=j; pred,p=fit(name,u[keep],v[keep]); test=~keep if j<4 else keep; out.append(np.mean(abs(pred(u[test])-v[test])))
  scores.append((np.mean(out[:4]),max(out[:4]),out[-1],name,p))
 for m,w,e,n,p in sorted(scores)[:25]: print(n,np.round([m,w,e],7),p.round(6))
