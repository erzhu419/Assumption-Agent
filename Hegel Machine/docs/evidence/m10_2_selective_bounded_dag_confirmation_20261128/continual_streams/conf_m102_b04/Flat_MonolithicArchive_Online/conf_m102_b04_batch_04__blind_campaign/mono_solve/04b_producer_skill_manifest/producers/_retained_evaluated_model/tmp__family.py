exec(open('/tmp/structure.py').read().split('for rates')[0])
from scipy.optimize import minimize
from itertools import combinations
Y=np.c_[y,z]; q=poly(u,2)
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=27).split(x)); w=abs(u)
def feature(p,mode):
 c,k,r=np.exp(p)
 if mode=='rat': h=w**k/(1+c*w**2)**r
 if mode=='pow': h=w**3/(1+c*w**k)**r
 if mode=='atan': h=(np.arctan(c*w**k)/(np.pi/2))**r
 if mode=='tanh': h=np.tanh(c*w**k)**r
 if mode=='gauss': h=w**k*np.exp(-c*w**r)
 if mode=='exp': h=(-np.expm1(-c*w**k))**r
 if mode=='soft': h=w**k/(1+(c*w)**r)**(k/r)
 if mode=='hill': h=(w**k/(c**k+w**k))**r
 return np.c_[q,np.sign(u)*h]
for mode in ['rat','pow','atan','tanh','gauss','exp','soft','hill']:
 for out in ['y','z','both']:
  b={'y':Y[:,:1],'z':z,'both':Y}[out]; E=[]; P=[]; S=[]
  for i,j in folds:
   def objective(p):
    a=feature(p,mode)[i]; return np.mean((a@ls(a,b[i])-b[i])**2)
   best=min([minimize(objective,np.log(p),method='L-BFGS-B',bounds=[(-5,4),(-1.5,2.45),(-3,3.5)],options={'ftol':1e-13,'gtol':1e-9}) for p in [[.6,3,1],[2,1,3],[.5,6,2]]],key=lambda o:o.fun)
   a=feature(best.x,mode); co=ls(a[i],b[i]); e=np.mean(abs(a[j]@co-b[j]),axis=0);E.append(np.r_[np.mean(e),e[-1]]);P.append(np.exp(best.x)); S.append(np.sqrt(best.fun))
  print(mode,out,'par',np.array(P).round(3).tolist(),'fit',round(np.mean(S),5),'52/20',np.round(E[0],5),'worst',np.max(E,axis=0).round(5),'avg',np.mean(E,axis=0).round(5))
