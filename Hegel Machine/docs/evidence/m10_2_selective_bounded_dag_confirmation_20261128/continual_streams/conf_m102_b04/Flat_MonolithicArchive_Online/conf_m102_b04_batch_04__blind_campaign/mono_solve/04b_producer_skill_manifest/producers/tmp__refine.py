exec(open('/tmp/structure.py').read().split('for rates')[0])
from scipy.optimize import minimize
from itertools import combinations
print('times',t)
Y=np.c_[y,z]; folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=27).split(x))
q=np.c_[np.ones(len(u)),u,u*u,np.column_stack([u[:,j]*u[:,k] for j,k in combinations(range(4),2)])]
def metrics(g,i,j):
 p=g[j]@ls(g[i],Y[i]); e=np.mean(abs(p-Y[j]),axis=0);return np.r_[e[0],np.mean(e[1:]),e[-1]]
def feat(p,mode):
 c=np.exp(p[0]); k=np.exp(p[1]); r=np.exp(p[2]); w=u if len(p)==3 else u-p[3:];
 if mode=='power': h=w**3/(1+c*abs(w)**k)**r
 if mode=='signed': h=w*abs(w)**k/(1+c*w*w)**r
 if mode=='tanh': h=np.tanh(c*w*abs(w)**(k-1))**1 if r==1 else np.sign(w)*abs(np.tanh(c*w*abs(w)**(k-1)))**r
 if mode=='atan': h=np.sign(w)*abs(np.arctan(c*w*abs(w)**(k-1)))**r
 return np.c_[q,h]
for mode in ['power','signed','tanh','atan']:
 for shift in [False,True]:
  errs=[]; print(mode,shift)
  for i,j in folds:
   def fun(p):
    g=feat(p,mode)[i]; er=(g@ls(g,Y[i])-Y[i]);return np.mean(er**2)
   bounds=[(-5,4),(-1,2),(-3,3)]+([(-.4,.4)]*4 if shift else []); ini=list(np.log([.4,3,1] if mode in ('power','signed') else [1,1,3]))+([0]*4 if shift else [])
   opt=minimize(fun,ini,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-14,'gtol':1e-9}); p=opt.x; g=feat(p,mode); errs.append(metrics(g,i,j));
   if len(errs)==1: print('parameters, rmse, coefficients',np.exp(p[:3]),p[3:],np.sqrt(opt.fun),ls(g[i],Y[i])[:,[0,1,2,-1]].round(5))
  print('CV',np.array(errs))
for c,k,r in [(.22,3,1),(.35,3,1),(1.25,3,.5),(.22,2,1),(.22,6,.5),(.22,6,1),(.22,3,.5),(.22,4,1),(1.25,2,.5)]:
 g=feat(np.log([c,k,r]),'power'); print((c,k,r),np.array([metrics(g,i,j) for i,j in folds]))
