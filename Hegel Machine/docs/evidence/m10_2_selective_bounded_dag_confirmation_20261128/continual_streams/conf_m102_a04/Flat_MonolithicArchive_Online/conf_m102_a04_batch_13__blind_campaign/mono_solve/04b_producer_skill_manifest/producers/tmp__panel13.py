exec(open('/tmp/probe13.py').read().split('for name,fn,key,q,bnd')[0].split('r=v-(a+b*u);')[0]+'\n')
p=s['panel_c']; u=p['u']; i,j=np.triu_indices(4,1)
from scipy.optimize import minimize_scalar
P=lambda u:u[:,i]*u[:,j]
funcs={'linear':lambda u,q:u,'quad':lambda u,q:u*u,'cross':lambda u,q:P(u),'linear+cross':lambda u,q:np.c_[u,P(u)],'poly2':lambda u,q:np.c_[u,u*u,P(u)],'abs':lambda u,q:abs(u),'cos':lambda u,q:np.cos(q*u),'rational2':lambda u,q:u*u/(1+q*u*u),'tanh':lambda u,q:np.tanh(q*u),'signed':lambda u,q:u/(1+q*abs(u)),'sin':lambda u,q:np.sin(q*u),'abssoft':lambda u,q:abs(u)/(1+q*abs(u)),'sqrt':lambda u,q:np.sqrt(1+q*u*u),'exp':lambda u,q:np.exp(-q*u*u),'cyc1':lambda u,q:u*np.roll(u,-1,axis=1),'cyc2':lambda u,q:u*np.roll(u,-2,axis=1),'diffsq':lambda u,q:(u-np.roll(u,-1,axis=1))**2,'sumsq':lambda u,q:(u+np.roll(u,-1,axis=1))**2}
for key in ('v1','v2'):
 Y=p[key]; results=[]
 for name,fn in funcs.items():
  for extra in ('','lin','quad','cross'):
   if extra and name in ('linear','cross','poly2','linear+cross'):continue
   def score(q,ret=False):
    F=one(fn(u,q)); F=np.c_[F,{'lin':u,'quad':u*u,'cross':P(u)}[extra]] if extra else F
    H=F@np.linalg.pinv(F);r=Y-H@Y
    return (np.std(r),np.sqrt(np.mean((r/(1-np.diag(H))[:,None])**2))) if ret else np.mean(r*r)
   opt=min((minimize_scalar(score,bounds=bd,method='bounded') for bd in ((.03,.85),(.85,1.9),(1.9,3.2),(3.2,6))),key=lambda a:a.fun);r,cv=score(opt.x,True);results.append((cv,r,name,extra,opt.x))
 print(key,*sorted(results)[:22],sep='\n')
