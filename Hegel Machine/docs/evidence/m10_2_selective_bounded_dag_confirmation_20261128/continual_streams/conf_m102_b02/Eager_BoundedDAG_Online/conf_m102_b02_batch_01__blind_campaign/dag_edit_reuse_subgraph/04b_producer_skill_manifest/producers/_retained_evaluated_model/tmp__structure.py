exec(open('/tmp/probe.py').read().split("print('raw',score(x))")[0])
from scipy.optimize import minimize_scalar
from itertools import combinations

def assess(D,v):
 H=D@np.linalg.pinv(D); r=H@v-v
 return np.sqrt(np.mean((r/np.maximum(1e-9,1-H.diagonal())[:,None])**2))
def basis(u,kind):
 a,b=np.array(list(combinations(range(4),2))).T
 return {'sq':u**2,'cube':u**3,'fourth':u**4,'fifth':u**5,'cross':u[:,a]*u[:,b],'triple':np.column_stack([np.prod(u[:,j],1) for j in combinations(range(4),3)]),'abs':abs(u),'xabs':u*abs(u)}[kind]
for p,k in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[p]['u'];v=s[p][k]; print('\n',p,k)
 for kinds in [(),('sq',),('cube',),('cross',),('triple',),('sq','cube'),('sq','fourth'),('cube','fifth'),('sq','cross'),('sq','triple'),('cube','cross'),('sq','xabs'),('sq','abs')]:
  D=np.column_stack([poly(u,1),*[basis(u,a) for a in kinds]]); X=np.column_stack([poly(x,1),*[basis(x,a) for a in kinds]]); C=np.linalg.lstsq(D,v,rcond=None)[0]; print(kinds,'loo',round(assess(D,v),6),'target',score(X@C));
  if kinds in [('sq',),('cube',),('cross',),('triple',)]: print('nonlinear coef',C[5:])
 for name,f,lo,hi in [('sin',np.sin,.2,14),('cos',np.cos,.2,14),('sincos',lambda q:np.c_[np.sin(q),np.cos(q)],.2,14),('tanh',np.tanh,.2,14),('asinh',np.arcsinh,.2,14),('rational',lambda q:1/(1+q*q),.05,5),('rationalodd',lambda q:q/(1+q*q),.05,5),('exp',lambda q:np.exp(-q*q),.05,5)]:
  for sq in (False,True):
   def Des(u,a): return np.c_[poly(u,1),u*u if sq else np.empty((len(u),0)),f(a*u)]
   g=np.linspace(lo,hi,230); es=[assess(Des(u,a),v) for a in g]; j=np.argmin(es); opt=minimize_scalar(lambda a:assess(Des(u,a),v),bounds=(g[max(0,j-1)],g[min(len(g)-1,j+1)]),method='bounded'); C=np.linalg.lstsq(Des(u,opt.x),v,rcond=None)[0]; print(name,sq,'loo,f',round(opt.fun,6),round(opt.x,5),'target',score(Des(x,opt.x)@C))
# Supervised lower bounds from low-degree known structure; no hyperparameter chosen on official split.
for deg in range(1,6):
 R=[]
 for j in folds:
  i=np.setdiff1d(np.arange(n),j); D=poly(x,deg); C=np.linalg.lstsq(D[i],tar[i],rcond=None)[0]; e=abs(D[j]@C-tar[j]);R.append([e[:,0].mean(),e[:,1:].mean()]);
 print('supervised poly',deg,R)
