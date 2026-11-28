exec(open('/tmp/discover05.py').read().split('for mode in')[0])
from itertools import combinations_with_replacement
np.set_printoptions(precision=6,suppress=True)
i=slice(0,56); j=slice(56,None)

def feat(x,p):
 d,powr,l,a,f,c,h=p
 return aff(x**3/(1+d*np.abs(x)**powr)+l*x+a*np.sin(f*x)+c*np.sin(h*x))

def panel(p):
 a=feat(U,p); op=ls(a,V); return (a@op-V).ravel(),op

def scalar(z,kind):
 l=aff(z)
 a,b,c,d=z.T
 lookup={'linear':l,'paired':np.c_[l,a*c+b*d],'separate':np.c_[l,a*c,b*d],'quadratic':np.c_[l,*[z[:,k]*z[:,l] for k,l in combinations_with_replacement(range(4),2)]]}
 return lookup[kind]

def fit_y(d,y,mode):
 ini=ls(d,y)
 if mode=='plain':return lambda q:q@ini
 def basis(h):
  if mode=='atan':return np.c_[np.ones(len(h)),h,np.arctan(h)]
  if mode=='tanh':return np.c_[np.ones(len(h)),h,np.tanh(h)]
  if mode=='sine':return np.c_[np.ones(len(h)),h,np.sin(h)]
  return np.c_[np.ones(len(h)),h,h**3]
 def res(p):
  a=basis(d@p);return a@ls(a,y)-y
 opt=min([least_squares(res,c*ini,max_nfev=180) for c in (.5,1,2)],key=lambda o:o.cost)
 p=opt.x; coef=ls(basis(d@p),y)
 return lambda q:basis(q@p)@coef

sol=[]
for f,h in [(4,9),(7,12),(9,14),(11,16),(14,20),(18,24),(9,23),(12,26)]:
 o=least_squares(lambda p:panel(p)[0],[.8,2,-.09,.009,f,.005,h],bounds=([.001,.5,-2,-1,1,-1,1],[8,5,2,1,34,1,34]),max_nfev=220)
 sol.append(o)
o=min(sol,key=lambda o:o.cost); p=o.x; r,op=panel(p); z=feat(X,p)@op
print('B',np.std(r),p,'time',times)
for k in ('linear','paired','separate','quadratic'):
 for mode in ('plain','atan','tanh','sine','cube'):
  d=scalar(z,k); f=fit_y(d[i],y[i],mode); print(k,mode,np.mean(abs(f(d[i])-y[i])),np.mean(abs(f(d[j])-y[j])))
A=aff(z); cf=ls(A[i],T[i]); sv=np.linalg.svd(A[i]@cf,full_matrices=False); print('fitted singulars',sv[1]); print('top temporal ratio',sv[2][0]/sv[2][0,-1]); print('second temporal vector',sv[2][1]); print('raw traj',np.mean(abs(A[j]@cf-T[j])), 'per timestamp',np.mean(abs(A[j]@cf-T[j]),axis=0))
for mode in ('exp-exp','log-exp','t-exp','t-log'):
 def timebasis(p):
  def term(k,r):return {'exp':lambda:-np.expm1(-r*times),'log':lambda:np.log1p(r*times),'t':lambda:times}[k]()
  return np.array([term(k,r) for k,r in zip(mode.split('-'),p)]).T
 def err(p):
  b=timebasis(p); fitted=A[i]@cf; return (fitted@b@np.linalg.pinv(b)-fitted).ravel()
 o=least_squares(err,[.5,1.5],bounds=(.001,30)); b=timebasis(o.x); pred=A[j]@cf@b@np.linalg.pinv(b); print(mode,o.x,np.mean(abs(pred-T[j])),np.linalg.norm(o.fun))
