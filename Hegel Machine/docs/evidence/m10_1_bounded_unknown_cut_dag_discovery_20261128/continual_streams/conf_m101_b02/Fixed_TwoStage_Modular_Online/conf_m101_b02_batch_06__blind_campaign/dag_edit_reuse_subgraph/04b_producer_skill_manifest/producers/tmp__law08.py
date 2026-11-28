import pickle,numpy as np
from scipy.optimize import minimize_scalar, least_squares
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
fit=lambda x,y:np.linalg.lstsq(x,y,rcond=None)[0]
one=lambda x:np.ones((len(x),1))
pairs=np.array(list(combinations(range(4),2))).T
cross=lambda x:x[:,pairs[0]]*x[:,pairs[1]]
lin=lambda x:np.c_[one(x),x]
quad=lambda x:np.c_[lin(x),x*x,cross(x)]
def loo(a,y):
 p=np.linalg.pinv(a);return (y-a@(p@y))/(1-np.sum(a*p.T,axis=1)[:,None])
pa=s['panel_a'];u=pa['u'];v=pa['v'];a=lin(u);w=np.ones(len(u))
for _ in range(10):
 r=np.linalg.norm(v-a@fit(a*w[:,None],v*w[:,None]),axis=1); w=1/np.sqrt(1+(2*r/np.median(r))**2)
k=np.argsort(r)[:-3];u,v=u[k],v[k];print('A full cubic coefficients');print(np.round(fit(np.c_[quad(u),u**3],v),4))
def ab(x,j):
 h=x[:,j]; q=x[:,(j+1)%4];return np.c_[one(x),h,q,h*q,h**3,q**3]
ac=np.array([fit(ab(u,j),v[:,j]) for j in range(4)]);print('A cyclic coefficients',np.round(ac,5));print('A cyclic residual',np.sqrt(np.mean([(ab(u,j)@ac[j]-v[:,j])**2 for j in range(4)])))
def af(x):return np.column_stack([ab(x,j)@ac[j] for j in range(4)])
X=np.r_[s['train']['x'],s['test']['x']];z=np.stack([least_squares(lambda t:af(t[None])[0]-x,x).x for x in X]);print('inverse residual, ranges',np.max(abs(af(z)-X)),z.min(),z.max())
def assay(p,key,f,x):
 p=s[p];return f(x)@fit(f(p['u']),p[key])
c1=lambda x:np.c_[one(x),cross(x)[:,[0,2,3,5]]]
for panel,key in [('panel_c','v1'),('panel_c','v2'),('panel_b','v')]:
 u=s[panel]['u'];v=s[panel][key];print(panel,key,'quadratic coef\n',fit(quad(u),v).round(5));print('C1 candidate',np.sqrt(np.mean(loo(c1(u),v)**2)))
 base=lin(u)
 opts=[]
 for name,func in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda t:t/(1+t*t)),('r2',lambda t:t*t/(1+t*t)),('rat3',lambda t:t**3/(1+t*t)),('sqrt',lambda t:t/np.sqrt(1+t*t)),('sinh',np.sinh),('exp',lambda t:np.exp(-t*t))]:
  for nm,base in [('linear',lin(u)),('cubic',np.c_[lin(u),u**3]),('square',np.c_[lin(u),u*u]),('cross',np.c_[lin(u),cross(u)])]:
   def score(w):return np.mean(loo(np.c_[base,func(u*w)],v)**2)
   rr=minimize_scalar(score,bounds=(.15,3.5),method='bounded');opts.append((rr.fun**.5,name,nm,rr.x))
 for nm,A in [('cubes',np.c_[lin(u),u**3]),('cubes+cross',np.c_[lin(u),u**3,cross(u)]),('cubes+fifth',np.c_[lin(u),u**3,u**5]),('cubes+fifth+7',np.c_[lin(u),u**3,u**5,u**7]),('cubes+square',np.c_[lin(u),u**3,u**2])]: opts.append((np.sqrt(np.mean(loo(A,v)**2)),nm,'',0))
 print('best',sorted(opts)[:12]); print('cubes+fifth coef',fit(np.c_[lin(u),u**3,u**5],v).round(5))
# Screen smooth regressors for evidence of a low-dimensional target or a sparse feature law.
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import PolynomialFeatures
Y=np.c_[s['train']['y'],s['train']['traj']];n=len(Y);folds=[np.arange(52,72),*np.array_split(np.random.default_rng(675).permutation(n),5)]
for name,x in [('raw',X),('inverse',z)]:
 for gam in [.04,.15,.4,1.0]:
  for penalty in [.00001,.001,.02]:
   vals=[]
   for va in folds:
    tr=np.setdiff1d(np.arange(n),va); p=KernelRidge(alpha=penalty,kernel='rbf',gamma=gam).fit(x[tr],Y[tr]).predict(x[va]); e=abs(Y[va]-p);vals.append([e[:,0].mean(),e[:,1:].mean()])
   print('rbf',name,gam,penalty,np.round(vals[0],4),np.round(np.max(vals,0),4))
 for deg in (2,3,4):
  g=PolynomialFeatures(deg).fit_transform(x); p=fit(g[:n],Y); print('poly train',name,deg,np.mean(abs(g[:n]@p-Y),0).round(4))
