from panels import *
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso, OrthogonalMatchingPursuit
from sklearn.model_selection import KFold
R=dict(np.load('/tmp/reg.npz')); u=R['u']; y=s['train']['y']; z=s['train']['traj']; t=s['eval_times']; q=1-np.exp(-.55*t); amp=z@q/(q@q); Y=np.c_[y,amp]; U,V=[s['panel_b'][k] for k in ('u','v')]
def fit(F,G):
 P=np.linalg.pinv(F); E=V-F@(P@V); return np.sqrt(np.mean((E/(1-np.sum(F*P.T,axis=1))[:,None])**2)),G@(P@V)
for n in range(1,6):
 F=lambda x:np.column_stack([np.ones(len(x))]+[x**k for k in range(1,2*n,2)]); f=F(U); sc,b=fit(f,F(u)); print('B odd',n,sc,np.linalg.lstsq(f,V,rcond=None)[0].round(4) if n==3 else '')
for name,g in [('sin',np.sin),('tanh',np.tanh),('soft',lambda x:x/(1+x*x))]:
 for p in ('cross','sq','triple','none'):
  def f(x,k):return np.column_stack((np.ones(len(x)),x,g(k*x),features(x)[p] if p!='none' else np.empty((len(x),0))))
  opt=min([minimize_scalar(lambda k:fit(f(U,k),f(u,k))[0],bounds=(lo,hi),method='bounded') for lo,hi in [(.15,1.8),(1.8,4.5)]],key=lambda a:a.fun); print('B extra',name,p,round(opt.x,4),round(opt.fun,7))
print('traj time fits'); mean=z.mean(0); o=minimize_scalar(lambda k:np.linalg.norm(mean-(mean@(1-np.exp(-k*t)))/np.sum((1-np.exp(-k*t))**2)*(1-np.exp(-k*t))),bounds=(.1,2),method='bounded');print(o.x,o.fun); q=1-np.exp(-o.x*t); E=z-(z@q/(q@q))[:,None]*q; print('traj residual profile',np.linalg.svd(E,full_matrices=False)[2][:2].round(3),'corr(y)',np.corrcoef(np.c_[y,E].T)[0,1:].round(2)); Y[:,1]=z@q/(q@q)
folds=[(np.arange(52),np.arange(52,72))]+list(KFold(4,shuffle=True,random_state=0).split(u));
# Candidate grammars: cyclic and noncyclic interactions, powers, saturations, and multiplicative gates.
D={}; blocks={k:R[k] for k in ('u','b','c1','c2','x')}; blocks['b2']=np.load('/tmp/bnew.npy')
for k,w in blocks.items():
 for degree in (1,2,3,4):
  p=PolynomialFeatures(degree,include_bias=False); f=p.fit_transform(w); sel=p.powers_.sum(1)==degree
  for n,v in zip(p.get_feature_names_out()[sel],f[:,sel].T):D[k+':'+n]=v
 for j in range(4):
  for n in range(j,4):
   a=w[:,j]; b=w[:,n]; args={'product':a*b,'sum':a+b,'diff':a-b,'sumsq':a*a+b*b}
   for label,arg in args.items():
    for c in (.5,1.,1.7,2.5,4.):
     for typ,f in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('rat',lambda x:x/(1+x*x))]:D[f'{k}:{typ}({c} {label}{j}{n})']=f(c*arg)
for k,w in blocks.items():
 for l,v in blocks.items():
  if k>=l:continue
  for j in range(4):
   for n in range(4):
    a,b=w[:,j],v[:,n];
    for typ,m in [('a*b',a*b),('a*b²',a*b*b),('a²*b',a*a*b),('gate',a/(1+b*b)),('gate2',b/(1+a*a)),('sinprod',np.sin(1.7*a*b)),('sinmul',np.sin(1.7*a)*b),('mulsin',a*np.sin(1.7*b))]:D[f'{k}{j}:{l}{n}:{typ}']=m
names=np.array(list(D)); F=np.column_stack(list(D.values())); keep=F.std(0)>1e-8; F=F[:,keep]; names=names[keep]
for j in range(2):
 for typ,pars in [('omp',[3,5,8,12,18,24]),('lasso',[.001,.003,.008,.015,.03]),('ridge',[.3,3,15,60])]:
  for par in pars:
   scores=[]
   for a,h in folds+[(np.arange(72),np.arange(72))]:
    X=(F-F[a].mean(0))/F[a].std(0); X=np.nan_to_num(X,posinf=0,neginf=0); M=(OrthogonalMatchingPursuit(n_nonzero_coefs=par) if typ=='omp' else Lasso(alpha=par,max_iter=2500,tol=.001) if typ=='lasso' else Ridge(alpha=par,solver='lsqr')).fit(X[a],Y[a,j]); scores.append(np.mean(np.abs(Y[h,j]-M.predict(X[h]))))
   ids=np.argsort(abs(M.coef_))[-6:][::-1]; print(j,typ,par,np.round(scores,4),list(zip(names[ids],M.coef_[ids].round(3))))
np.savez('/tmp/readout7.npz',F=F,names=names,Y=Y,q=q)
