import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=52
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
af=lambda x: np.c_[np.ones(len(x)),x]
a=s['panel_a']; d=af(a['u']); h=fit(d,a['v']); k=np.argsort(np.linalg.norm(d@h-a['v'],axis=1))[:41]; h=fit(d[k],a['v'][k]); z=(x-h[0])/np.diag(h[1:]); b=s['panel_b']; H=fit(af(b['u']),b['v']); w=af(z)@H
c=s['panel_c']; u=c['u']; Cs={}
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
I=np.arange(52); J=np.arange(52,72)
T=s['train']['traj']; L=af(w); pred=L@fit(L[I],T[I]); print('summary',s['train']['summary'],'y',np.quantile(y,[0,.25,.5,.75,1])); print('ranges',np.max(abs(z),0),np.max(abs(w),0));
for name,q in [('cal',z),('B',w)]:
 P=PolynomialFeatures(3).fit(q); F=P.transform(q); m=P.powers_.sum(1); names=P.get_feature_names_out(); print(name,'quadratic',list(zip(names[m<=2],fit(F[I][:,m<=2],y[I]).round(3))));
 D=[(f'{i}',q[:,i]) for i in range(4)]; D += [(f'{i}{sgn:+g}*{j}',q[:,i]+sgn*q[:,j]) for i,j in combinations(range(4),2) for sgn in [-2,-1,-.5,.5,1,2]]; D += [('endpoint',pred[:,-1]),('lin_y',af(q)@fit(af(q)[I],y[I]))]; D += [(f'p{i}{j}',q[:,i]*q[:,j]) for i,j in combinations_with_replacement(range(4),2)]
 rows=[]
 for tag,t in D:
  for shape,di in [('square',t*t),('abs',abs(t))]:
   for k in [.03,.1,.2,.4,.8,1.7]:
    d=1+k*di
    for b,A in [('entire2',F[:,m<=2]/d[:,None]),('entire3',F/d[:,None]),('linear+interact',np.c_[af(q),F[:,m==2]/d[:,None]]),('linear+cubic',np.c_[af(q),F[:,m>=2]/d[:,None]])]:
     r=A@fit(A[I],y[I])-y; rows.append((np.sqrt(np.mean(r[I]**2)),np.mean(abs(r[J])),tag,shape,k,b))
 for b in ['entire2','entire3','linear+interact','linear+cubic']:
  R=[r for r in rows if r[-1]==b]; print(b,'best training',sorted(R)[:4]); print(b,'best validation (exploratory)',sorted(R,key=lambda r:r[1])[:3]);
 for deg in [2,3,4]:
  A=PolynomialFeatures(deg).fit_transform(q); Q=PolynomialFeatures(2,include_bias=False).fit_transform(q); V=np.c_[A,-Q*y[:,None]]; C=fit(V[I],y[I]); den=1+Q@C[A.shape[1]:]; r=A@C[:A.shape[1]]/den-y; print('free rational',deg,np.std(r[I]),np.mean(abs(r[J])),den.min(),C[A.shape[1]:].round(3));
