import pickle,numpy as np
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; x=s['train']['x']; y=s['train']['y']; N=52
fit=lambda a,b: np.linalg.lstsq(a,b,rcond=None)[0]
af=lambda x: np.c_[np.ones(len(x)),x]
a=s['panel_a']; d=af(a['u']); h=fit(d,a['v']); k=np.argsort(np.linalg.norm(d@h-a['v'],axis=1))[:41]; h=fit(d[k],a['v'][k]); z=(x-h[0])/np.diag(h[1:]); b=s['panel_b']; H=fit(af(b['u']),b['v']); w=af(z)@H
c=s['panel_c']; u=c['u']; Cs={}
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge
from itertools import combinations_with_replacement
ops={}
for key in ('v1','v2'):
 v=c[key]; vals=[]
 for typ in ('tanh','sin','cos','softsign'):
  fun=getattr(np,typ,None) or (lambda a:a/(1+abs(a)))
  def obj(k):
   f=np.c_[af(u),fun(k*u)]; return np.linalg.norm(v-f@fit(f,v))
  r=minimize_scalar(obj,bounds=(.15,6),method='bounded'); vals.append((r.fun,r.x,typ,fun))
 for e,k,typ,_ in sorted(vals): print(key,typ,round(k,5),round(e,7))
 e,k,typ,fun=min(vals); f=np.c_[af(u),fun(k*u)]; coef=fit(f,v); print('coef',coef.round(4)); ops[key]=(lambda q,k=k,fun=fun,coef=coef:np.c_[af(q),fun(k*q)]@coef)
print('B coeff',H.round(5));T=s['train']['traj']; print('traj affine B coeff',fit(af(w[:N]),T[:N]).round(4)); print('traj affine B MAE',abs(af(w[N:])@fit(af(w[:N]),T[:N])-T[N:]).mean(0))
for name,q in [('raw',x),('cal',z),('B',w),('Braw',af(x)@H),('Binv',(z-H[0])@np.linalg.inv(H[1:]))]:
 V1,V2=ops['v1'](q),ops['v2'](q)
 print('input',name)
 for tag,F in [('v1',V1),('v2',V2),('q+v1',np.c_[q,V1]),('q+v2',np.c_[q,V2]),('q+v1+v2',np.c_[q,V1,V2]),('q+v1*v2',np.c_[q,V1*V2]),('q+v1+v2+v1*v2',np.c_[q,V1,V2,V1*V2]),('q+v1+v2+outer',np.c_[q,V1,V2,(V1[:,:,None]*V2[:,None,:]).reshape(len(q),16)])]:
  F=af(F); err=abs(F[N:]@fit(F[:N],y[:N])-y[N:]); res=F[:N]@fit(F[:N],y[:N])-y[:N]; print(tag,round(err.mean(),5),round(np.sqrt(np.mean(res**2)),5))
 for tag,r in [('v1',V1),('v2',V2),('v1+v2',V1+V2),('v1-v2',V1-V2),('v1*v2',V1*V2)]:
  if tag=='v1*v2':continue
  poly=PolynomialFeatures(2).fit_transform(r); F=np.c_[af(q),poly[:,1:]]; print('poly',tag,round(abs(F[N:]@fit(F[:N],y[:N])-y[N:]).mean(),5))
