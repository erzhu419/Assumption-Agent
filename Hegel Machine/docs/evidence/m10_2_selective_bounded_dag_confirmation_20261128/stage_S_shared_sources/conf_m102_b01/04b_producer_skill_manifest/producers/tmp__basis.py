import numpy as np,pickle,itertools
from scipy.optimize import least_squares
from sklearn.linear_model import Ridge
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study'];x=s['train']['x'];y=s['train']['y'];T=s['train']['traj'];t=s['eval_times']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=None)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
u,v=s['panel_a'].values(); L=ls(aff(u),v)
for _ in range(10):
 i=np.argsort(np.max(abs(aff(u)@L-v),axis=1))[:42];L=ls(aff(u[i]),v[i]);
i=np.max(abs(aff(u)@L-v),axis=1)<.025; L=ls(aff(u[i]),v[i]); z=(x-L[0])/np.diag(L[1:]); b=s['panel_b']; M=ls(aff(b['u']),b['v']); w=aff(z)@M
A=lambda p:np.c_[-np.expm1(-p[0]*t),np.log1p(p[1]*t)]
def er(p):
 a=A(p);return (a@ls(a,T[:52].T)-T[:52].T).ravel()
p=least_squares(er,[1.8,1],bounds=(.0001,20)).x; a=A(p);Y=np.c_[y,ls(a,T.T).T]; print('temporal',p,np.std(er(p)))
def score(F,label,alpha=0):
 F=np.asarray(F); B=ls(F[:52],Y[:52]) if not alpha else Ridge(alpha=alpha,fit_intercept=False).fit(F[:52],Y[:52]).coef_.T; Q=F@B
 return np.r_[np.mean(abs(Q[52:,0]-y[52:])),np.mean(abs(Q[52:,1:]@a.T-T[52:])),np.mean(abs(Q[:52,0]-y[:52])),np.mean(abs(Q[:52,1:]@a.T-T[:52:]))],label,B
functions={'tanh':np.tanh,'sin':np.sin,'atan':np.arctan,'rational':lambda a:a/(1+a*a),'cube':lambda a:a**3,'square':np.square,'abs':np.abs,'exp':lambda a:np.exp(-a*a),'id':lambda a:a}
res=[]
for n,W in [('x',x),('z',z),('w',w),('braw',aff(x)@M),('wno',w-M[0])]:
 for label,f in functions.items():
  for k in ([.4,.7,1,1.3,1.6,2,2.5,3] if label not in ['cube','square','abs','id'] else [1]):
   for add in (False,True):
    F=np.c_[aff(W),f(W*k)] if add else aff(f(W*k)); r=score(F,(n,label,k,add));res.append(r)
 for label,f in functions.items():
  H=f(W); F=np.c_[aff(W),H,W**2,*[W[:,i]*W[:,j] for i,j in itertools.combinations(range(4),2)]];res.append(score(F,(n,label,'pair')))
print('best scalar');
for e,l,b in sorted(res,key=lambda s:s[0][0])[:12]: print(l,e)
print('best traj');
for e,l,b in sorted(res,key=lambda s:s[0][1])[:12]: print(l,e)
print('additive w components')
for fn in [np.tanh,np.sin,np.arctan,lambda a:a/(1+a*a),lambda a:a**3]:
 F=np.c_[aff(w),fn(w)]; R=score(F,fn);print(R[0],R[2].T)
c=s['panel_c']; u=c['u'];print('C forms');
for k in ['v1','v2']:
 for H in [np.c_[aff(u),u**2,u**3],np.c_[aff(u),np.sin(u),np.cos(u)],np.c_[aff(u),np.tanh(u),u**2]]:
  q=ls(H,c[k]);print(k,np.std(H@q-c[k]))
