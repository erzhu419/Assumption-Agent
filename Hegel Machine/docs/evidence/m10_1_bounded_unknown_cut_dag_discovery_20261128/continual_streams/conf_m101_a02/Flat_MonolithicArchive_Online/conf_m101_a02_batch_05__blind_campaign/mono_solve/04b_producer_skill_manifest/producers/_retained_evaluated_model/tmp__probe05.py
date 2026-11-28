import numpy as np,pickle
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def A(x):return np.c_[np.ones(len(x)),x]
def fit(x,y):return np.linalg.lstsq(x,y,rcond=None)[0]
def e(x,y):return np.sqrt(np.mean((A(x)@fit(A(x),y)-y)**2))
u,v=s['panel_a'].values(); good=np.linalg.norm(u-v,axis=1)<4*np.median(np.linalg.norm(u-v,axis=1)); ab=np.array([fit(A(a[good]),b[good]) for a,b in zip(u.T,v.T)]).T
x=(s['train']['x']-ab[0])/ab[1];Y=s['train']['y'];T=s['train']['traj']; Z={}
trans={ 'dog':(lambda x,k,a:x*(1-a*np.exp(-k*x*x)),[1,1]),'sin':(lambda x,k:np.sin(k*x),[1]),'tanh':(lambda x,k:np.tanh(k*x),[1]),'sinh':(lambda x,k:np.sinh(k*x),[1]),'rational':(lambda x,k:x/(1+k*x*x),[1]),'cubic':(lambda x,k:x+k*x**3,[.3]),'quad':(lambda x,k:x+k*x*x,[.3]),'x+sin':(lambda x,k,a:x+a*np.sin(k*x),[2,.5]),'x+tanh':(lambda x,k,a:x+a*np.tanh(k*x),[1,.5]),'x+rational':(lambda x,k,a:x+a*x/(1+k*x*x),[1,.5]),'x2+cubic':(lambda x,k,a:x+k*x*x+a*x**3,[.3,.3])}
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u'];v=s[pn][vn]; L=[]; print('\n',pn,vn,'v range',v.min(),v.max())
 for name,(f,start) in trans.items():
  def proj(q):
   d=A(f(u,*q));return d@fit(d,v)-v
  q=least_squares(lambda q:proj(q).ravel(),start,max_nfev=300).x
  L.append((np.std(proj(q)),name,q));Z[pn[-1]+vn+name]=lambda x,f=f,q=q,c=fit(A(f(u,*q)),v):A(f(x,*q))@c
 for name,d in [('linear',A(u)),('linear+sq',np.c_[A(u),u**2]),('linear+cube',np.c_[A(u),u**3]),('linear+sq+cube',np.c_[A(u),u**2,u**3])]:
  c=fit(d,v);print(name,np.std(d@c-v),c[5:].round(4));
 for item in sorted(L)[:6]: print(item)
Z['raw']=lambda a:a
from itertools import combinations_with_replacement,product
def scores(z):
 ans=[]
 for names in ['aff','poly2','poly3','pair','archive','sin','rich']:
  a,b,c,d=z.T; pair=a*c+b*d; contrast=a+b-c-d
  if names=='aff': D=A(z)
  if names=='poly2':D=np.column_stack([np.ones(len(z))]+[z[:,j].prod(1) for k in [1,2] for j in combinations_with_replacement(range(4),k)])
  if names=='poly3':D=np.column_stack([np.ones(len(z))]+[z[:,j].prod(1) for k in [1,2,3] for j in combinations_with_replacement(range(4),k)])
  if names=='pair':D=np.c_[A(z),pair]
  if names=='archive':D=np.c_[A(z),pair,np.sin(1.7*contrast),np.tanh(contrast),pair/(1+contrast**2)]
  if names=='sin':D=np.c_[A(z),pair,np.sin(1.7*z)]
  if names=='rich':D=np.c_[A(z),pair,np.sin(1.7*z),np.sin(1.7*contrast),np.tanh(contrast),pair/(1+contrast**2)]
  ans.append((np.mean(abs(D[52:]@fit(D[:52],Y[:52])-Y[52:])),names))
 return sorted(ans)[:2],np.mean(abs(A(z)[52:]@fit(A(z)[:52],T[:52])-T[52:]))
for k,f in Z.items():
 z=f(x); r,t=scores(z); print(k,[(round(v,5),n) for v,n in r],round(t,5))
