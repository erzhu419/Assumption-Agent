exec(open('/tmp/discover.py').read().split('U,D,V=')[0])
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
pc=s['panel_c']; u=pc['u']; v=pc['v1']; q=coords['AinvB']; n=len(u)
# Mixing weights from full panel fit; then use the same mixing for every componentwise nonlinearity.
starts={ 'T':[1.6], 'LT':[.2,1.6], 'X3T':[.2,1.6], 'TT':[.15,1.6,4.], 'LTT':[.15,1.6,.12,4.], 'LTR':[.15,1.6,.12,4.], 'LTS':[.15,1.6,.12,2.], 'LTX3':[.15,1.6,.01], 'Twarp':[1.6,.15], 'TwarpL':[.2,1.6,.15], 'Tsin':[1.6,.1], 'erfL':[.15,1.2], 'atanL':[.15,1.5], 'LTden':[.15,1.6,.05], 'TwarpR':[1.6,.15], 'TTwarp':[.15,1.6,4.,.1], 'RT':[.2,1.6,4.], 'ST':[.2,1.6,2.]}
from scipy.special import erf
fun={ 'T':lambda x,a:np.tanh(a*x), 'LT':lambda x,b,a:b*x+np.tanh(a*x),'X3T':lambda x,b,a:b*x**3+np.tanh(a*x),'TT':lambda x,b,a,c:np.tanh(a*x)+b*np.tanh(c*x),'LTT':lambda x,b,a,d,c:b*x+np.tanh(a*x)+d*np.tanh(c*x),'LTR':lambda x,b,a,d,c:b*x+np.tanh(a*x)+d*c*x/(1+(c*x)**2),'LTS':lambda x,b,a,d,c:b*x+np.tanh(a*x)+d*np.sin(c*x),'LTX3':lambda x,b,a,d:b*x+np.tanh(a*x)+d*x**3, 'Twarp':lambda x,a,b:np.tanh(a*x+b*x**3),'TwarpL':lambda x,d,a,b:d*x+np.tanh(a*x+b*x**3),'Tsin':lambda x,a,b:np.tanh(a*x)+b*np.sin(x),'erfL':lambda x,b,a:b*x+erf(a*x),'atanL':lambda x,b,a:b*x+np.arctan(a*x),'LTden':lambda x,b,a,d:b*x+np.tanh(a*x)/(1+d*x*x),'TwarpR':lambda x,a,b:np.tanh(a*x/(1+b*x*x)), 'TTwarp': lambda x,b,a,c,d:np.tanh(a*x)+b*np.tanh(c*x+d*x**3), 'RT':lambda x,b,a,c:np.tanh(a*x)+b*c*x/(1+(c*x)**2), 'ST':lambda x,b,a,c:np.tanh(a*x)+b*np.sin(c*x)}
def fit(name,mask):
 f=fun[name]; X=u[mask]; Y=v[mask];
 def fit_at(p):
  Z=np.c_[np.ones(len(X)),f(X,*p)]; W=np.linalg.lstsq(Z,Y,rcond=None)[0]; return (Z@W-Y).ravel(),W
 start=starts[name]; lo=[.05 if v0>=1 else (-.05 if name in ('LTden','TwarpR') and j==len(start)-1 else -2) for j,v0 in enumerate(start)]; hi=[9 if v0>=1 else 2 for v0 in start]; p=least_squares(lambda p:fit_at(p)[0],start,bounds=(lo,hi),max_nfev=150).x
 return p,fit_at(p)[1]
def evalc(f,p,w,X):return np.c_[np.ones(len(X)),f(X,*p)]@w
def time(p):return -np.expm1(-t[:,None]*p)
features=lambda z:np.column_stack([np.prod(z[:,i],axis=1) for k in range(3) for i in combinations_with_replacement(range(4),k)])
folds=[np.arange(len(y))%4==i for i in range(4)]+[np.arange(len(y))>=56]; rows=[]
for name,f in fun.items():
 p,w=fit(name,np.ones(n,bool)); pp=evalc(f,p,w,q); ec=[]; ey=[]; et=[]
 for k in range(4):
  m=np.arange(n)%4==k; p0,w0=fit(name,~m);ec.append(np.mean(abs(evalc(f,p0,w0,u[m])-v[m])))
 for m in folds:
  T=time(least_squares(lambda p:(tr[~m]-tr[~m]@np.linalg.pinv(time(p).T)@time(p).T).ravel(),[.4,1.8],bounds=(.05,10)).x); D=features(pp); yh=D[m]@np.linalg.lstsq(D[~m],y[~m],rcond=None)[0]; A=np.c_[np.ones(len(pp)),pp]; th=A[m]@np.linalg.lstsq(A[~m],tr[~m]@np.linalg.pinv(T.T),rcond=None)[0]@T.T; ey.append(np.mean(abs(yh-y[m])));et.append(np.mean(abs(th-tr[m]),axis=0).max())
 rows.append((np.mean(ec),name,p,max(ec),np.mean(ey),max(ey),max(et))); 
for row in sorted(rows,key=lambda r:r[0]):print(row)
best=min(rows,key=lambda r:r[0])[1]; p,w=fit(best,np.ones(n,bool)); z=evalc(fun[best],p,w,q); D=features(z); print('y coeff',np.linalg.lstsq(D,y,rcond=None)[0]); print('panel C input min/max',u.min(0),u.max(0),'study Q',q.min(0),q.max(0)); A=np.c_[np.ones(len(z)),z]; W=np.linalg.lstsq(A,tr,rcond=None)[0]; print('time coeff',W); print('raw residual per horizon',np.std(tr-A@W,axis=0),'y residual',np.std(y-D@np.linalg.lstsq(D,y,rcond=None)[0])); print('summary computed',np.mean(y),np.std(y),tr[:,-1].mean(),tr[:,-1].std())
