exec(open('/tmp/explore.py').read().split('for k in')[0]); d=s['train']; x,y,z=d['x'],d['y'],d['traj']; from scipy.optimize import least_squares; from sklearn.preprocessing import PolynomialFeatures
splits=[(np.arange(52),np.arange(52,72))]+[(np.setdiff1d(np.arange(72),j),j) for j in np.array_split(np.random.default_rng(0).permutation(72),4)]; opt=[]
for k in (1,2,3,4):
 P=PolynomialFeatures(k).fit_transform(x)
 for ridge in (0,.001,.02,.1,1):
  opt.append((f'poly{k}/{ridge}',lambda i,j,P=P,ridge=ridge: P[j]@np.linalg.lstsq(np.r_[P[i],np.sqrt(ridge)*np.eye(P.shape[1])],np.r_[y[i],np.zeros(P.shape[1])],rcond=None)[0]))
X=np.c_[np.ones(len(x)),x];
for link in ['tanh','atan','sin','softsign','sqrt','cubic']:
 for extra in [0,1,2]:
  U=[np.zeros((len(x),0)),x,PolynomialFeatures(2,include_bias=False).fit_transform(x)][extra]
  f={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'softsign':lambda t:t/(1+abs(t)), 'sqrt':lambda t:t/np.sqrt(1+t*t),'cubic':lambda t:t-t**3/3}[link]
  def pred(i,j,U=U,f=f):
   b=np.linalg.lstsq(X[i],y[i],rcond=None)[0]; p0=np.r_[2,0,b/2,np.zeros(U.shape[1])]; fn=lambda p,k:p[0]*f(X[k]@p[2:7])+p[1]+U[k]@p[7:]; p=least_squares(lambda p:np.r_[fn(p,i)-y[i],.02*p[7:]],p0,max_nfev=250).x; return fn(p,j)
  opt.append((f'{link}/{extra}',pred))
for name,fit in opt:
 m=np.array([np.mean(abs(fit(i,j)-y[j])) for i,j in splits]); print(name,np.round(m,5),np.round(m.mean(),5))
b=np.linalg.lstsq(X,y,rcond=None)[0]; print('linear',b); q=X@b; print(np.c_[q,y,y-q][np.argsort(q)][::2]); print('trajectory linear',np.linalg.lstsq(X,z,rcond=None)[0])
