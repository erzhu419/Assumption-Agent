exec(open('/tmp/structure.py').read().split('Y=[]; T=[]')[0])
from scipy.optimize import least_squares
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
np.set_printoptions(precision=4,suppress=True,linewidth=150)
D=m._branch(p,z,2); Cz=Q(z)@coef
Cterms=z[:,[0,0,1,2]]*z[:,[1,3,2,3]]
ts=s['eval_times']; h=lambda w:-np.expm1(-ts[:,None]*w); w=least_squares(lambda w:(h(w)@ls(h(w),t.T)-t.T).ravel(),[1.8,.4],bounds=([1,.1],[3,.95])).x
print('rates',w,'projection MAE',abs(h(w)@ls(h(w),t.T)-t.T).mean()); amp=ls(h(w),t.T).T
for name,zz in [('A',z),('B',coords['B']),('Bi',coords['Binv'])]:
 pol=PolynomialFeatures(4,include_bias=False).fit(zz); P=pol.transform(zz); names=pol.get_feature_names_out(); lens=[4,14,34,69]; C=Q(zz)@coef; d=m._branch(p,zz,2)
 for title,F in [('c,d',ones(np.c_[C,d])),('all cubic+d',ones(np.c_[P[:,:34],d])),('c,d,z*c',ones(np.c_[C,d,(zz[:,:,None]*C[:,None,:]).reshape(n,-1)])),('c,d,same z*c',ones(np.c_[C,d,zz*C])),('c,d,roll z*c',ones(np.c_[C,d,np.roll(zz,1,axis=1)*C])),('c,d,diag',ones(np.c_[C,d,C*d])),('c,d,z*d',ones(np.c_[C,d,(zz[:,:,None]*d[:,None,:]).reshape(n,-1)])),('c,d,z*B',ones(np.c_[C,d,zz*(ones(zz)@B)])),('c,d,c*B',ones(np.c_[C,d,C*(ones(zz)@B)])),('c,d,d*B',ones(np.c_[C,d,d*(ones(zz)@B)]))]:
  if name=='A' or title in ['c,d','all cubic+d','c,d,z*c']:
   E=F@ls(F,np.c_[y,amp])-np.c_[y,amp]; print(name,title,'train',np.mean(abs(E),0),'worst', [round(max(score(F,k)),4) for k in (y,amp[:,0],amp[:,1])])
# Find reproducible sparse residual terms, starting from independently identified branch subspaces.
H0=ones(np.c_[Cz,D]); targets=np.c_[y,amp]; P=PolynomialFeatures(4,include_bias=False).fit_transform(z); pn=PolynomialFeatures(4,include_bias=False).fit(z).get_feature_names_out(); cd=np.c_[Cz,D]; dnames=np.array([f'{k}{i}' for k in 'cd' for i in range(4)]); zd=np.c_[z,cd]; zdnames=[f'z{i}' for i in range(4)]+list(dnames); poly=PolynomialFeatures(2,include_bias=False).fit(zd); X=np.c_[P,poly.transform(zd)]; names=np.r_[pn,poly.get_feature_names_out(zdnames)];
for j in range(3):
 print('TARGET',j)
 for r,v in cuts[:3]:
  H=H0.copy(); selected=[]; measures=[]
  for step in range(14):
   R=X[r]-H[r]@ls(H[r],X[r]); norm=np.linalg.norm(R,axis=0); E=targets[r,j]-H[r]@ls(H[r],targets[r,j]); k=np.argmax(np.where(norm>1e-7,abs(R.T@E)/np.maximum(norm,1e-7),0)); selected.append(names[k]); H=np.c_[H,X[:,k]]; measures.append(round(np.mean(abs(H[v]@ls(H[r],targets[r,j])-targets[v,j])),4))
  print(list(zip(selected,measures)))
