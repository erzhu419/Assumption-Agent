exec(open('/tmp/graph7.py').read().split('for dg in (1,2,3):')[0]); from scipy.optimize import least_squares
p=s['panel_b']; U=p['u']; V=p['v'];
# fit common componentwise activation and profile the affine forward map
fs={'rat':lambda x,w:x/(1+w[0]*x*x)+w[1]*x,'rat3':lambda x,w:(x+w[1]*x**3)/(1+w[0]*x*x+w[2]*x**4),'tanh':lambda x,w:np.tanh(w[0]*x)+w[1]*x,'sin':lambda x,w:np.sin(w[0]*x)+w[1]*x,'atan':lambda x,w:np.arctan(w[0]*x)+w[1]*x,'sqrt':lambda x,w:x/np.sqrt(1+w[0]*x*x)+w[1]*x,'rat2':lambda x,w:x/(1+w[0]*x*x)+w[1]*x/(1+w[2]*x*x)}
for name,f in fs.items():
 n=3 if name in ('rat3','rat2') else 2
 def err(w):
  X=aff(f(U,w)); return (X@ls(X,V)-V).ravel()
 r=min([least_squares(err,[a,b,.25][:n],bounds=([.001,-10,.001][:n],[12,10,12][:n]),max_nfev=120) for a in [.3,1.3,2.9] for b in [-.45,0,.3]],key=lambda r:np.linalg.norm(r.fun)); print('B shared',name,r.x,np.std(r.fun)); c=ls(aff(f(U,r.x)),V);D['B_'+name]=aff(f(z,r.x))@c
 def err2(w):
  X=aff(f(U,w)) if name!='rat' else aff(U/(1+U*U*w));return (X@ls(X,V)-V).ravel()
 if name=='rat':
  r=least_squares(err2,np.ones(4),bounds=(.001,20));print('B per coordinate',r.x,np.std(r.fun));X=aff(U/(1+U*U*r.x)); D['B_sep']=aff(z/(1+z*z*r.x))@ls(X,V)
for name,w in [('rat',.22),('rat',1.25),('atan',1.25),('atan',2.449),('sin',1.25),('tanh',1.25),('tanh',2.5),('sqrt',1.25),('sqrt',4.)]:D[f'{name}{w}']=fs[name](z,[w,0.])
# Features combining the native coordinates and componentwise nonlinearities, plus structured polynomials.
from itertools import combinations
Y0=Y[:,0]; T=Y[:,1:]; t=s['eval_times']; _,_,vh=np.linalg.svd(T-T[:52].mean(0),full_matrices=False); amp=T@vh[0]; scores=[]
for key in ['z','B(z)','C1(z)','C2(z)','C1(B(z))','C2(B(z))']+[k for k in D if '(' not in k and k not in ('z','x')]:
 h=D[key]; mul=np.array([h[:,i]*h[:,j] for i,j in combinations(range(4),2)]).T; cyc=h*roll(h)
 for tag,X in [('lin',aff(h)),('quadr',np.c_[aff(h),h*h,mul]),('offdiag',np.c_[aff(h),mul]),('cyc',np.c_[aff(h),cyc]),('z+quad',np.c_[aff(z),h,h*h,mul]),('z+cyc',np.c_[aff(z),h,cyc]),('z+offdiag',np.c_[aff(z),h,mul]),('cross',np.c_[aff(h),h*h,mul,z,z*h]),('cubicRidge',PolynomialFeatures(3).fit_transform(h))]:
  for reg in ([0,.03,.3] if tag=='cubicRidge' else [0]):
   A=X[:52]; c=ls(A,Y[:52]) if reg==0 else np.linalg.solve(A.T@A+reg*np.diag(np.r_[0,np.maximum(np.sum(A[:,1:]**2,axis=0),1e-8)]),A.T@Y[:52]); E=abs(X[52:]@c-Y[52:]);scores.append((E[:,0].mean(),E[:,1:].mean(),key,tag,reg))
print('top scalar',sorted(scores)[:16]);print('top trajectory',sorted(scores,key=lambda a:a[1])[:16]);
for key in ['z','B(z)','B_rat','B_sep','C2(B(z))']:
 h=D[key]; p=PolynomialFeatures(2);A=p.fit_transform(h); c=ls(A[:52],np.c_[Y0,amp][:52]);print(key,np.c_[p.get_feature_names_out(),c]);print('quant',np.quantile(h,[0,.5,1],axis=0))
