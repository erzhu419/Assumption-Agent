exec(open('/tmp/graph101.py').read().split('for key,p,v in')[0])
from scipy.special import erf
u=s['panel_b']['u']; v=s['panel_b']['v']; w=(x-off)/sc
fams={'tanh':np.tanh,'atan':np.arctan,'sin':np.sin,'rat':lambda h:h/(1+h*h),'soft':lambda h:h/np.sqrt(1+h*h),'erf':erf,'asinh':np.arcsinh,'oddexp':lambda h:h*np.exp(-h*h),'expcube':lambda h:h*(1-np.exp(-h*h)),'oddsig':lambda h:h*np.tanh(abs(h)),'cube':lambda h:h**3,'quint':lambda h:h**5,'signedquad':lambda h:h*abs(h),'sinh':np.sinh,'oddlog':lambda h:h*np.log1p(h*h),'pow':lambda h:h*np.log1p(abs(h)),'cubicrat':lambda h:h**3/(1+abs(h)**3),'cubictanh':lambda h:np.tanh(h**3)}
models=[]
for name,f in fams.items():
 for mode in ['mixed','shared','pure']:
  def design(q,r,f=f,mode=mode):
   h=f(q*r[0]); return np.c_[O(q),h] if mode=='mixed' else O(h+(r[1]*q if mode=='shared' else 0))
  lo=[.1,-9] if mode=='shared' else [.1]; hi=[5,9] if mode=='shared' else [5]; starts=[[k,0] if mode=='shared' else [k] for k in [.5,1.4,2.8]]
  def fun(r):
   d=design(u,r); return (d@L(d,v)-v).ravel()
  fit=min((least_squares(fun,p,bounds=(lo,hi),max_nfev=100) for p in starts),key=lambda o:o.cost); d=design(u,fit.x); C=L(d,v); z=design(w,fit.x)@C
  pair=z[:,0]*z[:,2]+z[:,1]*z[:,3]; A=np.c_[O(z),pair]; B=O(z); errs=[]
  for i,j in folds:
   errs.append([np.mean(abs(A[j]@L(A[i],Y[i,0])-Y[j,0])),np.mean(abs(B[j]@L(B[i],Y[i,1:])-Y[j,1:]))])
  models.append((np.sqrt(np.mean(fit.fun**2)),name,mode,fit.x,np.array(errs),C,z))
for by in [lambda m:m[0],lambda m:m[4].mean()]:
 print('rank');
 for e,n,mode,r,err,C,z in sorted(models,key=by)[:10]:print(n,mode,r,'panel RMSE',round(e,7),'56/16',err[0], 'worst',err.max(0))
for e,n,mode,r,err,C,z in sorted(models,key=lambda m:m[0])[:3]:
 print(n,mode,'C',C); print('scalar fit',L(np.c_[O(z),*[z[:,i]*z[:,j] for i,j in combinations(range(4),2)]],Y[:,0])); print('time coefficients',L(O(z),Y[:,1:]))
Z=Y[:,1:]; F={'exp':lambda k:-np.expm1(-k*t),'rat':lambda k:t/(1+k*t),'log':lambda k:np.log1p(k*t),'texp':lambda k:t*np.exp(-k*t),'rat2':lambda k:t/(1+k*t)**2,'rat3':lambda k:t/(1+k*t)**3,'expstep':lambda k:t*(-np.expm1(-k*t)),'hill':lambda k:t*t/(1+k*t*t),'sin':lambda k:np.sin(k*t),'tanh':lambda k:np.tanh(k*t),'gaus':lambda k:t*np.exp(-k*t*t),'exp2':lambda k:-np.expm1(-k*t*t)}
out=[]
for a,b in list(combinations(F,2))+[(a,a) for a in F]:
 def fun(k):
  T=np.column_stack((F[a](k[0]),F[b](k[1]))); return (T@L(T,Z.T)-Z.T).ravel()
 fit=min((least_squares(fun,p,bounds=(.005,8),max_nfev=100) for p in [[.5,1.5],[1.5,.5],[.2,3]]),key=lambda f:f.cost); out.append((np.sqrt(np.mean(fit.fun**2)),a,b,fit.x))
print('time',sorted(out)[:12])
