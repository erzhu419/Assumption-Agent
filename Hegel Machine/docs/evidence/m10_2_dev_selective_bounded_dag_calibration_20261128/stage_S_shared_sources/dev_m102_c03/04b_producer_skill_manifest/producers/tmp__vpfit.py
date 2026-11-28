exec(open('/tmp/grammar.py').read().split('rows=[]')[0])
from itertools import combinations
N=52
out=[]
funcs={'tanh':np.tanh,'sin':np.sin,'soft':lambda u:u/(1+abs(u)),'atan':np.arctan,'rat':lambda u:u/(1+u*u),'sig':lambda u:u/np.sqrt(1+u*u)}
for name in ['a','a1','ba','ba1','Ba','Ba1']:
 z=trans[name][:72]; z=z/z[:N].std(0)
 for base in [1,2,3]:
  p=[np.ones(len(z)),z]
  if base>=2:p +=[z*z]
  if base>=3:p +=[z[:,i]*z[:,j] for i,j in combinations(range(4),2)]
  p=np.column_stack(p)
  for j in range(3):
   y=Y[:,j]
   for kind,f in funcs.items():
    for mode in ['sep','one','addpair']:
     def design(q):
      if mode=='sep':F=f(z*q[:4]+q[4:])
      elif mode=='one':F=f(z@q[:4]+q[4])[:,None]
      else:
       i,k=pair; zz=z[:,[i,k]];F=np.c_[f(z*q[:4]+q[4:8]),f(zz@q[8:10]+q[10])]
      return np.column_stack([p,F])
     def obj(q):
      W=design(q)[:N]; return W@fit(W,y[:N])-y[:N]
     if mode=='one':
      w=fit(p[:N],y[:N])[1:5]; w=w/max(np.linalg.norm(w),1e-5); starts=[np.r_[w,.1],np.r_[2*w,.3],np.r_[.6*w,-.4]]; pairs=[None]
     else:
      starts=[np.r_[np.full(4,a),np.full(4,b)] for a,b in [(1.,0.),(.6,.05),(1.5,-.15)]];pairs=[None] if mode=='sep' else combinations(range(4),2)
     if mode=='addpair' and (name not in ['a','a1'] or kind not in ['tanh','soft','sin'] or base==2):continue
     for pair in pairs:
      best=None
      for q in starts:
       if mode=='addpair':q=np.r_[q,1.,-1.,.15]
       r=least_squares(obj,q,bounds=(-5,5),max_nfev=45)
       if best is None or np.linalg.norm(r.fun)<best[0]:best=np.linalg.norm(r.fun),r.x
      e,q=best; F=design(q); X=F[:N]; coeff=fit(X,y[:N]); res=X@coeff-y[:N]; hh=np.sum(X*np.linalg.pinv(X).T,1); loo=np.mean(abs(res)/np.maximum(1-hh,.05))*N/(N-len(q)); test=np.mean(abs(F[N:]@coeff-y[N:]));out.append((j,loo,test,e/np.sqrt(N),name,base,kind,mode,pair,np.round(q,3)))
for j in range(3):
 print('TARGET',j)
 for k in [1,2]: print(*sorted([r[1:] for r in out if r[0]==j],key=lambda r:r[k-1])[:7],sep='\n')
