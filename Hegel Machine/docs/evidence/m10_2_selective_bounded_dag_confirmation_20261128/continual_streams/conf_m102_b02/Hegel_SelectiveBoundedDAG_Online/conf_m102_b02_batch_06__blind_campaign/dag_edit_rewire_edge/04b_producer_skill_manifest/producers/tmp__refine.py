import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf
from itertools import combinations_with_replacement
f=np.load('/tmp/coords.npz'); X,C,D,Y=[f[k] for k in ('X','C','D','Y')]; n=len(Y)
rng=np.random.default_rng(51); splits=[(np.arange(n-20),np.arange(n-20,n))]+[(p[20:],p[:20]) for p in [rng.permutation(n) for _ in range(8)]]
xx=np.column_stack([X[:,i]*X[:,j] for i,j in combinations_with_replacement(range(4),2)])
cc=np.column_stack([C[:,i]*C[:,j] for i,j in combinations_with_replacement(range(4),2)])
links={'tanh':(np.tanh,lambda p:1-np.tanh(p)**2),'erf':(erf,lambda p:2/np.sqrt(np.pi)*np.exp(-p*p)),'rat':(lambda p:p/np.sqrt(1+p*p),lambda p:(1+p*p)**-1.5),'atan':(np.arctan,lambda p:1/(1+p*p)),'sin':(np.sin,np.cos)}
A0={'x':D[:,:5],'xc':D[:,:9],'d':D,'dxx':np.c_[D,xx],'dcc':np.c_[D,cc]}; B0={'none':D[:,:0],'const':D[:,:1],'x':D[:,:5],'xc':D[:,:9],'d':D,'cross':D[:,9:],'xcross':np.c_[D[:,:5],D[:,9:]],'xx':xx,'dxx':np.c_[D,xx]}
def fit(A,B,y,link,reg):
 fun,der=links[link]; ma=np.maximum(A.std(0),.04); ma[0]=1; a=A/ma; mb=np.maximum(B.std(0),.04); mb[mb==.04]=1; b=B/mb
 p=a.shape[1]; q=b.shape[1]; r=np.sqrt(reg)*np.r_[np.repeat(.5,p),0.,np.ones(q)]; r[0]=0
 init=np.r_[np.linalg.lstsq(a,y,rcond=None)[0]/2.5,2.5,np.zeros(q)]
 def calc(w):return a@w[:p],w[p],w[p+1:]
 def resid(w):
  z,h,v=calc(w);return np.r_[h*fun(z)+b@v-y,r*w]
 def jac(w):
  z,h,v=calc(w);return np.vstack([np.c_[h*der(z)[:,None]*a,fun(z),b],np.diag(r)])
 lo=np.full(len(init),-np.inf); hi=-lo; lo[p]=.15; hi[p]=15
 w=least_squares(resid,init,jac=jac,bounds=(lo,hi),max_nfev=65).x
 return w[:p]/ma,w[p],w[p+1:]/mb
from scipy.optimize import minimize_scalar
# Check whether the two bilinear branches share an index or a sparse interaction pattern.
links['qrat']=(lambda p:p/(1+p*p),lambda p:(1-p*p)/(1+p*p)**2)
links['asinh']=(np.arcsinh,lambda p:1/np.sqrt(1+p*p))
links['softsign']=(lambda p:p/(1+abs(p)),lambda p:1/(1+abs(p))**2)
links['cubic']=(lambda p:p-p**3,lambda p:1-3*p*p)
E=D[:,9:]; combos={}
for label,fs in [('diag',[(i,i) for i in range(4)]),('fwd',[(i,(i+1)%4) for i in range(4)]),('rev',[(i,(i-1)%4) for i in range(4)]),('diagfwd',[(i,i) for i in range(4)]+[(i,(i+1)%4) for i in range(4)])]:
 combos[label]=[4*i+j for i,j in fs]
results=[]
for ln in ('atan','rat','tanh','qrat','asinh','softsign','cubic'):
 for ak,A in [('xc',D[:,:9]),('d',D),*((k,np.c_[D[:,:9],E[:,j]]) for k,j in combos.items())]:
  for bk,B in [('cross',E),('xcross', B0['xcross']),('d',D),*((k,E[:,j]) for k,j in combos.items())]:
   if ak!='d' and bk not in ('cross','d'):continue
   for reg in (.01,.06,.15):
    scores=[]
    for i,j in splits:
     a,h,b=fit(A[i],B[i],Y[i],ln,reg);scores.append(np.mean(abs(h*links[ln][0](A[j]@a)+B[j]@b-Y[j])))
    results.append((np.mean(scores),max(scores),scores[0],ln,ak,bk,reg))
for k in (0,1):
 print('sort on',k)
 for m,w,s,*names in sorted(results,key=lambda x:x[k])[:14]:print(round(m,5),round(w,5),round(s,5),*names)
