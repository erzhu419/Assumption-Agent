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
results=[]
for ln in links:
 for ak,A in A0.items():
  for bk,B in B0.items():
   if ak not in ('x','xc','d') and bk not in ('none','const'):continue
   for reg in (.0001,.003,.03):
    scores=[]
    for i,j in splits:
     a,h,b=fit(A[i],B[i],Y[i],ln,reg); scores.append(np.mean(abs(h*links[ln][0](A[j]@a)+B[j]@b-Y[j])))
    results.append((round(np.mean(scores),6),round(max(scores),6),round(scores[0],6),ln,ak,bk,reg))
for row in sorted(results)[:28]:
 print(row)
for row in sorted(results)[:4]:
 _,_,_,ln,ak,bk,reg=row; a,h,b=fit(A0[ak],B0[bk],Y,ln,reg); print(ln,ak,bk,h,np.round(a,4),np.round(b,4),np.mean(abs(h*links[ln][0](A0[ak]@a)+B0[bk]@b-Y)))
print('trajectory block MAEs',[round(np.mean(np.abs((D[j]@np.linalg.lstsq(D[i],f['T'][i],rcond=None)[0])-f['T'][j])),6) for i,j in splits])
