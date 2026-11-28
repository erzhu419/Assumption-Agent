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
links['softsign']=(lambda z:z/(1+abs(z)),lambda z:1/(1+abs(z))**2)
E=D[:,9:]; pat={'diag':[0,5,10,15],'fwd':[1,6,11,12],'rev':[3,4,9,14],'opp':[2,7,8,13]}
Aopts={'diag':np.c_[D[:,:9],E[:,pat['diag']]],'diag+fwd':np.c_[D[:,:9],E[:,pat['diag']+pat['fwd']]],'all':D}
Bopts={k:E[:,j] for k,j in pat.items()}; Bopts.update(all=E,fd=E[:,pat['fwd']+pat['diag']],rd=E[:,pat['rev']+pat['diag']],fr=E[:,pat['fwd']+pat['rev']]); rows=[]
for ak,A in Aopts.items():
 for bk,B in Bopts.items():
  for r in (0.,.001,.01,.15):
   scores=[]
   for i,j in splits:
    a,h,b=fit(A[i],B[i],Y[i],'softsign',r); z=A[j]@a; scores.append(np.mean(abs(h*z/(1+abs(z))+B[j]@b-Y[j])))
   a,h,b=fit(A,B,Y,'softsign',r); z=A@a; rows.append((np.mean(scores),max(scores),scores[0],ak,bk,r,np.mean(abs(h*z/(1+abs(z))+B@b-Y)),h,a,b))
for item in sorted(rows,key=lambda x:x[0])[:12]:print([round(v,6) if isinstance(v,(float,np.floating)) else v for v in item[:8]])
for item in sorted(rows,key=lambda x:x[6])[:4]: print('best fit',item[:8],np.round(item[8],3),np.round(item[9],3))
