from ops import *
from scipy.special import erf
from scipy.optimize import least_squares
F.update({'cubic':lambda z:z**3,'logcosh':lambda z:np.logaddexp(z,-z)-np.log(2),'softodd':lambda z:z/(1+abs(z)),'oddexp':lambda z:z*np.exp(-z*z),'sinh':np.sinh,'quintic':lambda z:z**5,'tanh3':lambda z:np.tanh(z)**3})
ops={}
for label,p,key,names in [('b','panel_b','v',['erf','ratodd','tanh','oddexp','sinh','asinh','sin','cubic','tanh3']),('c','panel_c','v1',['asinh','atan','rootodd','tanh','sin','erf']),('d','panel_c','v2',['log2','root2','rat2','logcosh'])]:
 u,v=s[p]['u'],s[p][key]; results=[]
 for n in names:
  for m in ['none','sq','cube','fifth','even','abs']:
   def design2(x,q,n=n,m=m):
    extra={'sq':lambda x:x*x,'cube':lambda x:x**3,'fifth':lambda x:x**5,'even':lambda x:np.log1p(x*x),'abs':abs};return np.c_[np.ones(len(x)),x,F[n](q*x),*([extra[m](x)] if m!='none' else [])]
   def fun(k):
    g=design2(u,np.exp(k[0]));return (g@np.linalg.lstsq(g,v,rcond=None)[0]-v).ravel()
   k=min((least_squares(fun,[z],bounds=(-2.5,2.1),max_nfev=70) for z in [-.65,.4,1.25]),key=lambda o:o.cost).x[0];q=np.exp(k);g=design2(u,q); w=np.linalg.lstsq(g,v,rcond=None)[0]; r=(g@w-v);loo=np.sqrt(np.mean((r/np.maximum(.025,1-(g*np.linalg.pinv(g).T).sum(1))[:,None])**2));results.append((loo,n,m,q,np.std(r),w,lambda x,f=design2,q=q,w=w:f(x,q)@w))
 for e,n,m,q,r,w,f in sorted(results,key=lambda r:r[0])[:7]: print(label,n,m,round(q,5),round(e,7),round(r,7),'coef ratio',np.round((w[1:5]*w[5:9]).sum(1)/(w[1:5]**2).sum(1),4))
 ops[label]=sorted(results,key=lambda r:r[0])[0][-1]
np.savez('/tmp/nodes.npz',x=x,b=ops['b'](x),c=ops['c'](x),d=ops['d'](x),bc=ops['c'](ops['b'](x)),bd=ops['d'](ops['b'](x)))
