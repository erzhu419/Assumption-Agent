from refine04 import *
from numpy.polynomial.chebyshev import chebvander

def signed_ratio(q,k,h,l): return q**3/(1+k*q*q+h*q**4)+l*q

def design(z): return np.c_[aff(z),z[:,0]*z[:,2]+z[:,1]*z[:,3]]

def solve_shape(f,p0,bounds,idx,alpha=1):
 def profile(p):
  pu=aff(f(u,p)); pw=aff(f(w[idx],p)); B=ls(pu,v); c=ls(pw,t[idx]); z=pw@B; d=design(z); cy=ls(d,y[idx]); return np.r_[(pu@B-v).ravel(),alpha*(pw@c-t[idx]).ravel()/np.sqrt(t.shape[1]),alpha*(d@cy-y[idx])],B,c,cy
 if len(p0): p=least_squares(lambda p:profile(p)[0],p0,bounds=bounds,max_nfev=100).x
 else: p=np.array([])
 r,B,c,cy=profile(p)
 def pred(q):
  a=aff(f(q,p)); return design(a@B)@cy,a@c
 return pred,p,r

def forms():
 yield 'quartic denominator',lambda q,p:signed_ratio(q,*p),[.55,.05,-.09],([0,0,-1],[8,8,1])
 yield 'sextic denominator',lambda q,p:q**3/(1+p[0]*q*q+p[1]*q**4+p[2]*q**6)+p[3]*q,[.4,0,.05,-.09],([0,0,0,-1],[8,8,8,1])
 yield 'tanh cubic',lambda q,p:np.tanh(p[0]*q**3)+p[1]*q,[.5,-.05],([.01,-4],[8,4])
 yield 'atan cubic',lambda q,p:np.arctan(p[0]*q**3)+p[1]*q,[.5,-.05],([.01,-4],[8,4])
 yield 'sqrt cubic',lambda q,p:q**3/np.sqrt(1+p[0]*q**6)+p[1]*q,[.5,-.09],([0,-1],[8,1])
 yield 'sin cubic',lambda q,p:np.sin(p[0]*q**3)+p[1]*q,[.5,-.05],([.01,-4],[2,4])
 yield 'exp cubic',lambda q,p:q**3*np.exp(-p[0]*abs(q)**p[1])+p[2]*q,[.2,3,-.09],([.001,.5,-1],[4,12,1])
 yield 'powered flexible',lambda q,p:q**3/(1+p[0]*abs(q)**p[1])**p[3]+p[2]*q,[3.5,6,-.09,.17],([.015,.6,-4,.02],[100,20,4,5])
 for deg in (5,7,9,11):
  def shape(q,p,deg=deg):
   cheb=chebvander(q/1.25,deg); return cheb[...,3]+cheb[...,1]*p[0]+cheb[...,5::2]@p[1:]
  m=(deg-1)//2; yield 'odd cheb '+str(deg),shape,[2.3]+[-.1]+[0]*(m-2),(-np.ones(m)*15,np.ones(m)*15)

def scores(i,j):
 return lambda pr: np.array([abs(pr[0]-y[j]).mean(),abs(pr[1]-t[j]).mean()])
if __name__=='__main__':
 print('times',times,'telemetry',S['telemetry_vec'],'panel span',u.min(0),u.max(0),'summary',tr['summary']); print('actual summary',y.mean(),y.std(),t[:,-1].mean(),t[:,-1].std())
 order=np.random.default_rng(194).permutation(72); folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(order,j),j) for j in np.array_split(order,5)]
 for name,f,p,b in forms():
  for a in (0.7,3,15):
   out=[]
   for i,j in folds:
    pred,fit,r=solve_shape(f,p,b,i,a); out.append(scores(i,j)(pred(w[j])))
   out=np.array(out); print(name,a,np.round(fit,4),'panel RMS',round(np.sqrt(np.mean(r[:v.size]**2)),6),'56/16',np.round(out[0],5),'mean',np.round(out[1:].mean(0),5),'max',np.round(out.max(0),5))
