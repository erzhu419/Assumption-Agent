from shape04 import *

def forms2():
 yield 'clipped cubic',lambda q,p:np.clip(q,-p[0],p[0])**3+p[1]*q,[.92,-.09],([.6,-.5],[1.25,.5])
 yield 'clip cubic rational',lambda q,p:(lambda a:a**3/(1+p[2]*a*a))(np.clip(q,-p[0],p[0]))+p[1]*q,[1.03,-.09,.4],([.6,-.5,0],[1.25,.5,5])
 yield 'clip whole',lambda q,p:np.clip(q**3+p[1]*q,-p[0],p[0]),[.68,-.09],([.2,-.5],[1.4,.5])
 yield 'power tail',lambda q,p:q**3+p[1]*q+p[2]*np.sign(q)*np.maximum(abs(q)-p[0],0)**p[3],[.63,-.07,-2.,1.8],([.25,-.5,-10,.6],[1.15,.5,0,4])
 yield 'rational then hinge',lambda q,p:q**3/(1+p[0]*q*q)+p[1]*q+p[2]*np.sign(q)*np.maximum(abs(q)-p[3],0),[.4,-.09,-.25,.9],([0,-.5,-3,.4],[5,.5,3,1.22])
 yield 'soft knee',lambda q,p:q**3/(1+(abs(q)/p[0])**p[1])**(p[2]/p[1])+p[3]*q,[.82,12,1,-.09],([.55,3,.1,-.5],[1.25,100,4,.5])
 yield 'quintic tail',lambda q,p:q**3+p[1]*q+p[2]*q**5+p[3]*np.sign(q)*np.maximum(abs(q)-p[0],0),[.95,-.09,-.25,-.25],([.4,-.5,-1,-4],[1.2,.5,1,4])
 yield 'rational shifted',lambda q,p:q**3/(1+p[0]*q*q+p[1]*q**4+p[2]*q**6)+p[3]*q,[.5,-.2,.2,-.09],([0,-.8,0,-.5],[5,5,5,.5])
 for deg in (13,15):
  def fun(q,p,deg=deg):
   a=chebvander(q/1.25,deg); return a[...,3]+a[...,1]*p[0]+a[...,5::2]@p[1:]
  m=(deg-1)//2; yield 'odd cheb '+str(deg), fun, [4,-.3,0,0,0]+[0]*(m-5),(-np.ones(m)*15,np.ones(m)*15)

if __name__=='__main__':
 order=np.random.default_rng(194).permutation(len(w)); edge=np.argsort(abs(w).max(1))[-16:]; folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(order,j),j) for j in np.array_split(order,5)]+[(np.setdiff1d(order,edge),edge)]
 for name,f,p,b in list(forms2())+[r for r in forms() if r[0]=='odd cheb 11']:
  for alpha in (.7,4,15):
   es=[]
   for i,j in folds:
    pred,fit,r=solve_shape(f,p,b,i,alpha); es.append(scores(i,j)(pred(w[j])))
   es=np.array(es); print(name,alpha,np.round(fit,5),'panel',round(np.sqrt(np.mean(r[:v.size]**2)),6),'56/16',np.round(es[0],5),'max random',np.round(es[:-1].max(0),5),'edge',np.round(es[-1],5),flush=True)
