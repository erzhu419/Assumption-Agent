from shape04 import *
from itertools import product

def candidates2():
 yield 'rational + sinusoid',lambda q,p:q**3/(1+p[0]*q*q)+p[1]*q+p[2]*np.sin(p[3]*q),[.5,-.09,.01,8],([0,-1,-.5,1],[5,1,.5,30]),3
 yield 'cubic + sinusoid',lambda q,p:q**3+p[0]*q+p[1]*np.sin(p[2]*q),[-.09,.01,8],([-1,-.5,1],[1,.5,30]),2
 yield 'fifth + sinusoid',lambda q,p:q**3+p[0]*q+p[1]*q**5+p[2]*np.sin(p[3]*q),[-.09,-.2,.01,8],([-1,-1,-.5,1],[1,1,.5,30]),3
 yield 'modulated rational',lambda q,p:q**3/(1+p[0]*q*q)+p[1]*q+p[2]*q*np.cos(p[3]*q),[.5,-.09,.01,8],([0,-1,-.5,1],[5,1,.5,30]),3
 yield 'rational cubic harmonic',lambda q,p:q**3/(1+p[0]*q*q)+p[1]*q+p[2]*np.sin(p[3]*q**3),[.5,-.09,.01,8],([0,-1,-.5,1],[5,1,.5,30]),3
 yield 'linear plus sine cubic',lambda q,p:p[0]*q+np.sin(p[1]*q**3),[-.09,.8],([-1,.1],[1,4]),1

def run(row, i, j, a):
 name,f,init,b,ix=row
 starts=(3,6,9,12,15,18,22,27) if name!='linear plus sine cubic' else (.4,.8,1.6,3.2)
 opts=[]
 for k in starts:
  p=init.copy(); p[ix]=k
  pred,fit,r=solve_shape(f,p,b,i,a); opts.append((np.dot(r,r),pred,fit,r))
 _,pred,fit,r=min(opts,key=lambda o:o[0]); return scores(i,j)(pred(w[j])),fit,np.sqrt(np.mean(r[:v.size]**2))
order=np.random.default_rng(194).permutation(72); edge=np.argsort(abs(w).max(1))[-16:]; folds=[(np.arange(56),np.arange(56,72))]+[(np.setdiff1d(order,j),j) for j in np.array_split(order,5)]+[(np.setdiff1d(order,edge),edge)]
for row in candidates2():
 out=[run(row,i,j,1) for i,j in folds]; es=np.array([o[0] for o in out]); print(row[0],np.round(out[0][1],5),round(out[0][2],6),'56/16',es[0],'worst',es.max(0),'edge',es[-1],flush=True)
# Remaining noise and the shape of the residual after projecting onto temporal modes.
for name,f,p,b in __import__('knee04').forms2():
 if name!='odd cheb 15': continue
 pred,fit,r=solve_shape(f,p,b,np.arange(72),4); yy,tt=pred(w); print('fit',fit,'residual sv',np.linalg.svd(t-tt,compute_uv=False),'signal sv',np.linalg.svd(tt,compute_uv=False)); print('shape',np.round(f(np.arange(0,1.251,.05),fit),5))
