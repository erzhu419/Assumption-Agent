from panel_probe import *
from scipy.optimize import minimize_scalar

def trial(name,f,qo=None,detail=False):
 def resid(q):
  A=aff(f(u,q));return A@ls(A,v)-v
 if qo is None:q=[]
 elif len(qo)==2 and np.isscalar(qo[0]):
  grid=np.linspace(*qo,81); i=np.argmin([np.mean(resid([k])**2) for k in grid]); q=[minimize_scalar(lambda k:np.mean(resid([k])**2),bounds=(grid[max(0,i-1)],grid[min(i+1,len(grid)-1)]),method='bounded').x]
 else:
  fits=[least_squares(lambda q:resid(q).ravel(),s,max_nfev=200) for s in qo]; q=min(fits,key=lambda o:o.cost).x
 A=aff(f(u,q));b=ls(A,v);r=resid(q);w=aff(f(x,q))@b
 h=np.sum(A*np.linalg.pinv(A).T,axis=1)
 print(name,np.round(q,6), np.mean(r*r)**.5,np.mean((r/(1-h[:,None]))**2)**.5,score(w),flush=True)
 if detail:print('b',np.round(b,5),'resid',np.round(np.sqrt(np.mean(r*r,axis=0)),6))
trial('squares',lambda z,q:z*z,detail=True)
for name,f in [('cos',lambda z,g:np.cos(g*z)),('exp',lambda z,g:np.exp(-g*z*z)),('rat',lambda z,g:z*z/(1+g*z*z)),('tanh',lambda z,g:np.tanh(g*z*z)),('atan',lambda z,g:np.arctan(g*z*z)),('cosh',lambda z,g:np.cosh(g*z)),('abs',lambda z,g:abs(z)**g)]:
 trial(name,lambda z,q:f(z,q[0]),(.05,10),True)
 trial(name+' plus linear',lambda z,q:np.c_[f(z,q[0]),z],(.05,10))
 trial(name+' plus square',lambda z,q:np.c_[f(z,q[0]),z*z],(.05,10))
 trial(name+' combined square',lambda z,q:z*z+q[1]*f(z,q[0]),[[g,h] for g in [1,2.15,3.5,5] for h in [-2,-.3,.3,2]])
trial('square+fourth',lambda z,q:z*z+q[0]*z**4,(-2,3),True)
trial('square+fourth+sixth',lambda z,q:z*z+q[0]*z**4+q[1]*z**6,[[0,0],[-.3,.1]])
trial('fourth',lambda z,q:z**4)
trial('cos shifted',lambda z,q:np.cos(q[0]*z+q[1]),[[g,b] for g in [1,2.15,4,6] for b in [0,.3]],True)
trial('coupled square+next product',lambda z,q:z*z+q[0]*z*np.roll(z,-1,axis=1),(-2,2))
