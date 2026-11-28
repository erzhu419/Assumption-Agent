import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe_dag import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold
A=aff(q); U=p['v1']; V=p['v2']; F=powers(q,2); n=len(q)
print('v1 affine coefficients',fit(A,U)); print('v2 quadratic coefficients',fit(F,V)); print('v1 moments',U.min(0),U.max(0))
models={'sin':np.sin,'tanh':np.tanh,'cubic':lambda x:x**3,'rational':lambda x:x/(1+x*x),'softsign':lambda x:x/(1+abs(x))}
def f(x,c,key):
 if key=='cubic': return (x@c)**3
 return models[key](x@c)
def vp(X,Y,c,L,key):
 H=np.column_stack([L,f(X,c,key)]); return H@fit(H,Y)-Y
rng=np.random.default_rng(200)
for key in models:
 for skip in (False,True):
  LS=[]; scores=[]; pars=[]
  for j in range(4):
   target=U[:,j]; direction=fit(A,target); L=A if skip else A[:,:1]
   starts=[direction*g for g in (.5,1,2,3)]+[rng.normal(size=5) for _ in range(7)]
   opt=min([least_squares(lambda c:vp(A,target,c,L,key),init,max_nfev=250) for init in starts],key=lambda o:o.cost)
   c=opt.x; H=np.column_stack([L,f(A,c,key)]); scores.append(np.mean(abs(H@fit(H,target)-target))); pars.append(np.round(np.r_[c,fit(H,target)],3)); LS.append(c)
  print(key,skip,np.round(scores,4),pars)
# Compare known cubic basis under validation, including the official split and extremes.
for label, Z in [('cal',x),('mixed',z)]:
 print('cubic coefficients',label,np.round(fit(powers(Z,3),y),3))
 for kind,A2 in [('cubic',powers(Z,3)),('pure cubic',np.column_stack([powers(Z,2),Z**3])),('sin+quad',np.column_stack([powers(Z,2),np.sin(2*Z)])),('tanh+quad',np.column_stack([powers(Z,2),np.tanh(2*Z)]))]:
  print(label,kind,'loo',loo(A2,y),'last',np.mean(abs(y[56:]-A2[56:]@fit(A2[:56],y[:56]))))
