import contextlib,io
with contextlib.redirect_stdout(io.StringIO()): from probe import *
from scipy.optimize import least_squares
for p,k in [('panel_a','v'),('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 x=s[p]['u'];Y=s[p][k];X=poly(x);Q=poly(x,2);print('\n',p,k,'ranges',Y.min(0),Y.max(0))
 for j,y in enumerate(Y.T):
  entries=[]
  for name,f in [('tanh',np.tanh),('sin',np.sin),('softsign',lambda a:a/(1+abs(a))),('rat',lambda a:a/(1+a*a)),('linear',lambda a:a)]:
   for branch in [0,1]:
    def pred(c,x): return f(x@c[:5]) if not branch else c[5]+c[6]*f(x@c[:5]);
    c=np.r_[ls(X,y),0.,1.] if branch else ls(X,y)
    cc=least_squares(lambda c:pred(c,X)-y,c,max_nfev=500,loss='soft_l1',f_scale=.008).x
    entries.append((round(np.median(abs(pred(cc,X)-y)),5),name,branch,cc))
  print(j,sorted(entries,key=lambda a:a[0])[:3]);print('Q',ls(Q,y))
