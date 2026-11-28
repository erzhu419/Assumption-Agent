from graph import *
from scipy.optimize import least_squares
from scipy.special import erf
ops={'tanh':np.tanh,'atan':np.arctan,'asinh':np.arcsinh,'sin':np.sin,'sinh':np.sinh,'erf':erf,'rat':lambda x:x/(1+x*x),'soft':lambda x:x/np.sqrt(1+x*x),'exp':lambda x:-np.expm1(-x*x)*np.sign(x)}
for n,k in [('b','v'),('c','v1')]:
 p=s['panel_'+n];x=p['u'];y=p[k]; L=[]
 def add(name,fn,guess,lo,hi):
  def res(t):
   D=lin(fn(x,t));return (D@ls(D,y)-y).ravel()
  best=min([least_squares(res,g,bounds=(lo,hi),max_nfev=160) for g in guess], key=lambda r:r.cost);t=best.x; D=lin(fn(x,t));a,b=look(D,y); L.append((b,a,name,np.round(t,5)));return t
 add('odd-poly',lambda x,t:x+t[0]*x**3+t[1]*x**5+t[2]*x**7,[[.2,0,0]],[-20]*3,[20]*3)
 for nm,f in ops.items():
  for en in ['polyoutside','polyinside','oddplus','linearplus']:
   if en=='polyoutside':fn=lambda x,t:f(t[0]*x)+t[1]*x+t[2]*x**3; G=[[a,b,0] for a in [.65,1.25,2.5,4]] if (b:=(-.7 if n=='b' else .2)) else [];l=[.1,-15,-10];h=[7,15,10]
   if en=='polyinside':fn=lambda x,t:f(t[0]*x+t[1]*x**3)+t[2]*x;G=[[a,0,0] for a in [.65,1.25,2.5,4]];l=[.1,-10,-15];h=[7,10,15]
   if en=='oddplus':fn=lambda x,t:x**3+t[1]*f(t[0]*x);G=[[a,.5] for a in [.65,1.25,2.5,4]];l=[.1,-15];h=[7,15]
   if en=='linearplus':fn=lambda x,t:x+t[1]*f(t[0]*x);G=[[a,b] for a in [.65,1.25,2.5,4] for b in [-.9,.5]];l=[.1,-15];h=[7,15]
   add(nm+'_'+en,fn,G,l,h)
 print(n,sorted(L)[:14]);print('range x',x.min(),x.max())
