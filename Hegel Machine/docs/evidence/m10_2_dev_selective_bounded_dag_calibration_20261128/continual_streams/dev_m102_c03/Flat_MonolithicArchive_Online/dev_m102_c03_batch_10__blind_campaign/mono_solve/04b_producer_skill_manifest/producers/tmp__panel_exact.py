import pickle,numpy as np
from scipy.optimize import least_squares
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u=s['panel_c']['u']; v=s['panel_c']['v1']; q=np.roll(u,-1,axis=1); r=np.roll(u,1,axis=1); h=np.roll(u,2,axis=1); cyc=u*q
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=1e-10)[0]
aff=lambda x:np.c_[np.ones(len(x)),x]
fs={'tanh':np.tanh,'sin':np.sin,'soft':lambda x:x/(1+abs(x)),'rat':lambda x:x/(1+x*x),'atan':np.arctan,'smooth':lambda x:x/np.sqrt(1+x*x)}
results=[]
for name,f in fs.items():
 for g,extra in [('none',np.empty((len(u),0))),('linear',u),('cycle',cyc),('both',np.c_[u,cyc]),('square',u*u),('cubic',u**3),('linear_cubic',np.c_[u,u**3])]:
  def run(p,full=False):
   a=aff(np.c_[f(u*p),extra]); b=ls(a,v); e=(a@b-v); return (b,e) if full else e.ravel()
  for dim in (1,4):
   opt=min((least_squares(run,np.full(dim,k),bounds=(.08,4),gtol=1e-11) for k in (.5,1.4,2.4)),key=lambda x:x.cost)
   results.append((np.sqrt(2*opt.cost/v.size),name,g,opt.x))
for row in sorted(results,key=lambda x:x[0])[:25]: print(row)
results=[]
for name,f in fs.items():
 for shift,r in [('next',q),('prev',r),('opp',h)]:
  for form in ('sum','prod','mix','mixprod','poly'):
   for lin in (False,True):
    def residual(p):
     a,b=p
     x={'sum':u+b*r,'prod':u+b*u*r,'mix':u+b*r*(1+u),'mixprod':u+b*r*(1+u*u),'poly':u+b*u**3}[form]
     A=aff(np.c_[u,f(a*x)] if lin else f(a*x)); return (A@ls(A,v)-v).ravel()
    opt=min((least_squares(residual,[p,k],bounds=([.05,-3],[5,3]),gtol=1e-11) for p in (.65,1.5,2.4) for k in (-.2,.2)),key=lambda x:x.cost)
    results.append((np.sqrt(2*opt.cost/v.size),name,shift,form,lin,opt.x))
print('inner mixtures:')
for row in sorted(results,key=lambda x:x[0])[:25]: print(row)
