import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe_dag import *
from scipy.optimize import least_squares
from sklearn.model_selection import KFold

def compact(x,z,k,mode):
 a,b,c,d=z.T
 base=[aff(z), np.sin(k[0]*(x[:,2]-z[:,3]))]
 if mode=='bilinear': base += [a*c,b*d]
 if mode=='sine': base += [a*c,b*np.sin(k[1]*d)]
 if mode=='double': base += [a*np.sin(k[2]*c),b*np.sin(k[1]*d)]
 if mode=='mixed': base += [a*c,b*d,b*np.sin(k[1]*d)]
 if mode=='rational': base += [a*c,b*d/(1+k[1]*d*d)]
 return np.column_stack(base)

splits=[(np.arange(52),np.arange(52,72)),*KFold(6,shuffle=True,random_state=500).split(x)]
for mode in ('bilinear','sine','double','mixed','rational'):
 k0={'bilinear':[2.5],'sine':[2.5,3.5],'double':[2.5,3.5,1.0],'mixed':[2.5,3.5],'rational':[2.5,0.8]}[mode]
 stats=[]
 for tr,te in splits+[(np.arange(len(x)),np.arange(len(x)))]:
  def fun(k):
   B=compact(x[tr],z[tr],k,mode); return B@fit(B,y[tr])-y[tr]
  lo=[0.7]+[0.05]*(len(k0)-1); hi=[5]+[7]*(len(k0)-1)
  o=least_squares(fun,k0,bounds=(lo,hi)); B=compact(x,z,o.x,mode); c=fit(B[tr],y[tr]); stats.append(np.mean(abs(B[te]@c-y[te])))
  if len(tr)==len(x): print(mode,'full frequencies',np.round(o.x,4),'coefficients',np.round(c,4),'largest residuals',np.sort(abs(o.fun))[-5:]); print(' residual correlations',np.round(np.corrcoef(np.column_stack([o.fun,powers(x,2)[:,5:]]).T)[0,1:],3))
 print(' official / OOF mean / OOF worst / fit',np.round([stats[0],np.mean(stats[1:-1]),max(stats[1:-1]),stats[-1]],5))

# A shared coordinatewise nonlinearity predicts all four panel C channels with five affine coefficients per output.
for name,fun in [('tanh',np.tanh),('sin',np.sin),('cubic',lambda u:u+u**3),('rational',lambda u:u/(1+u*u)),('linear+square',lambda u:u+u*u)]:
 for key in ('v1','v2'):
  target=p[key]
  def pred(q,k): return aff(fun(q*k))
  def res(k):
   B=pred(q,k); return (B@fit(B,target)-target).ravel()
  o=least_squares(res,[1.]*4,bounds=(.02,4)); print('panel',name,key,np.round(o.x,3),np.round(np.mean(abs(o.fun)),5),np.round(loo(pred(q,o.x),target),4))
