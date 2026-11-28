exec(open('/tmp/fit06.py').read().split("for name,h in [('cal',q)")[0])
from scipy.optimize import least_squares
from itertools import product

def profile(u,v,feature,starts,bounds):
 def resid(p):
  f=feature(u,p);return (f@ls(f,v)-v).ravel()
 opt=min((least_squares(resid,p,bounds=bounds,max_nfev=180) for p in starts),key=lambda o:o.cost)
 w=ls(feature(u,opt.x),v)
 print('ref',opt.x,np.sqrt(np.mean(resid(opt.x)**2)));return lambda h:feature(h,opt.x)@w
print('frozen B rmse',np.sqrt(np.mean((panelb(b['u'])-b['v'])**2)))
C1=profile(c['u'],c['v1'],lambda h,p:np.c_[design(h,1),np.arctan(p[0]*h)],[[2]],(.01,15))
C2=profile(c['u'],c['v2'],lambda h,p:np.c_[design(h,2),np.tanh(p[0]*h*h)],[[1]],(.01,15))
B1=profile(b['u'],b['v'],lambda h,p:np.c_[design(h,1),h*np.exp(-p[0]*h*h)],[[1]],(.01,15))
print('temporal coeffs',np.linalg.svd(tr,full_matrices=False)[2][:3,-1])
trans={'B':panelb,'E':B1,'C':C1,'D':C2}; coords={'q':q}
for n in (1,2,3):
 for names in product(trans,repeat=n):
  if names.count('D')>1:continue
  h=q
  for k in names:h=trans[k](h)
  coords[''.join(names)]=h

def errs(f,target):
 return [np.mean(abs(f[te]@ls(f[ti],target[ti])-target[te])) for ti,te in splits]
splits=[(np.arange(52),np.arange(52,len(y))),(np.arange(20,len(y)),np.arange(20))]
for target in [tr,y]:
 score=[]
 for name,h in coords.items():
  for d in (1,2):
   f=design(h,d);score.append((errs(f,target),name,d))
  if len(name)>1 and name[:-1] in coords:
   f=np.c_[design(coords[name[:-1]],1),h];score.append((errs(f,target),name,'combined'))
 print('best',sorted(score,key=lambda s:sum(s[0]))[:20])
print('well fit c2 coef')
print(ls(np.c_[design(c['u'],2),np.tanh(1.38*c['u']**2)],c['v2']).round(3))
