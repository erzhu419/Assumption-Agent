exec(open('/tmp/explore.py').read().split("p=s['panel_c'];")[0])
from scipy.optimize import minimize_scalar
p=s['panel_c']; a=p['u']; i,j=np.triu_indices(4,1)
def candidates(a):
 yield 'aff',f(a)
 for c in [.22,.5,1.,1.25,2.]:
  r=a/(1+c*a*a); g=np.tanh(c*a)
  for name,v in [('tanh',g),('sin',np.sin(c*a)),('rat',r),('mixed',a+.22*g),('sat',g*g),('ratsq',a*a/(1+c*a*a)),('cos',np.cos(c*a)),('exp',np.exp(-c*a*a))]:
   yield (name,c,'pure'),f(v)
   yield (name,c,'add'),np.c_[f(a),v]
   if name in ('tanh','sin','rat','mixed'):
    yield (name,c,'int'),np.c_[f(a),v,v[:,i]*v[:,j]]
    yield (name,c,'both'),np.c_[f(a),v,v*v]
 for v,name in [(a,'raw'),(np.tanh(1.25*a),'tanh'),(a/(1+.22*a*a),'rat')]:
  yield (name,'cross'),np.c_[f(a),v[:,i]*v[:,j]]
 yield 'quad',terms(a,2)[0]
 yield 'diag-cube',np.c_[f(a),a*a,a**3]
for k in ['v1','v2']:
 y=p[k];rank=[]
 for name,g in candidates(a):
  w=np.linalg.lstsq(g,y,rcond=None)[0];err=y-g@w;hat=(np.linalg.qr(g)[0]**2).sum(1);rank.append((np.mean(abs(err/(1-hat[:,None]))),str(name),np.std(err)))
 print(k,sorted(rank)[:15]); print('affine residuals', (y-f(a)@np.linalg.lstsq(f(a),y,rcond=None)[0]).round(3))
