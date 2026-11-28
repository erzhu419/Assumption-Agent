import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe14 import *
from scipy.optimize import root
ls=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
A=s['panel_a']; C=s['panel_c']; u=A['u']; v=A['v']; good=np.max(abs(v-(u*f['gain']+f['offset'])),axis=1)<1
print('sizes',u.shape,C['u'].shape, 'times',s['eval_times'])
def profile(u,v,tag):
 best=[]
 for bas,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('atan',np.arctan),('rational',lambda a:a/(1+a*a)),('soft',lambda a:a/np.sqrt(1+a*a)),('erf',__import__('scipy').special.erf)]:
  def d(q,p,fn=fn):return np.c_[poly(q,1),fn(q*p)]
  def res(p):a=d(u,p);return (a@ls(a,v)-v).ravel()
  opt=min((least_squares(res,[w],bounds=(.07,12)) for w in (.25,.6,1.25,2.15,3.5,6,9)),key=lambda o:o.cost)
  c=ls(d(u,opt.x),v); err=np.mean(abs(opt.fun)); print(tag,bas,opt.x,err); best.append((err,lambda q,d=d,p=opt.x,c=c:d(q,p)@c,bas,c,opt.x))
 _,fn,bas,c,p=min(best,key=lambda t:t[0]); print(tag,bas,np.round(c,4)); return fn,c
fa,ca=profile(u[good],v[good],'A'); fc1,_=profile(C['u'],C['v1'],'C1'); fc2,_=profile(C['u'],C['v2'],'C2')
def inverse(row):
 p=root(lambda q:fa(q[None])[0]-row,(row-f['offset'])/f['gain']); return p.x
q=np.array([inverse(row) for row in tr['x']]); b=F._b_features(q,f['p'])@f['w']; print('q bounds',q.min(0),q.max(0)); print('a inversion',np.max(abs(fa(q)-tr['x'])))
features={'q':q,'b':b,'1q':fc1(q),'2q':fc2(q),'1b':fc1(b),'2b':fc2(b),'21q':fc2(fc1(q)),'12q':fc1(fc2(q)),'21b':fc2(fc1(b)),'12b':fc1(fc2(b))}
for name,a in features.items():
 for j in (1,2):
  aa=poly(a,j); report('y '+name+str(j),aa,y); report('z '+name+str(j),aa,z)
