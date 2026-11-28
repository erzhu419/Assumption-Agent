import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
 from probe14 import *
u=s['panel_a']['u']; v=s['panel_a']['v']; good=np.max(abs(v-(u*f['gain']+f['offset'])),axis=1)<1
print('A bad rows',np.where(~good)); print('clean residual ranges',np.min((v-u*f['gain']-f['offset'])[good],axis=0),np.max((v-u*f['gain']-f['offset'])[good],axis=0))
def fit(a,t):
 c=np.linalg.lstsq(a[good],t[good],rcond=None)[0]; print(np.mean(abs(a[good]@c-t[good]),axis=0)); return c
for d in (1,2,3):
 a=poly(u,d); c=fit(a,v); print('degree',d,'coefs',[(j,np.round(row,4).tolist()) for j,row in enumerate(c) if np.max(abs(row))>.015])
for label,func in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:
 for freq in (1,2,2.15,3,4,6):
  a=np.c_[poly(u,1),func(freq*u)]; print(label,freq,end=' '); fit(a,v)
for r in (1,2):
 u=s['panel_c']['u']; v=s['panel_c'][f'v{r}']; print('C',r)
 a=poly(u,2); c=np.linalg.lstsq(a,v,rcond=None)[0]; print(np.round(c,3));
 for label,fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh)]:
  def des(w): return np.c_[poly(u,1),fn(w*u)]
  def res(w): a=des(w); return (a@np.linalg.lstsq(a,v,rcond=None)[0]-v).ravel()
  opt=min([least_squares(res,[w],bounds=(.1,15)) for w in (1,2,3,5,8,12)],key=lambda o:o.cost); print(label,opt.x,np.mean(abs(opt.fun)))
