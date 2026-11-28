exec(open('/tmp/readout_probe.py').read().split('for name,p in')[0])
from itertools import combinations_with_replacement
c=s['panel_c']; cu=c['u']; R=np.random.default_rng(42); folds=np.array_split(R.permutation(n),6); tar=np.c_[y,t]; raw=s['train']['x']; lat=np.cbrt(q)
def basis(x,d): return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],1) for h in range(1,d+1) for j in combinations_with_replacement(range(x.shape[1]),h)])
def score(a,tar=tar):
 errs=[]
 for h in folds:
  i=np.setdiff1d(np.arange(n),h); e=np.abs(a[h]@np.linalg.lstsq(a[i],tar[i],rcond=1e-9)[0]-tar[h]); errs.append([e[:,0].mean(),e[:,1:].mean()])
 return np.mean(errs,0),np.max(errs,0)
cfits={}
for key in ['v1','v2']:
 v=c[key]; print('PANEL C',key)
 for name,fn in [('tanh',lambda x,f:np.tanh(f*x)),('aff+tanh',lambda x,f:np.c_[x,np.tanh(f*x)]),('aff+cos',lambda x,f:np.c_[x,np.cos(f*x)]),('aff+sin',lambda x,f:np.c_[x,np.sin(f*x)]),('aff+sin+cos',lambda x,f:np.c_[x,np.sin(f*x),np.cos(f*x)]),('poly3+cos',lambda x,f:np.c_[x,x**2,x**3,np.cos(f*x)]),('aff+squash',lambda x,f:np.c_[x,x/np.sqrt(1+f*x*x)]),('aff+atan',lambda x,f:np.c_[x,np.arctan(f*x)]),('aff+cbrt',lambda x,f:np.c_[x,np.cbrt(x)])]:
  def loss(f,ret=False):
   a=aff(fn(cu,f)); w=np.linalg.lstsq(a,v,rcond=None)[0]; return w if ret else np.mean((a@w-v)**2)**.5
  fs=np.linspace(.03,7,90); j=np.argmin([loss(f) for f in fs]); o=minimize_scalar(loss,bounds=(fs[max(j-1,0)],fs[min(j+1,89)]),method='bounded'); print(name,np.round([o.x,o.fun],7)); cfits[key+name]=(o.fun,fn,o.x,loss(o.x,True))
print('READOUT shortlist')
ops={'B':lambda x:aff(np.cos(f*x))@co,'cos':lambda x:np.cos(f*x),'invB':lambda x:(x-co[0])@np.linalg.pinv(co[1:])}
for k,(e,fn,g,w) in cfits.items():
 if e<.04: ops[k]=lambda x,fn=fn,g=g,w=w:aff(fn(x,g))@w
xs={'q':q,'lat':lat,'raw':raw,'lat2':lat**2,'asinhq':np.arcsinh(q)}
for k,x in list(xs.items()):
 for j,op in ops.items(): xs[k+':'+j]=op(x)
results=[]
for k,x in list(xs.items()):
 for j,fn in {'id':lambda x:x,'cos':lambda x:np.cos(f*x),'sin':lambda x:np.sin(f*x),'tanh':np.tanh,'square':np.square}.items():
  a=fn(x)
  for d in (1,2,3):
   m=basis(a,d); avg,worst=score(m); results.append((sum(avg),k,j,d,avg,worst))
   if d<3:
    avg,worst=score(np.c_[m,lat,q,np.cos(f*lat),np.cos(f*q)]);results.append((sum(avg),k,j+'+orig',d,avg,worst))
for r in sorted(results,key=lambda r:r[0])[:22]: print(r)
