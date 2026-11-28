exec(open('/tmp/compose.py').read().split('for name,c in base.items():')[0])
from itertools import combinations, product
from scipy.optimize import least_squares
for pn,vn in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[pn]['u']; v=s[pn][vn]; n=len(u); opts=[]
 def fitf(fun,lim,extras):
  def mat(x,a):return np.column_stack([np.ones(len(x)),fun(x,a),*[ee(x) for ee in extras]])
  sol=minimize_scalar(lambda a:loo(mat(u,a),v)[0],bounds=lim,method='bounded');p=np.linalg.lstsq(mat(u,sol.x),v,rcond=None)[0];return sol.fun,sol.x,lambda x:mat(x,sol.x)@p
 extra={'none':[],'lin':[lambda x:x],'sq':[lambda x:x*x],'cube':[lambda x:x**3],'adj':[lambda x:x*np.roll(x,-1,axis=1)],'linadj':[lambda x:x,lambda x:x*np.roll(x,-1,axis=1)],'linsq':[lambda x:x,lambda x:x*x]}
 for nm,fn,lim in [('odd3rat',lambda x,a:x**3/(1+a*x*x),(.01,2)),('odd3ratabs',lambda x,a:x**3/(1+a*abs(x)),(.01,3)),('odd3atan',lambda x,a:np.arctan(a*x**3),(.01,4)),('odd3tanh',lambda x,a:np.tanh(a*x**3),(.01,4)),('odd3sin',lambda x,a:np.sin(a*x**3),(.01,1.5)),('odd3sat',lambda x,a:x**3/np.sqrt(1+a*x*x),(.01,5)),('atan',lambda x,a:np.arctan(a*x),(.1,5)),('tanh',lambda x,a:np.tanh(a*x),(.1,5)),('exp',lambda x,a:np.expm1(-a*x*x),(.01,1)),('exppos',lambda x,a:np.expm1(a*x*x),(.01,1)),('evenrat',lambda x,a:x*x/(1+a*x*x),(.01,1)),('cos',lambda x,a:np.cos(a*x),(.1,3)),('sinc',lambda x,a:np.sinc(a*x),(.04,1)),('odd3exp',lambda x,a:x**3*np.exp(-a*x*x),(.01,2)),('oddratpower',lambda x,a:x*np.abs(x)**a,(1,4)),('expcube',lambda x,a:np.expm1(-a*x*x*x),(.01,1))]:
  for en,ex in extra.items():
   err,a,fun=fitf(fn,lim,ex);opts.append((err,nm,en,a,fun))
 opts.sort(key=lambda o:o[0]);print('PANEL',pn,vn,[o[:4] for o in opts[:12]],flush=True)
 if pn=='panel_b': B1=opts[0][4]
 elif vn=='v1': C1=opts[0][4]
 else: D1=opts[0][4]
mods={'B':B1,'C':C1,'D':D1}; vals={'z':z}
for depth in range(2):
 for n,v in list(vals.items()):
  if len(n)==depth+1:
   for m,h in mods.items():vals[n+m]=h(v)
_,_,vt=np.linalg.svd(tr[:52],full_matrices=False); target=tr@vt[:2].T
best=[]; outer={**mods,'I':lambda x:x}
def add(n,v):
 if not np.isfinite(v).all():return
 a=aff(v); b=np.linalg.lstsq(a[:52],target[:52],rcond=None)[0];t=np.mean(abs((a[:72]@b-target)@vt[:2]),axis=1);worst=max(np.mean(t[:52]),np.mean(t[52:72]));
 if worst<.125: best.append((worst,n,v))
ops={'*':lambda x,y:x*y,'/':lambda x,y:x/(1+y*y),'~':lambda x,y:x*np.tanh(y),'+':lambda x,y:x+y,'-':lambda x,y:x-y,'sq':lambda x,y:x+y*y,'mix':lambda x,y: x*(1+y),'rat':lambda x,y:x/(1+abs(y))}
for n,v in vals.items():
 add(n,v)
 for k,w in vals.items():
  for o,h in ops.items():
   a=h(v,w)
   for l,e in outer.items():add(n+o+k+':'+l,e(a))
   if len(n)+len(k)<=4:
    for j,r in vals.items():
     for l,e in outer.items():
      add(n+o+k+'+'+j+':'+l,e(a+r));add(n+o+k+'-'+j+':'+l,e(a-r));add('('+n+o+k+')*'+j+':'+l,e(a*r));add('('+n+o+k+')/'+j+':'+l,e(a/(1+r*r)))
best.sort(key=lambda o:o[0]); winning=[]
for err,n,v in best[:45]:
 pair=score(v);winning.append((max(pair),n,err,pair));
print('BEST',sorted(winning)[:25]);print('TRAJ',best[0][:2] if best else None)
