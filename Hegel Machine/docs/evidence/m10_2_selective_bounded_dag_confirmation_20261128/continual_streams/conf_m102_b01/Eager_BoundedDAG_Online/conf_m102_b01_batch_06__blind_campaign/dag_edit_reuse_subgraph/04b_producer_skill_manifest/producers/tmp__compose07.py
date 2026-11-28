exec(open('/tmp/probe07.py').read().split('for name,x')[0])
from scipy.optimize import least_squares, minimize
from itertools import combinations_with_replacement

def lc(x):return np.logaddexp(x,-x)-np.log(2)
def calibrate(a,z,design,p0,bounds=(-np.inf,np.inf)):
 def res(p):
  F=design(a,p);return (F@fit(F,z)-z).ravel()
 r=least_squares(res,p0,bounds=bounds,gtol=1e-12,ftol=1e-12,xtol=1e-12,max_nfev=400)
 F=design(a,r.x); e=res(r.x).reshape(z.shape);loo=np.sqrt(np.mean((e/(1-np.diag(F@np.linalg.pinv(F)))[:,None])**2));print('cal',r.x,'rms',np.std(e),'loo',loo);return loo,lambda x:design(x,r.x)@fit(F,z)
bs=[];cs=[]
for fn,p0 in [(lambda x,p:np.c_[aff(x),np.sin(p[0]*x)], [2.9]),(lambda x,p:np.c_[aff(x),np.sin(p[0]*x),np.cos(p[0]*x)], [2.9]),(lambda x,p:np.c_[aff(x),np.sin(p[0]*x+p[1])], [2.9,0]),(lambda x,p:np.c_[np.ones(len(x)),x+p[1]*np.sin(p[0]*x)], [2.9,.4]),(lambda x,p:np.c_[aff(x),np.sin(p[0]*x)*np.roll(x,-1,axis=1)], [2.9]),(lambda x,p:np.c_[aff(x),np.sin(p*x)], [2.9]*4)]:
 bs.append(calibrate(b['u'],b['v'],fn,p0,([1]+[-10]*(len(p0)-1),[6]+[10]*(len(p0)-1))))
for fn,p0 in [(lambda x,p:aff(x+p[1]*lc(p[0]*x)),[1,.9]),(lambda x,p:aff(x+p[1]*np.cos(p[0]*x)),[1,-.9]),(lambda x,p:aff(x+p[1]*x*x/(1+p[0]*x*x)),[.2,.45]),(lambda x,p:np.c_[aff(x),lc(p[0]*x),x*x],[1]),(lambda x,p:aff(x+p[1]*x*x+p[2]*lc(p[0]*x)),[1,.22,.45])]:
 cs.append(calibrate(c['u'],c['v2'],fn,p0,([.05]+[-10]*(len(p0)-1),[6]+[10]*(len(p0)-1))))
bs.sort(key=lambda q:q[0]);cs.sort(key=lambda q:q[0]); C2=cs[0][1]; print('bests',bs[0][0],cs[0][0]);
base={'u':u,'w':w,**{f'B{i}':fn(u) for i,(_,fn) in enumerate(bs[:3])}}
seeds=dict(base)
for key,x in seeds.items():
 base['p'+key]=P(x);base['q'+key]=C2(x);base['pq'+key]=np.c_[P(x),C2(x)];base['xpq'+key]=np.c_[x,P(x),C2(x)]
for k in ('B0','B1'):
 base[k+'puqu']=np.c_[base[k],P(u),C2(u)];base[k+'pBu']=np.c_[base[k],P(base[k]),C2(u)];base[k+'puqB']=np.c_[base[k],P(u),C2(base[k])]
rows=[]; z=np.c_[y,T];
for name,x in base.items():
 for an,act in [('id',lambda x:x),('atan',np.arctan),('tanh',np.tanh),('asinh',np.arcsinh),('sin',np.sin),('inv3',lambda x:np.sinh(np.arcsinh(x)/3))]:
  for gain in ([1.] if an=='id' else [.65,1.25,2.5,4.,6.]):
   a=act(gain*x)
   for mode in range(5):
    fs=[a,np.c_[x,a],poly(a,2)[:,1:] if x.shape[1]<=8 else None,np.c_[a,pair(a)] if x.shape[1]==4 else None,np.c_[x,poly(a,2)[:,1:]] if x.shape[1]==4 else None][mode]
    if fs is None or fs.shape[1]>49:continue
    v0,v1=score(aff(fs));rows.append((*v0,*v1,name,an,gain,mode))
for j in (0,1,2,3): print('BEST',j,*sorted(rows,key=lambda q:q[j])[:12],sep='\n')
