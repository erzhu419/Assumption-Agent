exec(open('/tmp/probe12.py').read().split('for key,tar')[0])
from scipy.optimize import minimize_scalar
for key,tar in [('panel_b','v'),('panel_c','v1'),('panel_c','v2')]:
 u=s[key]['u']; v=s[key][tar]; print(key,tar,'range',v.min(0),v.max(0))
 for j in range(4):
  y=v[:,j]; out=[]
  for d in [1,2]:
   f=poly(u,d); base=np.linalg.norm(f@ls(f,y)-y);out.append((base,d,'direct',0))
   for name,fn in [('atanh',lambda a,k:np.arctanh(a/k)),('asin',lambda a,k:np.arcsin(a/k)),('aratsat',lambda a,k:a/np.sqrt(1-(a/k)**2)),('atan',lambda a,k:np.tan(a/k)),('cubic',lambda a,k:a+k*a**3)]:
    lo=max(abs(y))*1.00001 if name!='cubic' else -3; hi=lo*8 if name!='cubic' else 3
    def loss(k):
     t=fn(y,k); return np.linalg.norm(f@ls(f,t)-t)*np.linalg.norm(y)/np.linalg.norm(t)
    p=minimize_scalar(loss,bounds=(lo,hi),method='bounded');out.append((p.fun,d,name,p.x))
  print(j,sorted(out)[:7]); print('linear',np.round(ls(aff(u),y),3))
# search paired sums/differences and power, periodic, and saturating coordinatewise terms
for key,tar in [('panel_b','v'),('panel_c','v1'),('panel_c','v2'),('train','y'),('train','traj')]:
 u=s[key]['u'] if 'u' in s[key] else q; y=s[key][tar]; folds=np.array_split(np.arange(len(u)),6); res=[]
 for h in [u,u+np.roll(u,-1,axis=1),u-np.roll(u,-1,axis=1),u*np.roll(u,-1,axis=1),np.roll(u,-1,axis=1)**2-u**2]:
  for nm, fn in [('sin',np.sin),('cos',np.cos),('tanh',np.tanh),('abs',np.abs),('power2',lambda x:x**2),('exp',np.exp),('sat',lambda x:x/(1+x*x))]:
   for k in [.5,1,1.5,2,2.15,3,4,5,6]:
    for base in [aff(u),poly(u,2)]:
     f=np.c_[base,fn(k*h)]; res.append((cv(f,y,folds),nm,k,base.shape[1],len(res)//126))
 print('grammar',key,tar,sorted(res)[:8])
