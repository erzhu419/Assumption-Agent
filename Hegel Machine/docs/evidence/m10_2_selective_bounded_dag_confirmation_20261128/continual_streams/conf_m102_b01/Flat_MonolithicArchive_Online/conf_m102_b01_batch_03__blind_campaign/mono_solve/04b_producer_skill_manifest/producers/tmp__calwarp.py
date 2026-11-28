exec(open('/tmp/explore.py').read().split("a=s['panel_a']")[0]); from scipy.optimize import root
A=s['panel_a']; u,v=A['u'],A['v']; B=ls(aff(s['panel_b']['u']),s['panel_b']['v']); C=s['panel_c']; rows=np.argsort(np.linalg.norm(aff(u)@ls(aff(u),v)-v,axis=1))[:-3]; print('excluded',np.setdiff1d(np.arange(len(u)),rows)); print('excluded UV',np.c_[u,v][np.setdiff1d(np.arange(len(u)),rows)]);
def feat(u,d):
 if d=='cyclic':return np.c_[aff(u),u*u,u*np.roll(u,-1,axis=1),np.sin(u),np.tanh(u)]
 return poly(u,d)
def invert(x,w,d):
 return np.array([root(lambda z:(feat(z[None,:],d)@w)[0]-k,k).x for k in x]);
for d in [1,2,3,'cyclic']:
 F=feat(u,d); w=ls(F[rows],v[rows]); rf=np.linalg.norm(F@w-v,axis=1); hh=np.sum(F[rows]*np.linalg.pinv(F[rows]).T,axis=1); print('A',d,'res',np.sort(rf),'loo',np.quantile(rf[rows]/np.maximum(1-hh,1e-6),[.5,.9,.99])); print('coefs',w[:15]); z=invert(x,w,d); print('inverse',np.max(abs(feat(z,d)@w-x)),z.min(0),z.max(0));
 for n,g in [('z',z),('Bz',aff(z)@B)]+[(k+name,poly(src,2)@ls(poly(C['u'],2),C[k])) for k in ('v1','v2') for name,src in [('z',z),('Bz',aff(z)@B)]]:
  candidates=[]
  for gain in [.5,1,1.25,2,3,4,6,10]:
   for label,fn in [('tanh',np.tanh),('atan',np.arctan),('asinh',np.arcsinh),('sin',np.sin),('ic',lambda a: (2/np.sqrt(3))*np.sinh(np.arcsinh(1.5*np.sqrt(3)*a)/3)),('rat',lambda a:a/(1+a*a)),('soft',lambda a:a/np.sqrt(1+a*a))]:
    h=fn(g*gain)
    for k,f in enumerate([aff(h),np.c_[aff(g),h],poly(h,2),np.c_[poly(h,2),g],np.c_[poly(g,2),h]]):
     candidates.append((float(score(f,y)),float(score(f,q).mean()),label,gain,k))
  print(n,'y',min(candidates),'traj',min(candidates,key=lambda a:a[1]))
