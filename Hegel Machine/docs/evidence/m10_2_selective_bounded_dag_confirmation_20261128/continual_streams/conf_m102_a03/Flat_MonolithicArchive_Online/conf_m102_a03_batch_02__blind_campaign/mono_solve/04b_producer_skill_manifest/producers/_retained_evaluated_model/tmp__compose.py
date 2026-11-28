exec(open('/tmp/discover.py').read().split('A=old._calibration')[0])
from scipy.optimize import least_squares
A=old._calibration(s['panel_a']); z=old._decode(x,A); pb=s['panel_b']; B=fit(poly(pb['u'],1),pb['v']); w=poly(z,1)@B
pc=s['panel_c']; U=pc['u']; V1=pc['v1']; V2=pc['v2']
def models():
 fs=[('tanh',np.tanh,2.2),('rat',lambda a:a/(1+a*a),1.35),('atan',np.arctan,2.2),('sin',np.sin,3.4),('soft',lambda a:a/np.sqrt(1+a*a),2.2),('erf',__import__('scipy').special.erf,1.5)]
 for name,fun,start in fs:
  for add in [False,True]:
   for variant in [0,1]:
    def basis(f,p): return np.c_[poly(f,1),fun(p[0]*f),np.sin(p[-1]*f)] if add else np.c_[poly(f,1),fun(p[0]*f), np.cos(p[0]*f)] if variant else np.c_[poly(f,1),fun(p[0]*f)]
    if add and variant:continue
    def res(p):
     d=basis(U,p);return (d@fit(d,V1)-V1).ravel()
    p=least_squares(res,[start,1.3] if add else [start],bounds=(.2,7),max_nfev=200).x
    d=basis(U,p); c=fit(d,V1); err=d@c-V1; print(name,add,variant,p,'v1 rms/loo',np.mean(err**2)**.5,np.mean(abs(err)/(1-np.diag(d@np.linalg.pinv(d)))[:,None]))
    yield (name+str(add)+str(variant),lambda f,b=basis,p=p,c=c:b(f,p)@c)
def d2(f,p):return np.c_[poly(f,1),np.cos(p[0]*f)]
def r2(p):
 d=d2(U,p);return (d@fit(d,V2)-V2).ravel()
p=least_squares(r2,[1.3]).x; c=fit(d2(U,p),V2); print('v2',p,np.mean(r2(p)**2)**.5,c)
D2=lambda f:d2(f,p)@c
allres=[]
for name,D1 in models():
 for label,f in [('z',z),('w',w),('x',x)]:
  a,b=D1(f),D2(f)
  for typ,base in [('C1',a),('C2',b),('C12',np.c_[a,b]),('L+C1',np.c_[f,a]),('L+C2',np.c_[f,b]),('L+C12',np.c_[f,a,b]),('Csum',a+b),('Cdiff',a-b),('L+Csum',np.c_[f,a+b]),('L+Cdiff',np.c_[f,a-b]),('L+prod',np.c_[f,a*b]),('L+C12+prod',np.c_[f,a,b,a*b]),('C1quad',poly(a,2)),('C2quad',poly(b,2)),('L+C12+p2',np.c_[f,a,b,a*b,a*a,b*b])]:
   d=poly(base,1); pred=d@fit(d[:52],y[:52]); e=np.mean(abs(pred-y)[:52]); v=np.mean(abs(pred-y)[52:]);allres.append((v,e,name,label,typ))
print('by validation',*sorted(allres)[:20],sep='\n');print('by training',*sorted(allres,key=lambda q:q[1])[:20],sep='\n')
