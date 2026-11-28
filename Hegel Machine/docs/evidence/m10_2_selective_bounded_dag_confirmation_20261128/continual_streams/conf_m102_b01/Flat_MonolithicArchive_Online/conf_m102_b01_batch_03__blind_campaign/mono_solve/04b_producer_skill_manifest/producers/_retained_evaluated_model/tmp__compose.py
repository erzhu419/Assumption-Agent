exec(open('/tmp/explore.py').read().split("a=s['panel_a']")[0]); from scipy.optimize import minimize_scalar
A=s['panel_a']; B=s['panel_b']; C=s['panel_c']; aa=[]
for u,v in zip(A['u'].T,A['v'].T):
 f=aff(u[:,None]); k=np.arange(len(u));
 for _ in range(8):
  w=ls(f[k],v[k]); k=np.argsort(abs(f@w-v))[:42]
 aa.append(w)
a,b=np.array(aa).T; z=(x-a)/b; B=ls(aff(B['u']),B['v']); print('cal',a,b); print('B',B)
for n in ('v1','v2'):
 f=poly(C['u'],2); w=ls(f,C[n]); print(n,'coefs\n',w,'err',np.std(f@w-C[n],0))
DD={}
for name,w in [('raw',x),('cal',z),('Br',aff(x)@B),('Bz',aff(z)@B)]:
 DD[name]=w
 for n in ('v1','v2'): DD[name+n]=poly(w,2)@ls(poly(C['u'],2),C[n]); DD[name+n+'B']=aff(DD[name+n])@B
for name,w in DD.items():
 best=[]
 for g in [.3,.7,1,1.5,2.5,4,7]:
  for fn in (np.tanh,np.arctan,np.arcsinh,np.sin,lambda u:np.sinh(np.arcsinh(u)/3)):
   v=fn(w*g)
   for kind, f in enumerate([aff(v),np.c_[aff(w),v],poly(v,2),np.c_[poly(v,2),w],np.c_[poly(w,2),v]]):best.append((score(f,y),score(f,q).mean(),g,fn.__name__,kind))
 print(name,'raw',score(aff(w),y),score(aff(w),q).mean(),'best y/t',min(best),min(best,key=lambda v:v[1]));
