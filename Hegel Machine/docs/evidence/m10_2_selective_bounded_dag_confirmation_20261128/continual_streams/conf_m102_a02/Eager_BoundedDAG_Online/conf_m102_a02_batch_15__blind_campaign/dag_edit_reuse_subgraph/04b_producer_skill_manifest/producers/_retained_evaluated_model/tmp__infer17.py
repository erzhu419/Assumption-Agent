import pickle, numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def fit(a,b): return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def aff(a): return np.column_stack([np.ones(len(a)),a])
u,v=map(s['panel_a'].get,('u','v')); keep=np.max(abs(v),axis=1)<3
print('A kept',keep.sum(),flush=True); u,v=u[keep],v[keep]; w=np.roll(u,2,axis=1)
F={'sin':np.sin,'tanh':np.tanh,'erf':erf,'rat':lambda z:z/(1+z*z),'asinh':np.arcsinh,'cube':lambda z:z**3,'atan':np.arctan}
def block(x,z,f,g,k): return np.column_stack([x*0+1,x,z,f(k[0]*x),g(k[1]*z)])
def evaluate(f,g,k,ret=False):
 ds=[block(u[:,j],w[:,j],f,g,k) for j in range(4)]; c=np.array([fit(d,v[:,j]) for j,d in enumerate(ds)]); r=np.column_stack([d@c[j]-v[:,j] for j,d in enumerate(ds)]); return (r,c) if ret else r.ravel()
results=[]
for fn,f in F.items():
 for gn,g in F.items():
  best=min([least_squares(lambda k:evaluate(f,g,k),k,bounds=(.15,6),max_nfev=50) for k in ([1,1],[2,2],[3.5,3.5])],key=lambda t:np.mean(t.fun**2)); results.append((np.mean(best.fun**2)**.5,fn,gn,best.x))
results.sort(key=lambda a:a[0]); print('A top',results[:12]); rms,fn,gn,k=results[0]; f,g=F[fn],F[gn]; _,c=evaluate(f,g,k,True); print('A coef',c)
def calibrate(x):
 def forward(a): return np.sum(c*np.stack([a*0+1,a,np.roll(a,2),f(k[0]*a),g(k[1]*np.roll(a,2))],axis=1),axis=1)
 return np.array([least_squares(lambda a:forward(a)-b,b/1.5,max_nfev=60).x for b in x])
def panel(p,key,f,bounds):
 u,v=p['u'],p[key]
 def design(z,k):return np.column_stack([aff(z),f(k*z)])
 def err(k):a=design(u,k);return np.mean((v-a@fit(a,v))**2)
 k=minimize_scalar(err,bounds=bounds,method='bounded').x
 print('panel',key,k,err(k)**.5)
 return lambda z:design(z,k)@fit(design(u,k),v)
z=calibrate(s['train']['x']); g=panel(s['panel_b'],'v',np.cos,(2.3,2.9))(z); h=panel(s['panel_c'],'v1',np.tanh,(1.5,3))(z); l=panel(s['panel_c'],'v2',np.cos,(.4,2))(z)
ys=np.column_stack([s['train']['y'],s['train']['traj']]); rng=np.random.default_rng(75); splits=[(np.arange(52),np.arange(52,72))]+[(a[:52],a[52:]) for a in [rng.permutation(72) for _ in range(12)]]
def score(a):
 e=np.array([np.abs(a[j]@fit(a[i],ys[i])-ys[j]).mean(0) for i,j in splits]); return np.round([e[0,0],e[0,1:].mean(),e[:,0].max(),e[:,1:].mean(1).max()],5)
from itertools import combinations_with_replacement
for name,a in [('z',z),('g',g),('h',h),('l',l),('zg',np.column_stack([z,g])),('zghl',np.column_stack([z,g,h,l])),('cos',np.cos(2.6*z))]:
 print(name,score(aff(a))); 
 if a.shape[1]<=8: print(name,'quad',score(np.column_stack([aff(a)]+[a[:,i]*a[:,j] for i,j in combinations_with_replacement(range(a.shape[1]),2)])))
for base,D in [('zg',aff(np.column_stack([z,g]))),('zghl',aff(np.column_stack([z,g,h,l])))]:
 res=[]
 for a,an in [(z,'z'),(g,'g'),(h,'h'),(l,'l')]:
  for b,bn in [(z,'z'),(g,'g'),(h,'h'),(l,'l')]:
   for j in range(4):
    B=np.column_stack([D,a*np.roll(b,j,axis=1)]); S=score(B); res.append((an+bn+str(j),S))
 print(base,'top y',sorted(res,key=lambda x:x[1][2])[:8]);print(base,'top t',sorted(res,key=lambda x:x[1][3])[:8]);
np.savez('/tmp/features17.npz',z=z,g=g,h=h,l=l,targets=ys)
