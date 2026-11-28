import numpy as np,pickle,itertools
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; np.set_printoptions(precision=5,suppress=True,linewidth=160)
def aff(u): return np.column_stack((np.ones(len(u)),u))
def lib(u):
 D={'1':np.ones(len(u))}; d=u.shape[1]
 for j in range(d):
  x=u[:,j]
  for n,f in [('x',lambda x:x),('q',lambda x:x*x),('cube',lambda x:x**3),('tanh',np.tanh),('sin',np.sin),('cos',np.cos),('atan',np.arctan),('soft',lambda x:x/(1+x*x)),('sat',lambda x:x/np.sqrt(1+x*x)),('abs',np.abs)]: D[f'{n}{j}']=f(x)
 for j,k in itertools.combinations(range(d),2):
  x,y=u[:,j],u[:,k]; D[f'prod{j}{k}']=x*y
  for a,z in [('+',x+y),('-',x-y)]:
   for n,f in [('tanh',np.tanh),('sin',np.sin),('cos',np.cos)]:D[f'{n}{j}{a}{k}']=f(z)
 return D
for p in ['panel_a','panel_b','panel_c']:
 u=s[p]['u']; X=aff(u); D=lib(u)
 for key,v in s[p].items():
  if key=='u':continue
  B=np.linalg.lstsq(X,v,rcond=None)[0]; R=v-X@B
  print(p,key,'range',u.min(0),u.max(0),'aff',B,'rms',np.sqrt(np.mean(R**2,0)), 'max',np.max(abs(R),0))
  for j in range(v.shape[1]):
   names=list(D); T=np.column_stack(list(D.values())); inds=[names.index(k) for k in ['1','x0','x1','x2','x3']]; y=v[:,j]; r=R[:,j]; print('target',j,end=' ')
   for _ in range(5):
    Z=T[:,inds]; Q=np.linalg.qr(Z)[0]; W=T-Q@(Q.T@T); c=np.abs(r@W)/np.maximum(np.linalg.norm(W,axis=0),1e-7); c[inds]=0; i=np.argmax(c); inds.append(i); Z=T[:,inds]; b=np.linalg.lstsq(Z,y,rcond=None)[0]; r=y-Z@b; print(names[i],round(b[-1],5),round(np.sqrt(np.mean(r*r)),6),end='; ')
   print()
print('telemetry',s['telemetry_vec'])
