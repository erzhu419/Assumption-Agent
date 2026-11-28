exec(open('/tmp/readout_probe.py').read().split('for name,p in')[0])
from itertools import combinations,combinations_with_replacement
from scipy.linalg import qr
lat=np.cbrt(q); raw=s['train']['x']; tar=np.c_[y,t]; c=s['panel_c'];
def parts(x):
 return {**{str(i):x[:,i] for i in range(4)},**{f'{i}{sg}{j}':x[:,i]+sg*x[:,j] for i,j in combinations(range(4),2) for sg in [-1,1]},**{f'{i}*{j}':x[:,i]*x[:,j] for i,j in combinations_with_replacement(range(4),2)}}
def blocks(x):
 out={}; p=parts(x)
 for nm,z in p.items():
  for g in [1,1.3,2.15,2.6,4.3,5.2]:
   for nm2,fn in [('sin',np.sin),('cos',np.cos),('atan',np.arctan),('tanh',np.tanh),('rat',lambda a:a/(1+a*a)),('evenrat',lambda a:1/(1+a*a))]:out[nm2,nm,g]=fn(g*z)
 return out
def sparse(x,Y,idx,jdx,base,groups,steps):
 m=np.column_stack(list(groups.values())); mu=m[idx].mean(0); m-=mu; m/=np.maximum(np.linalg.norm(m[idx],axis=0),1e-10); a=base.copy(); K=list(groups); sel=[]
 for it in range(steps):
  w=np.linalg.lstsq(a[idx],Y[idx],rcond=1e-9)[0]; r=Y[idx]-a[idx]@w
  if it in (0,1,2,3,5,8,12,17,23,31):print(it,np.round(np.mean(abs(r),axis=0),5),np.round(np.mean(abs(Y[jdx]-a[jdx]@w),axis=0),5),sel[-4:])
  basis=qr(a[idx],mode='economic')[0]; mm=m[idx]-basis@(basis.T@m[idx]); cor=mm.T@r/np.maximum(np.linalg.norm(mm,axis=0).reshape((-1,)+(1,)*(r.ndim-1)),1e-8); k=np.argmax(np.abs(cor) if r.ndim==1 else np.linalg.norm(cor,axis=1));sel.append(K[k]); a=np.c_[a,m[:,k]]; m[:,k]=0
for key,fn in [('v1',lambda x:aff(np.arctan(2.15*x))),('v2',lambda x:np.c_[aff(x),np.cos(1.3*x)]),('v2',lambda x:np.c_[aff(x),x**2,x**3,np.cos(2.6*x)])]:
 print('CAL',key); xx=c['u']; sparse(xx,c[key],np.arange(18),np.arange(18,24),fn(xx),blocks(xx),9)
for name,x in [('lat',lat),('raw',raw),('q',q),('B',aff(np.cos(f*lat))@co)]:
 print('LABELS',name); sparse(x,tar[:,[0,1,8]],np.arange(54),np.arange(54,72),aff(np.cos(f*x)),blocks(x),32)
print('time',s['eval_times']); print('correlation',np.corrcoef(tar.T)[0]); print('trajectory examples',np.column_stack([y,t[:,0],t[:,-1]])[:20])
