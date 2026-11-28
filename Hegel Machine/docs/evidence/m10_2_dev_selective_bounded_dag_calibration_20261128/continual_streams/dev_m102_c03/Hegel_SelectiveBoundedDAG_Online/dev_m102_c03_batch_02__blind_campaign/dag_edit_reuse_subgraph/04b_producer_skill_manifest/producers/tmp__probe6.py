import pickle,numpy as np
from scipy.optimize import least_squares, minimize_scalar
from itertools import combinations_with_replacement as cwr
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; u,v,w=[s['panel_c'][i] for i in ('u','v1','v2')]; p=s['panel_b']; B=np.linalg.lstsq(np.c_[np.ones(len(p['u'])),p['u']],p['v'],rcond=None)[0]; x=s['train']['x']; p=s['panel_a']; k=np.linalg.norm(p['v']-p['u'],axis=1)<2; A=np.linalg.lstsq(np.c_[np.ones(k.sum()),p['u'][k]],p['v'][k],rcond=None)[0]; z=(x-A[0])@np.linalg.inv(A[1:]); t=s['eval_times']; q=-np.expm1(-t[:,None]*[.41250015,1.80994588]); Y=np.c_[s['train']['y'],np.linalg.lstsq(q,s['train']['traj'].T,rcond=None)[0].T]
def h(z): return np.c_[np.ones(len(z)),z]
def poly(x,d):return np.column_stack([np.ones(len(x))]+[np.prod(x[:,idx],axis=1) for j in range(1,d+1) for idx in cwr(range(x.shape[1]),j)])
def fit(f,y,k=slice(None)):return np.linalg.lstsq(f[k],y[k],rcond=None)[0]
def rms(a):return (a**2).mean()**.5
for xx,n in [(u,'u'),(u@B[1:]+B[0],'Bu')]:
 F=h(xx); L=fit(F,v)
 for name,fun in [('tanh',np.tanh),('sin',np.sin),('atan',np.arctan),('soft',lambda x:x/np.sqrt(1+x*x))]:
  for m in (0,1,2):
   def pred(P,zz):
    if m==0:return fun(h(zz)@P[:20].reshape(5,4))*P[20:24]+P[24:]
    if m==1:return h(fun(zz*P[:4]+P[4:8]))@P[8:].reshape(5,4)
    return h(fun(h(zz)@P[:20].reshape(5,4)))@P[20:].reshape(5,4)
   base=(np.r_[L.ravel(),np.ones(4),np.zeros(4)] if m==0 else np.r_[np.ones(4),np.zeros(4),L.ravel()] if m==1 else np.r_[L.ravel(),np.zeros(4),np.eye(4).ravel()]); j=np.arange(len(u))%5!=0
   res=least_squares(lambda a:(pred(a,xx[j])-v[j]).ravel(),base,max_nfev=700); print('v1',n,name,m,round(rms(res.fun),6),round(rms(pred(res.x,xx[~j])-v[~j]),6));
   if n=='u' and rms(pred(res.x,xx[~j])-v[~j])<.006: print('PARAM',np.round(res.x,4))
for xx,n in [(u,'u'),(v,'v1'),(np.c_[u,v],'uv')]:
 if n=='uv':continue
 j=np.arange(len(u))%5!=0
 for name,fun in [('diag2',lambda a:a*a),('sat',lambda a:np.exp(-.3*a*a)),('cos',lambda a:np.cos(a)),('tanh2',lambda a:np.tanh(a)**2)]:
  for lin in (False,True):
   d=np.c_[h(xx),fun(xx)] if lin else h(fun(xx)); print('v2',n,name,lin,round(rms(d[j]@fit(d,w,j)-w[j]),5),round(rms(d[~j]@fit(d,w,j)-w[~j]),5))
 if n=='u':
  print('even c', np.round(fit(h(u*u),w),3))
for n,xx in [('raw',x),('z',z),('Bz',z@B[1:]+B[0])]:
 for d in [1,2,3]:
  F=poly(xx,d); b=fit(F,Y); E=Y-F@b; jj=np.arange(len(z))>=52; p=F[jj]@fit(F,Y,~jj);print('read',n,d,'train RMS',np.round((E**2).mean(0)**.5,4),'cv MAE',np.round(np.abs(p-Y[jj]).mean(0),4));
  if d==2 and n!='raw': print('coefs',np.round(b,3).T)
print('Y range',Y.min(0),Y.max(0))
