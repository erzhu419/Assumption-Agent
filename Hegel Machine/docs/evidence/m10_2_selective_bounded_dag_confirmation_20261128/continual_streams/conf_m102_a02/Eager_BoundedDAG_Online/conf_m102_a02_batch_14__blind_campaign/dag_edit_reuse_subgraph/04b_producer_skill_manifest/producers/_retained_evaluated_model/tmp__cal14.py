import numpy as np,pickle,itertools
from scipy.optimize import least_squares,minimize_scalar
from scipy.special import erf
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; t=s['train']; x=t['x']; y=np.c_[t['y'],t['traj']]
def ols(a,b):return np.linalg.lstsq(a,b,rcond=1e-10)[0]
def mons(p,d):return [()] + [ix for k in range(1,d+1) for ix in itertools.combinations_with_replacement(range(p),k)]
def des(x,terms):return np.column_stack([np.prod(x[:,i],axis=1) for i in terms])
p=s['panel_a'];u=p['u'];v=p['v'];mask=np.max(abs(v),axis=1)<4; U=u[mask];V=v[mask]; terms=mons(4,4)
for d in (1,2,3,4):
 a=des(U,mons(4,d));co=ols(a,V);print('panel_a poly',d,np.std(a@co-V,axis=0))
cs=[];ts=[]
for i in range(4):
 a=des(U,terms); ids=[0]+list(range(1,5)); f=V[:,i]; print('coord',i)
 for step in range(12):
  r=f-a[:,ids]@ols(a[:,ids],f); b=a-a[:,ids]@ols(a[:,ids],a); norms=np.linalg.norm(b,axis=0); score=abs(r@b)/np.maximum(norms,1e-10);score[ids]=0;j=np.argmax(score);ids.append(j); c=ols(a[:,ids],f); print(terms[j],round(np.std(a[:,ids]@c-f),7),np.round(c,4))
  if np.std(a[:,ids]@c-f)<0.001:break
 cs.append(c);ts.append([terms[j] for j in ids])
def zmap(x):
 def fun(a):return np.array([sum(w*np.prod(a[list(ix)]) for w,ix in zip(c,tt)) for c,tt in zip(cs,ts)])
 return np.array([least_squares(lambda a:fun(a)-b,b,gtol=1e-11).x for b in x])
z=zmap(x);print('inverse train range and panel check',z.min(),z.max(),np.max(abs(zmap(V)-U)))
def panel(p,key,f):
 u,v=p['u'],p[key]; design=lambda a,w:np.c_[np.ones(len(a)),a,f(w*a)]; loss=lambda w:np.mean((lambda a:a@ols(a,v)-v)(design(u,w))**2);w=minimize_scalar(loss,bounds=(.2,4),method='bounded').x; a=design(u,w);co=ols(a,v);print('panel',key,w,loss(w)**.5,np.round(co,3));return lambda z:design(z,w)@co
g=panel(s['panel_b'],'v',erf)(z);j=panel(s['panel_c'],'v1',lambda a:a/np.sqrt(1+a*a))(z);k=panel(s['panel_c'],'v2',lambda a:np.logaddexp(0,a))(z)
np.savez('/tmp/train14.npz',z=z,g=g,j=j,k=k,y=y)
