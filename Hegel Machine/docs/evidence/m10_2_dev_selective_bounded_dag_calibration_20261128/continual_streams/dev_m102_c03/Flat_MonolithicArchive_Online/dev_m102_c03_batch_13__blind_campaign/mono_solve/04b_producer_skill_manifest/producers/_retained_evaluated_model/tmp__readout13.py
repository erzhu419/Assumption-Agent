import numpy as np,pickle
from scipy.optimize import least_squares
from itertools import combinations_with_replacement
s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
def ls(a,b):return np.linalg.lstsq(a,b,rcond=1e-9)[0]
def aff(a):return np.c_[np.ones(len(a)),a]
def cyc(a):return a*np.roll(a,-1,axis=1)
def poly(a,d):return np.column_stack([np.prod(a[:,p],axis=1) for p in combinations_with_replacement(range(a.shape[1]),d)])
def cross(a,b):return (a[:,:,None]*b[:,None,:]).reshape(len(a),-1)
def operators(s,x):
 p=s['panel_a']; u=p['u']; v=p['v']; co=[]
 for j in range(4):
  a=aff(u[:,j:j+1]); e=abs(a@ls(a,v[:,j])-v[:,j]); keep=e<max(.001,4*np.median(e)); co.append(ls(a[keep],v[keep,j]))
 off,gain=np.array(co).T; z=(x-off)/gain
 p=s['panel_b']; B=ls(aff(p['u']),p['v']); w=aff(z)@B
 p=s['panel_c']; u=p['u']; fs=[]
 for k in [1,2]:
  v=p['v'+str(k)]
  def design(a,r):
   if k==1:return np.c_[aff(a),a**3,a/(1+r*abs(a))]
   return np.c_[aff(a),cyc(a),a**3,np.tanh(a*r)]
  def coef(r):return ls(design(u,r),v)
  r=least_squares(lambda r:(design(u,r)@coef(r)-v).ravel(),[.66 if k==1 else 1.2],bounds=(.1,3),gtol=1e-12).x
  c=coef(r); fs.append(lambda a,design=design,r=r,c=c,k=k: (np.c_[aff(a),a**3,a/(1+r*abs(a))] if k==1 else np.c_[aff(a),cyc(a),a**3,np.tanh(a*r)])@c)
  print('branch',k,r, np.sqrt(np.mean((design(u,r)@c-v)**2)),c.round(3))
 return z,w,fs

def time(s,ids):
 tr=s['train']['traj'][ids]; t=s['eval_times']; modes=lambda r:-np.expm1(-t[:,None]*r)
 r=least_squares(lambda r:(modes(r)@ls(modes(r),tr.T)-tr.T).ravel(),[.4,1.85],bounds=([.1,.9],[.85,4])).x
 return modes(r)

def designs(z,w,fs):
 D={}
 for name,a in [('z',z),('w',w)]:
  b,c=[f(a) for f in fs]; x=cross(b,c); pair=np.sum(b*c,axis=1); u=np.c_[b,c]; v=np.c_[a,u]
  for k,f in {'a':a,'bc':u,'pair':np.c_[u,pair],'diag':np.c_[u,b*c],'allcross':np.c_[u,x],'bpoly':np.c_[b,poly(b,2)],'cpoly':np.c_[c,poly(c,2)],'bcquad':np.c_[u,poly(u,2)],'abc':v,'abc_cross':np.c_[v,x],'abc_quad':np.c_[v,poly(v,2)],'apoly2':np.c_[a,poly(a,2)],'apoly3':np.c_[a,poly(a,2),poly(a,3)],'apoly4':np.c_[a,poly(a,2),poly(a,3),poly(a,4)],'abc_pair':np.c_[v,pair],'abc_pair_poly':np.c_[v,pair,poly(a,2),poly(a,3)],'abc_cross_poly':np.c_[v,x,poly(a,2),poly(a,3)]}.items():D[name+'_'+k]=f
 b,c=fs[0](z),fs[1](w); u=np.c_[b,c]; D['mixed']=u; D['mixed_pair']=np.c_[u,np.sum(b*c,axis=1)]; D['mixed_cross']=np.c_[u,cross(b,c)]; D['mixed_quad']=np.c_[u,poly(u,2)]
 b,c=fs[0](w),fs[1](z); u=np.c_[b,c]; D['rev']=u; D['rev_pair']=np.c_[u,np.sum(b*c,axis=1)]; D['rev_cross']=np.c_[u,cross(b,c)]; D['rev_quad']=np.c_[u,poly(u,2)]
 return D

def experiments():
 t=s['train']; ids=np.arange(52); j=np.arange(52,72); z,w,fs=operators(s,t['x']); D=designs(z,w,fs); q=time(s,ids); Y=np.c_[t['y'],t['traj']]
 for name,X in D.items():
  a=aff((X-X[ids].mean(0))/np.maximum(X[ids].std(0),1e-9)); A=a[ids]; U,E,V=np.linalg.svd(A,full_matrices=False); results=[]
  for r in [0,1e-6,1e-5,1e-4,.001,.01,.1,1]:
   filt=np.where(E>1e-8,E/(E**2+max(r,1e-20)),0); W=(V.T*filt)@(U.T@Y[ids]); p=a[j]@W; p[:,1:]=(p[:,1:]@q)@np.linalg.inv(q.T@q)@q.T; er=np.mean(abs(p-Y[j]),axis=0); inn=np.sqrt(np.mean((A@W-Y[ids])**2,axis=0)); results.append((er[0],er[1:].mean(),r,inn[0],inn[1:].mean()))
  print(name, np.round(results[0],5),np.round(min(results,key=lambda x:x[0]),5),np.round(min(results,key=lambda x:x[1]),5))
if __name__=='__main__':experiments()
