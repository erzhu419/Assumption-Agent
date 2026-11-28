import pickle, numpy as np, runpy, glob
from scipy.optimize import minimize_scalar
F=runpy.run_path('/holdout_task/_frozen_nodes/node_00.py'); A=runpy.run_path(sorted(glob.glob('/tmp/hegel_continual_archive/*solver.py'))[-1]); s=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; tr=s['train']; x=tr['x']; z=tr['traj']; a=A['affine']; fit=A['fit']; gp=A['gp']; cx,v0=F['_coordinates'](s,x).T.reshape(2,4,len(x)).transpose(0,2,1); b,c=s['panel_b'],s['panel_c']; bx=a(x)@fit(a(b['u']),b['v']); u=c['u'];
def cal(v,form,bounds):
 def obj(k):
  X=form(u,k); return np.linalg.norm(X@fit(X,v)-v)
 k=minimize_scalar(obj,bounds=bounds,method='bounded',options={'xatol':1.e-12}).x; print('panel cal',k,obj(k)); return k,fit(form(u,k),v)
vform=lambda q,k:a(np.arctan(k*q)); wform=lambda q,k:np.c_[a(q),q*q/(1+k*q*q)]; k,V=cal(c['v1'],vform,(.1,6)); h,W=cal(c['v2'],wform,(0,3)); w=lambda q:wform(q,h)@W; v=lambda q:vform(q,k)@V
Q={'raw':x,'A':cx,'B':bx,'Av':np.c_[cx,v(cx)],'Aatan':np.c_[cx,vform(cx,k)[:,1:]],'Bw':np.c_[bx,w(bx)],'Bsq':wform(bx,h)[:,1:],'Bwonly':w(bx),'Xw':np.c_[x,w(x)],'Aw':np.c_[cx,w(cx)],'Bv':np.c_[bx,v(bx)],'BAv':np.c_[bx,v(cx)],'Bwv':np.c_[bx,w(bx),v(cx)]}
from scipy.spatial.distance import cdist
from scipy.linalg import solve
from itertools import combinations_with_replacement
splits=[np.arange(52,72),np.arange(0,20),np.arange(26,46),np.random.default_rng(726).permutation(len(x))[:20]]
def poly(q,d):
 return np.column_stack([np.prod(q[:,j],axis=1) for k in range(1,d+1) for j in combinations_with_replacement(range(q.shape[1]),k)])
def kern(q,kind,l):
 q=(q-q.mean(0))/q.std(0); D=(q[:,None]-q[None])**2
 if kind=='add': return np.exp(-D/2/l**2).sum(2)
 if kind=='joint': return np.exp(-D.sum(2)/2/l**2)
 if kind=='pair':
  E=np.exp(-D/2/l**2); return sum(E[:,:,i]*E[:,:,j] for i in range(q.shape[1]) for j in range(i+1,q.shape[1]))
def gram(F):
 F=F/(np.sqrt(np.mean(F**2,0))+1e-12); return F@F.T/F.shape[1]
res=[]
def evalG(name,G):
 for lam in [1e-6,1e-4,.001,.01,.1]:
  err=[]
  for va in splits:
   ti=np.setdiff1d(np.arange(len(x)),va); _,_,vt=np.linalg.svd(z[ti],full_matrices=False); yy=z[ti]@vt[:2].T
   K=G[np.ix_(ti,ti)]+lam*np.eye(len(ti)); V=G[np.ix_(va,ti)]; pred=V@solve(K,yy,assume_a='pos')@vt[:2]
   err.append(np.abs(pred-z[va]).mean())
  res.append((np.mean(err),name,lam,np.round(err,5).tolist()))
for key,q in {'x':x,'a':cx,'b':bx,'c':v(cx),'w':w(bx),'xc':np.c_[x,v(cx)],'xw':np.c_[x,w(bx)],'xb':np.c_[x,bx],'all':np.c_[x,bx,v(cx),w(bx)]}.items():
 q=(q-q.mean(0))/q.std(0)
 for d in range(1,5):
  G=1+gram(poly(q,d)); evalG((key,'poly',d),G)
 for l in [.4,.7,1,1.5,2,3,5]:
  for base in ['add','joint','pair']:
   G=kern(q,base,l); evalG((key,base,l),1+G+.5*gram(poly(q,2)))
 for ff in [.5,1,1.5,2,2.5,3]:
  G=1+gram(np.c_[poly(q,2),np.sin(ff*q),np.cos(ff*q)]); evalG((key,'trig',ff),G)
print('Best by mean'); print(*sorted(res)[:24],sep='\n'); print('Best by max');print(*sorted(res,key=lambda t:max(t[-1]))[:16],sep='\n')
for i in range(4):
 va=splits[i]; ti=np.setdiff1d(np.arange(len(x)),va); _,_,vt=np.linalg.svd(z[ti],full_matrices=False); y=tr['y']; print('scalar gp',i,np.mean(abs(gp(Q['Aatan'][ti],Q['Aatan'][va],y[ti],True)-y[va])))
