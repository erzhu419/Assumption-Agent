from operators import *
from scipy.special import erf,hyp2f1
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from importlib.util import spec_from_file_location,module_from_spec
spec=spec_from_file_location('baseline','/holdout_task/batch_solver.py'); old=module_from_spec(spec); spec.loader.exec_module(old)
T=S['train']; X=old._calibrate(S['panel_a'],T['x']); y=T['y']; Q=T['traj']; times=S['eval_times']; n=len(X)
u=S['panel_b']['u']; v=S['panel_b']['v']; cu=S['panel_c']['u']; cv=S['panel_c']['v1']
def db(x,k): return np.c_[aff(x),erf(k*x)]
def dc(x,k,p): return np.c_[aff(x),x*hyp2f1(.5,p,1.5,-(k*x)**2)]
a=least_squares(lambda p:(dc(cu,*p)@fit(dc(cu,*p),cv)-cv).ravel(),[7,.065],bounds=([2,.016],[14,.4])); ck=a.x; cw=fit(dc(cu,*ck),cv)
bk=minimize_scalar(lambda k:np.mean((db(u,k)@fit(db(u,k),v)-v)**2),bounds=(.7,2.1),method='bounded').x; bw=fit(db(u,bk),v)
print('panel B linear vs nonlinear coeff ratios',bw[5:]/bw[1:5]); print('panel C residual rms',np.std(a.fun),'rank',np.linalg.svd(a.fun.reshape(-1,4),compute_uv=False),'linear vs nonlinear coef ratios',cw[5:]/cw[1:5])
ix,jx=np.triu_indices(4)
def quad(x): return np.c_[aff(x),x[:,ix]*x[:,jx]]
def pred(i,j,weight,c_free,denom):
 def ts(z): return -np.expm1(-np.exp(z[:,None])*times)
 def calc(p,xx):
  k,w=p[0],p[1:37].reshape(9,4); h=p[37:39]; d=p[39] if denom else 0; cp=p[40:42] if c_free else ck; cfit=fit(dc(cu,*cp),cv) if c_free else cw
  b=db(xx,k)@w; c=dc(b,*cp)@cfit
  return quad(c)/(1+d*np.sum(c*c,axis=1,keepdims=True)),aff(c),ts(h)
 def res(p):
  f,g,h=calc(p,X[i]); ey=f@fit(f,y[i])-y[i]; et=g@fit(g,Q[i]@np.linalg.pinv(h))@h-Q[i]; ep=db(u,p[0])@p[1:37].reshape(9,4)-v
  return np.r_[ey/.014,et.ravel()/.018,(ep*weight/.0055).ravel(),((dc(cu,*p[40:42])@fit(dc(cu,*p[40:42]),cv)-cv)/.0007).ravel() if c_free else []]
 opt=np.r_[bk,bw.ravel(),np.log([.415,1.81]),.025,ck] if c_free else np.r_[bk,bw.ravel(),np.log([.415,1.81]),.025]
 if weight:
  lo=np.r_[.8,[-np.inf]*36,-1.7,.15,0.,[2,.016] if c_free else []]; hi=np.r_[2.3,[np.inf]*36,-.25,1.3,.5,[14,.4] if c_free else []]
  opt=least_squares(res,opt,bounds=(lo,hi),max_nfev=24).x
 else: opt[39]=0
 f,g,h=calc(opt,X); py=f[j]@fit(f[i],y[i]); pt=g[j]@fit(g[i],Q[i]@np.linalg.pinv(h))@h
 return [np.mean(abs(py-y[j])),np.mean(abs(pt-Q[j]))],opt[[0,37,38,39]],np.std(db(u,opt[0])@opt[1:37].reshape(9,4)-v)
near=np.partition(cdist(X,X),5,axis=1)[:,5]; rad=np.linalg.norm(X,axis=1)
sp=[(np.arange(52),np.arange(52,n)), *KFold(4,shuffle=True,random_state=91).split(X), *[(np.argsort(xx)[:-15],np.argsort(xx)[-15:]) for xx in (near,rad,*X.T,*(-X).T)]]
for cfg in [(0,False,False),(.5,False,False),(1,False,False),(2,False,False),(1,True,False),(1,False,True)]:
 out=[pred(i,j,*cfg) for i,j in sp]; e=np.asarray([x[0] for x in out]); print(cfg,'public',e[0],'random max',e[1:5].max(0),'stress mean',e[5:].mean(0),'stress max',e[5:].max(0),'params public',out[0][1:]); print('worst slices',np.argmax(e[5:],axis=0))
