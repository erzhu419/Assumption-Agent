exec(open('/tmp/probe_dag.py').read().split('for k in (')[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
p=s['panel_a']; a=p['u']; v=p['v']; keep=np.linalg.norm(v-a,axis=1)<2.5; a=a[keep]; v=v[keep]; print('kept',len(a),np.where(~keep)[0]); print('non-outlier delta',np.max(abs(v-a),axis=0))
h=-np.expm1(-np.array([.4,1.8])[:,None]*t); w=np.linalg.lstsq(h.T,tr.T,rcond=None)[0].T; Y=np.c_[y,w]; n=52
bs=[np.arange(n,len(x))]+list(np.array_split(np.arange(len(x)),5))
def scores(F,reg=0):
 out=[]
 for j in bs:
  i=np.setdiff1d(np.arange(len(x)),j); b=np.linalg.solve(F[i].T@F[i]+reg*np.eye(F.shape[1]),F[i].T@Y[i]) if reg else np.linalg.lstsq(F[i],Y[i],rcond=None)[0]; e=F[j]@b-Y[j]; out.append([abs(e[:,0]).mean(),abs(e[:,1:]@h).mean()])
 return np.round(np.r_[out[0],np.max(out[1:],axis=0)],5)
for d in [1,2,3]:
 F=D(a,d); b=np.linalg.lstsq(F,v,rcond=None)[0]; e=v-F@b; H=F@np.linalg.pinv(F); print('A degree',d,'rmse',np.sqrt((e**2).mean()),'loo mae',np.mean(abs(e/np.maximum(1e-8,1-np.diag(H))[:,None]))); print(b.round(4))
 xi=np.array([least_squares(lambda z:D(z[None],d)[0]@b-row, row,max_nfev=60).x for row in x]); print('inverse ranges',xi.min(0).round(3),xi.max(0).round(3),'fwd err',np.max(abs(D(xi,d)@b-x))); print('xi moments',xi.std(0).round(3))
 for k in [1,2,3,4]:
  for reg in ([0] if k<3 else [.001,.01,.1]): print('scores',d,k,reg,scores(D(xi,k),reg))
 if d==2:
  print('calibrated coefficients y | slow | fast',np.linalg.lstsq(D(xi,2),Y,rcond=None)[0].round(4)); np.save('/tmp/inva2.npy',xi)
# Compact residual ridges in inverse coordinates; compare the untouched last block.
for coord in [x,np.load('/tmp/inva2.npy')]:
 for j,z in enumerate(Y.T):
  A=D(coord,1); Q=D(coord,2); rng=np.random.default_rng(124)
  for key,fun in [('tanh',np.tanh),('sin',np.sin),('rat',lambda q:q/(1+q*q)),('poly3',lambda q:q**3)]:
   def f(p,ret=False):
    F=np.column_stack([Q,fun(A@p)]); c=np.linalg.lstsq(F[:n],z[:n],rcond=1e-9)[0]; e=F@c-z; return (e,c) if ret else e[:n]
   dr=np.linalg.lstsq(A[:n],z[:n],rcond=None)[0][1:]; dr/=np.linalg.norm(dr); starts=[np.r_[0,dr],np.r_[0,2*dr]]+list(rng.normal(size=(7,5))); out=min((least_squares(f,p,bounds=(-6,6),max_nfev=100) for p in starts),key=lambda r:r.cost); e,c=f(out.x,True); print('ridge',int(coord is x),j,key,np.mean(abs(e[:n])).round(5),np.mean(abs(e[n:])).round(5),out.x.round(3),round(c[-1],4))
