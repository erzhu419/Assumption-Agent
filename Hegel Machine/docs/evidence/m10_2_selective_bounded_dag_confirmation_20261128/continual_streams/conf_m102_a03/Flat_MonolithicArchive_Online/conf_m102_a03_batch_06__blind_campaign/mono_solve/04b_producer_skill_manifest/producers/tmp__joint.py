exec(open('/tmp/readout.py').read().split('for nam,a0')[0])
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
np.set_printoptions(precision=5,suppress=True,linewidth=150)
U,S,V=np.linalg.svd(T,full_matrices=False); L=U[:,:3]*S[:3]; Y=np.c_[y,L];
print('times',s['eval_times'],'telemetry',s['telemetry_vec']); print('time modes',V[:4]); print('y/trajectory correlations',np.corrcoef(Y.T));
for d in (1,2,3):
 q=af(polys(L,d)); p=q@fit(q[ix],y[ix]); print('score predicts y',d,np.mean(abs(p[val]-y[val])),np.mean(abs(p[ix]-y[ix])))
q=af(polys(w,2)); co=fit(q[ix],Y[ix]); print('quadratic coefficients',co); er=Y-q@co; print('resid corr',np.corrcoef(er.T)); print('rows [w,y,yres,modes,mode_res]',np.c_[w,y,er[:,0],L,er[:,1:]][:25])
# Profile low-dimensional nonlinearities over calibrated B outputs.
for base in (w,h):
 for deg in (1,2):
  q=af(polys(base,deg)); norm=np.maximum(base.std(0),.01); b=base/norm; 
  for kind,F in [('tanh',np.tanh),('sin',np.sin),('dead',lambda a:np.sign(a)*np.maximum(abs(a)-.35,0)**1.5),('softplus',lambda a:np.logaddexp(0,a)),('hinge',lambda a:np.maximum(0,a)),('sigmoid',lambda a:1/(1+np.exp(-np.clip(a,-60,60)))),('squarehinge',lambda a:np.maximum(0,a)**2),('abs',abs),('sat',lambda a:a/np.sqrt(1+a*a))]:
   def design(p):return np.c_[q,F(af(b)@p)]
   def res(p):
    a=design(p)[ix]; return ((a@fit(a,Y[ix])-Y[ix])/np.array([1.,1.,.1,.1])).ravel()
   rng=np.random.default_rng(88); r=min((least_squares(res,p,max_nfev=100) for p in np.r_[np.c_[np.zeros(4),np.eye(4)],rng.normal(size=(9,5))]),key=lambda r:r.cost); q1=design(r.x); pred=q1@fit(q1[ix],Y[ix]); print('ridge function',base is w,deg,kind,'val',np.mean(abs(pred[val]-Y[val]),0),'train',np.mean(abs(pred[ix]-Y[ix]),0),'param',r.x)
# Kernel family on the strongly supported panel-B features, and on B plus raw coordinates.
from scipy.spatial.distance import cdist
out=[]
for name,H in [('w',w),('h',h),('hz',np.c_[h,z*.5]),('wz',np.c_[w,z*.5])]:
 H=H/H[ix].std(0); d=cdist(H,H); q=af(polys(H,2)); q/=q[ix].std(0).clip(0.3); qk=q@q.T
 for le in (.55,.9,1.3,1.8,2.5,3.5,5.):
  a=d/le
  for kn,K in [('rbf',np.exp(-a*a/2)),('M32',(1+a)*np.exp(-a)),('M52',(1+a+a*a/3)*np.exp(-a))]:
   for lin in (0, .1, 1):
    k=K+lin*qk; e,u=eigh(k[np.ix_(ix,ix)]); pred=k[np.ix_(val,ix)]@u
    for n0 in (1e-5,.001,.014,.07):
     p=pred@((u.T@Y[ix])/(e[:,None]+n0)); er=np.mean(abs(p-Y[val]),0); out.append((er[0],np.mean(abs((p[:,1:]-L[val])@V[:3])),name,le,kn,lin,n0))
print('kernel top', sorted(out)[:10]); print('kernel state top',sorted(out,key=lambda r:r[1])[:5]);
