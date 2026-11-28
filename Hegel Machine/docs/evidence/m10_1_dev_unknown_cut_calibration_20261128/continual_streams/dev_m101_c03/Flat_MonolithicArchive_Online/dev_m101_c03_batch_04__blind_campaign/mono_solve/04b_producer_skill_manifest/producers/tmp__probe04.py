import pickle,itertools,numpy as np
from scipy.optimize import least_squares
S=pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']; fit=lambda a,b:np.linalg.lstsq(a,b,rcond=None)[0]
def basis(x,degree=2,cross=False):
    return np.column_stack([np.ones(len(x))]+[np.prod(x[:,j],axis=1) for k in range(1,degree+1) for j in (itertools.combinations(range(4),k) if cross else itertools.combinations_with_replacement(range(4),k))])
def calibrate(s,x):
    p=s['panel_a'];u,v=p['u'],p['v']; coeff=[]
    for a,b in zip(u.T,v.T):
        i,j=np.triu_indices(len(a),1); slope=np.median((b[i]-b[j])/(a[i]-a[j])); r=abs(b-a*slope-np.median(b-a*slope)); m=r<=6*np.median(r); coeff.append(fit(np.c_[np.ones(m.sum()),a[m]],b[m]))
    o,g=np.array(coeff).T; return (x-o)/g
p=S['panel_c']; u=p['u']; Q=basis(u); C=basis(u,cross=True); print('v1 coefficients\n',np.round(fit(C,p['v1']),5));print('v2 quadratic coefficients\n',np.round(fit(Q,p['v2']),3))
v=p['v2']; r2=(u*u).sum(1); print('radial relation',np.round(np.c_[r2,np.linalg.norm(v,axis=1),np.linalg.norm(Q@fit(Q,v)-v,axis=1)],3))
for label,D in [('radial',r2[:,None]),('diag',u*u),('quad',Q[:,5:]),('linquad',Q[:,1:])]:
 for typ in ('numerator','postbias'):
    def f(a):
        den=1+D@a
        if typ=='postbias': M=np.c_[np.ones(len(u)),Q[:,1:]/den[:,None],1/den];return M@fit(M,v)-v
        M=Q/den[:,None];return M@fit(M,v)-v
    res=least_squares(lambda a:f(a).ravel(),np.zeros(D.shape[1]),max_nfev=500); print(label,typ,np.sqrt(np.mean(res.fun**2)),np.round(res.x,4))
t=S['eval_times']; Z=S['train']['traj'];
def tb(r):return -np.expm1(-t[:,None]*r)
for initial in ([.2,1.5],[.6,3],[1.,2.]):
 res=least_squares(lambda r:(tb(r)@fit(tb(r),Z.T)-Z.T).ravel(),initial,bounds=(.00001,15)); print('rates',res.x,'rmse',np.sqrt(np.mean(res.fun**2)))
R=res.x; Y=np.c_[S['train']['y'],fit(tb(R),Z.T).T]; X=S['train']['x']; z=calibrate(S,X); pb=S['panel_b']; b=basis(z,1)@fit(basis(pb['u'],1),pb['v']);
vs={'x':X,'z':z,'b':b}
for k,h in list(vs.items()):
 vs[k+'c1']=basis(h,cross=True)@fit(C,p['v1']);vs[k+'c2q']=basis(h)@fit(Q,p['v2'])
for k,h in vs.items():
 for d in [1,2,3]:
    A=basis(h,d); pred=A[56:]@fit(A[:56],Y[:56]); print(k,d,np.round(abs(pred-Y[56:]).mean(0),4))
