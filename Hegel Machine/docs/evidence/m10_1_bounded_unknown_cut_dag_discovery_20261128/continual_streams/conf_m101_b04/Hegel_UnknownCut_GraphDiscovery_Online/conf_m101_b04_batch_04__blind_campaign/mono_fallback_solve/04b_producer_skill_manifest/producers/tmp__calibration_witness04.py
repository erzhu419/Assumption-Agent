import importlib.util, pickle
import numpy as np
from scipy.optimize import least_squares
spec = importlib.util.spec_from_file_location('batch_solver','/holdout_task/batch_solver.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
s = pickle.load(open('/holdout_task/public_batch.pkl','rb'))[0]['study']
p = s['panel_a']; u,v = p['u'],p['v']; a=m._calibration(p)
print('study fields:',sorted(s),'; train fields:',sorted(s['train']))
U=np.roll(u,-1,axis=1)
def cal(w,x):
    b=np.roll(x,-1,axis=1)
    return w[:,0]+w[:,1]*x+w[:,2]*b+w[:,3]*x*b+w[:,4]*x**3
clean=np.abs(cal(a,u)-v)<0.001
print('clean counts by channel:',clean.sum(0),'smallest excluded absolute residual:',np.min(np.abs(cal(a,u)-v)[~clean]))
for name,features in [('cyclic cubic',lambda x,b:np.c_[np.ones(len(x)),x,b,x*b,x**3]),('affine',lambda x,b:np.c_[np.ones(len(x)),x,b]),('cyclic quadratic',lambda x,b:np.c_[np.ones(len(x)),x,b,x*b])]:
    errors=[]
    for j in range(4):
        k=clean[:,j]; F=features(u[k,j],U[k,j]); y=v[k,j]; h=(np.linalg.qr(F)[0]**2).sum(1)
        errors.append(np.sqrt(np.mean(((y-F@m._fit(F,y))/(1-h))**2)))
    print('clean-panel leave-one-out RMSE',name,errors)
x=np.r_[s['train']['x'],s['test']['x']]; z=m._decode(x,a); b=np.roll(z,-1,axis=1); j=np.zeros((len(z),4,4)); k=np.arange(4)
j[:,k,k]=a[:,1]+a[:,3]*b+3*a[:,4]*z*z; j[:,k,(k+1)%4]=a[:,2]+a[:,3]*z
print('inverse: max absolute round-trip error',np.max(np.abs(cal(a,z)-x)), 'min Jacobian singular value',np.linalg.svd(j,compute_uv=False).min())
p=s['panel_b']; W=m._fit(m._affine(p['u']),p['v']); print('panel B affine RMSE',np.mean((m._affine(p['u'])@W-p['v'])**2)**.5)
z=m._affine(z[:72])@W; y=s['train']['y']; t=s['train']['traj']; F=m._scalar_design(z)
D=np.array([z[:,0]-z[:,2],z[:,1]-z[:,3],z[:,0]+z[:,2],z[:,1]+z[:,3]]).T
# Compare structured residual challengers on the preregistered split and a large-difference slice.
for name,hold in [('52/20',np.arange(52,72)),('large channel differences',np.argsort(np.linalg.norm(D[:,:2],axis=1))[-20:])]:
    train=np.setdiff1d(np.arange(72),hold)
    def score(P): return np.abs(y[hold]-P[hold]@m._fit(P[train],y[train])).mean()
    print(name,'scalar MAE compact / pair sum-difference squares:',score(F),score(np.c_[F,D**2]))
    # Profile one common positive denominator on fitting rows only.
    best=None
    for d in D.T:
        for c in (0.03,0.1,0.3,1.0):
            P=F/(1+c*d[:,None]**2); r=np.mean((P[train]@m._fit(P[train],y[train])-y[train])**2)
            if best is None or r<best[0]:best=(r,P)
    print(name,'best training-profiled rational MAE:',score(best[1]))
    state=m._trajectory(s['eval_times'],t[train],z[train],z[hold]); E=np.abs(state-t[hold]); print(name,'trajectory overall / worst timestamp MAE',E.mean(),E.mean(0).max())
